"""Fine-tune a HuggingFace image classification model on a FiftyOne dataset.

This operator supports:
- Any HuggingFace model compatible with ``pipeline("image-classification")``
- Train/val splitting via percentage or existing sample tags
- Delegated (background) execution with progress reporting
- Optional push to HuggingFace Hub
- SDK invocation via ``__call__``

The FiftyOne-to-PyTorch conversion follows the standard pattern:

1. Subclass ``fiftyone.utils.torch.GetItem`` to define what fields to
   extract (``required_keys``) and how to transform them (``__call__``).
2. Use ``field_mapping`` so the GetItem uses a generic key
   (``"classification"``) while the actual field name can vary.
3. Call ``view.to_torch(get_item)`` to produce a PyTorch Dataset.
"""

import logging
import os
import random

import fiftyone.operators as foo
from fiftyone.operators import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_classification_fields(ctx):
    """Return the names of all Classification label fields on the dataset."""
    fields = []
    for name, field in ctx.dataset.get_field_schema().items():
        if hasattr(field, "document_type"):
            fqn = (
                f"{field.document_type.__module__}"
                f".{field.document_type.__name__}"
            )
            if fqn == "fiftyone.core.labels.Classification":
                fields.append(name)
    return fields


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class FinetuneClassification(foo.Operator):
    """Fine-tune a HuggingFace image-classification model."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="finetune_classification",
            label="Fine-tune Classification Model",
            description=(
                "Fine-tune any HuggingFace image-classification model "
                "on your FiftyOne dataset"
            ),
            dynamic=True,
            # Allow both paths: immediate for SDK, delegated for App
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    # ---- placement --------------------------------------------------------

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Fine-tune Classification",
                icon="/icons/adjust-svgrepo-com.svg",
                prompt=True,
            ),
        )

    # ---- dynamic input form -----------------------------------------------

    def resolve_input(self, ctx):
        inputs = types.Object()

        # -- Label field (auto-detect Classification fields) ----------------
        classification_fields = _get_classification_fields(ctx)

        if not classification_fields:
            inputs.view(
                "no_fields_warning",
                types.Warning(
                    label="No classification fields found",
                    description=(
                        "Your dataset needs at least one field of type "
                        "fiftyone.core.labels.Classification"
                    ),
                ),
            )
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Classification"),
            )

        inputs.enum(
            "label_field",
            values=classification_fields,
            required=True,
            label="Label Field",
            description="The Classification field to train on",
        )

        label_field = ctx.params.get("label_field")
        if not label_field:
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Classification"),
            )

        # Show a quick summary of the classes
        classes = ctx.dataset.distinct(f"{label_field}.label")
        classes = sorted([c for c in classes if c is not None])
        num_classes = len(classes)
        preview = ", ".join(classes[:10])
        if num_classes > 10:
            preview += f" ... (+{num_classes - 10} more)"

        inputs.view(
            "class_info",
            types.Notice(
                label=f"Found {num_classes} classes: {preview}",
            ),
        )

        # -- Model name (free-text; any pipeline-compatible model) ----------
        inputs.str(
            "model_name",
            required=True,
            label="Model Name",
            description=(
                "Any HuggingFace `AutoModelForImageClassification` model ID that supports "
                "image-classification pipelines. "
                "Examples: google/vit-base-patch16-224, "
                "microsoft/resnet-50, facebook/convnext-tiny-224"
            ),
            default="google/vit-base-patch16-224",
        )

        # -- View target (dataset vs current view) --------------------------
        inputs.view_target(ctx)

        # -- Split strategy -------------------------------------------------
        inputs.enum(
            "split_strategy",
            values=["percentage", "tags"],
            label="Split Strategy",
            description="How to split data into train / validation sets",
            default="percentage",
            view=types.RadioGroup(),
        )

        split_strategy = ctx.params.get("split_strategy", "percentage")

        if split_strategy == "percentage":
            inputs.float(
                "train_split",
                default=0.8,
                label="Train Split",
                description="Fraction of data used for training (rest is validation)",
            )
        else:
            inputs.str(
                "train_tag",
                default="train",
                label="Train Tag",
                description="Sample tag that identifies training samples",
            )
            inputs.str(
                "val_tag",
                default="val",
                label="Validation Tag",
                description="Sample tag that identifies validation samples",
            )

        # -- Training hyper-parameters --------------------------------------
        inputs.int(
            "num_epochs",
            default=3,
            label="Epochs",
            description="Number of training epochs",
        )
        inputs.int(
            "batch_size",
            default=8,
            label="Batch Size",
            description="Per-device training and evaluation batch size",
        )
        inputs.float(
            "learning_rate",
            default=5e-5,
            label="Learning Rate",
            description="Peak learning rate for the optimizer",
        )

        # -- Output ---------------------------------------------------------
        file_explorer = types.FileExplorerView(
            choose_dir=True,
            button_label="Choose a directory...",
            choose_button_label="Accept",
        )
        inputs.file(
            "output_dir",
            required=True,
            label="Output Directory",
            description="Directory where the fine-tuned model will be saved",
            view=file_explorer,
        )

        inputs.bool(
            "push_to_hub",
            default=False,
            label="Push to HuggingFace Hub",
            description=(
                "Upload the model to HuggingFace Hub after training. "
                "Requires a valid HF_TOKEN secret or hf auth login."
            ),
            view=types.CheckboxView(),
        )

        if ctx.params.get("push_to_hub"):
            inputs.str(
                "hub_model_id",
                required=True,
                label="Hub Model ID",
                description="Repository ID on the Hub, e.g. your-username/model-name",
            )

        return types.Property(
            inputs,
            view=types.View(label="Fine-tune Classification Model"),
        )

    # ---- delegation -------------------------------------------------------

    def resolve_delegation(self, ctx):
        """Always delegate from the App — training is expensive.

        SDK callers control delegation via ``request_delegation`` in the
        execution context (passed through the ``delegate`` kwarg of
        ``__call__``).
        """
        return True

    # ---- execution --------------------------------------------------------

    def execute(self, ctx):
        """App execution path — extracts params from ctx, delegates to
        ``_run_training``."""
        hub_token = (
            ctx.secrets.get("HF_TOKEN")
            if ctx.params.get("push_to_hub")
            else None
        )

        # FileExplorerView returns {"absolute_path": "..."} from the App;
        # SDK callers pass a plain string.
        raw_output_dir = ctx.params.get("output_dir", "./finetuned_model")
        if isinstance(raw_output_dir, dict):
            output_dir = raw_output_dir.get(
                "absolute_path", "./finetuned_model"
            )
        else:
            output_dir = raw_output_dir

        return self._run_training(
            target_view=ctx.target_view(),
            label_field=ctx.params["label_field"],
            model_name=ctx.params["model_name"],
            split_strategy=ctx.params.get("split_strategy", "percentage"),
            train_split=ctx.params.get("train_split", 0.8),
            train_tag=ctx.params.get("train_tag", "train"),
            val_tag=ctx.params.get("val_tag", "val"),
            num_epochs=ctx.params.get("num_epochs", 3),
            batch_size=ctx.params.get("batch_size", 8),
            learning_rate=ctx.params.get("learning_rate", 5e-5),
            output_dir=output_dir,
            push_to_hub=ctx.params.get("push_to_hub", False),
            hub_model_id=ctx.params.get("hub_model_id"),
            hub_token=hub_token,
        )

    # ---- core training logic ----------------------------------------------

    def _run_training(
        self,
        target_view,
        label_field,
        model_name,
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        output_dir="./finetuned_model",
        push_to_hub=False,
        hub_model_id=None,
        hub_token=None,
    ):
        """Run the full fine-tuning pipeline.

        This method contains all the training logic and is called by both
        ``execute()`` (App / delegated path) and ``__call__()`` (SDK path).
        """
        # Heavy imports deferred to execution time
        import json

        import numpy as np
        import torch
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
        )

        output_dir = os.path.expanduser(output_dir)

        # -- 1. Build class mapping -----------------------------------------
        logger.info("Building class mapping...")

        classes = target_view.distinct(f"{label_field}.label")
        classes = sorted([c for c in classes if c is not None])
        label2id = {c: i for i, c in enumerate(classes)}
        id2label = {i: c for c, i in label2id.items()}
        num_labels = len(classes)

        if num_labels < 2:
            raise ValueError(
                f"Need at least 2 classes to fine-tune, but found "
                f"{num_labels} in field '{label_field}'"
            )

        logger.info("Classes (%d): %s", num_labels, classes)

        # -- 2. Split into train / val --------------------------------------
        logger.info("Splitting dataset (strategy=%s)...", split_strategy)

        if split_strategy == "tags":
            train_view = target_view.match_tags(train_tag)
            val_view = target_view.match_tags(val_tag)
        else:
            sample_ids = list(target_view.values("id"))
            random.shuffle(sample_ids)
            n_train = int(len(sample_ids) * train_split)
            train_view = target_view.select(sample_ids[:n_train])
            val_view = target_view.select(sample_ids[n_train:])

        # Filter out samples that have no label (None)
        train_view = train_view.exists(f"{label_field}.label")
        val_view = val_view.exists(f"{label_field}.label")

        n_train = len(train_view)
        n_val = len(val_view)

        logger.info("Train samples: %d  |  Val samples: %d", n_train, n_val)
        print(f"Train samples: {n_train}  |  Val samples: {n_val}")

        if n_train == 0:
            raise ValueError(
                "No training samples found. "
                "Check your split strategy / tags."
            )
        if n_val == 0:
            raise ValueError(
                "No validation samples found. "
                "Check your split strategy / tags."
            )

        # -- 3. Load processor and model ------------------------------------
        print(f"Loading processor for {model_name}...")
        processor = AutoImageProcessor.from_pretrained(model_name)

        print(f"Loading model {model_name}...")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # -- 4. Build PyTorch datasets via GetItem + to_torch() -------------
        #
        # This follows FiftyOne's standard pattern:
        #   1. Subclass GetItem — declare required_keys, implement __call__
        #   2. Use field_mapping so the generic key "classification" resolves
        #      to whatever the real label field is (e.g. "ground_truth")
        #   3. Call view.to_torch(get_item) to get a PyTorch Dataset
        # -----------------------------------------------------------------
        print("Building datasets...")

        class _ClassificationGetItem(GetItem):
            """Bridge from FiftyOne Classification samples to HF format.

            Each sample is transformed into the dict that HuggingFace
            ``Trainer`` expects::

                {
                    "pixel_values": Tensor[C, H, W],
                    "labels":       Tensor (scalar, long),
                }
            """

            def __init__(self, proc, l2id, field_mapping=None):
                self.proc = proc
                self.l2id = l2id
                super().__init__(field_mapping=field_mapping)

            @property
            def required_keys(self):
                # "classification" is a generic key — field_mapping maps it
                # to the real field name (e.g. "ground_truth").
                return ["filepath", "classification"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                classification = d.get("classification")

                inputs = self.proc(images=image, return_tensors="pt")
                # Squeeze the batch dim the processor adds
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(
                    self.l2id[classification.label],
                    dtype=torch.long,
                )
                return inputs

        field_mapping = {"classification": label_field}

        train_getter = _ClassificationGetItem(
            processor, label2id, field_mapping=field_mapping,
        )
        val_getter = _ClassificationGetItem(
            processor, label2id, field_mapping=field_mapping,
        )

        train_dataset = train_view.to_torch(train_getter)
        val_dataset = val_view.to_torch(val_getter)

        # -- 5. Configure Trainer -------------------------------------------
        print("Configuring trainer...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id if push_to_hub else None,
            hub_token=hub_token,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
        )

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = (preds == labels).mean()
            return {"accuracy": float(acc)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics,
        )

        # -- 6. Train -------------------------------------------------------
        print(
            f"Training {model_name} for {num_epochs} epoch(s) "
            f"on {n_train} samples..."
        )

        train_result = trainer.train()

        # -- 7. Save model & processor -------------------------------------
        print(f"Saving model to {output_dir}...")

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        # Persist the class mapping alongside the model for convenience
        mapping_path = os.path.join(output_dir, "class_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "label2id": label2id,
                    "id2label": {
                        str(k): v for k, v in id2label.items()
                    },
                },
                f,
                indent=2,
            )

        # -- 8. (Optional) push to Hub -------------------------------------
        if push_to_hub:
            print("Pushing to HuggingFace Hub...")
            trainer.push_to_hub()

        # -- Return results -------------------------------------------------
        metrics = train_result.metrics
        results = {
            "model_name": model_name,
            "output_dir": output_dir,
            "num_classes": num_labels,
            "train_samples": n_train,
            "val_samples": n_val,
            "train_loss": metrics.get("train_loss"),
            "epochs_completed": metrics.get("epoch"),
        }

        print("Training complete!")
        logger.info("Results: %s", results)

        return results

    # ---- output display ---------------------------------------------------

    def resolve_output(self, ctx):
        outputs = types.Object()
        results = ctx.results or {}

        train_loss = results.get("train_loss")
        loss_str = (
            f"{train_loss:.4f}" if train_loss is not None else "N/A"
        )
        epochs = results.get("epochs_completed")
        epochs_str = str(int(epochs)) if epochs is not None else "N/A"

        summary = (
            "### Training Complete\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| **Model** | `{results.get('model_name', 'N/A')}` |\n"
            f"| **Saved to** | `{results.get('output_dir', 'N/A')}` |\n"
            f"| **Classes** | {results.get('num_classes', 'N/A')} |\n"
            f"| **Train samples** | {results.get('train_samples', 'N/A')} |\n"
            f"| **Val samples** | {results.get('val_samples', 'N/A')} |\n"
            f"| **Final train loss** | {loss_str} |\n"
            f"| **Epochs** | {epochs_str} |\n"
        )

        outputs.str(
            "summary",
            label="Summary",
            view=types.MarkdownView(),
            default=summary,
        )

        return types.Property(
            outputs,
            view=types.View(label="Fine-tuning Complete!"),
        )

    # ---- SDK support via __call__ -----------------------------------------

    def __call__(
        self,
        sample_collection,
        label_field,
        model_name,
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        output_dir="./finetuned_model",
        push_to_hub=False,
        hub_model_id=None,
        delegate=False,
    ):
        """Execute fine-tuning via the SDK through the operator framework.

        In a notebook you must ``await`` the result::

            result = await classification_trainer(
                dataset,
                label_field="ground_truth",
                model_name="google/vit-base-patch16-224",
            )

        Args:
            sample_collection: a
                :class:`fiftyone.core.collections.SampleCollection`
                (Dataset or DatasetView) to train on.
            label_field: the name of the ``Classification`` label field.
            model_name: any HuggingFace model ID that supports
                ``image-classification`` pipelines.
            split_strategy: ``"percentage"`` or ``"tags"``.
            train_split: fraction of data for training (used when
                ``split_strategy="percentage"``).
            train_tag: sample tag for training data (used when
                ``split_strategy="tags"``).
            val_tag: sample tag for validation data (used when
                ``split_strategy="tags"``).
            num_epochs: number of training epochs.
            batch_size: per-device batch size.
            learning_rate: peak learning rate.
            output_dir: local directory for saving the model.
            push_to_hub: whether to push to HuggingFace Hub.
            hub_model_id: Hub repository ID (required when *push_to_hub*
                is ``True``).
            delegate: whether to schedule as a delegated operation
                (default ``False`` — runs immediately). Set to ``True``
                to run in the background via an orchestrator.

        Returns:
            an :class:`ExecutionResult`, or an ``asyncio.Task`` in
            notebook contexts.
        """
        # inputs.file() expects {"absolute_path": "..."} — wrap plain
        # strings so the framework's validator doesn't choke.
        if isinstance(output_dir, str):
            output_dir = {"absolute_path": os.path.abspath(output_dir)}

        params = dict(
            label_field=label_field,
            model_name=model_name,
            split_strategy=split_strategy,
            train_split=train_split,
            train_tag=train_tag,
            val_tag=val_tag,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )

        ctx = dict(view=sample_collection.view())
        return foo.execute_operator(
            self.uri,
            ctx,
            params=params,
            request_delegation=delegate,
        )
