"""Fine-tune a HuggingFace object detection model on a FiftyOne dataset.

This operator supports:
- Any HuggingFace model compatible with ``AutoModelForObjectDetection``
- Multiple bounding-box formats (COCO, YOLO, XYXY) selectable via dropdown
- Train/val splitting via percentage or existing sample tags
- Delegated (background) execution with progress reporting
- Optional push to HuggingFace Hub
- SDK invocation via ``__call__``

The FiftyOne-to-PyTorch conversion follows the standard pattern:

1. Subclass ``fiftyone.utils.torch.GetItem`` to define what fields to
   extract (``required_keys``) and how to transform them (``__call__``).
2. Use ``field_mapping`` so the GetItem uses a generic key
   (``"detections"``) while the actual field name can vary.
3. Call ``view.to_torch(get_item)`` to produce a PyTorch Dataset.

Object detection works at the **image level** — each sample keeps all of
its detections, unlike SAM-style patch training.
"""

import logging
import os
import random

import fiftyone.operators as foo
from fiftyone.operators import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bbox format specifications
# ---------------------------------------------------------------------------

BBOX_FORMATS = {
    "coco": {
        "label": "COCO [x_min, y_min, width, height] absolute pixels",
        "description": (
            "Used by DETR, Conditional DETR, Deformable DETR, "
            "Table Transformer"
        ),
    },
    "yolo": {
        "label": "YOLO [center_x, center_y, width, height] normalized 0-1",
        "description": "Used by YOLOS, YOLO-family models",
    },
    "xyxy": {
        "label": "XYXY [x_min, y_min, x_max, y_max] absolute pixels",
        "description": "Used by Faster R-CNN and other torchvision models",
    },
}


def _fo_to_coco(bbox, img_w, img_h):
    """FiftyOne [rx, ry, rw, rh] normalised -> COCO [x, y, w, h] pixels."""
    rx, ry, rw, rh = bbox
    return [rx * img_w, ry * img_h, rw * img_w, rh * img_h]


def _fo_to_yolo(bbox, img_w, img_h):
    """FiftyOne [rx, ry, rw, rh] normalised -> YOLO [cx, cy, w, h] normalised."""
    rx, ry, rw, rh = bbox
    return [rx + rw / 2, ry + rh / 2, rw, rh]


def _fo_to_xyxy(bbox, img_w, img_h):
    """FiftyOne [rx, ry, rw, rh] normalised -> XYXY [x1, y1, x2, y2] pixels."""
    rx, ry, rw, rh = bbox
    return [rx * img_w, ry * img_h, (rx + rw) * img_w, (ry + rh) * img_h]


_BBOX_CONVERTERS = {
    "coco": _fo_to_coco,
    "yolo": _fo_to_yolo,
    "xyxy": _fo_to_xyxy,
}


def _compute_area(bbox, bbox_format, img_w, img_h):
    """Return the area of a bbox in pixels."""
    if bbox_format == "coco":
        # [x, y, w, h] absolute
        return bbox[2] * bbox[3]
    elif bbox_format == "yolo":
        # [cx, cy, w, h] normalised -> area in pixels
        return bbox[2] * img_w * bbox[3] * img_h
    elif bbox_format == "xyxy":
        # [x1, y1, x2, y2] absolute
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    raise ValueError(f"Unsupported bbox format: {bbox_format!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_detection_fields(ctx):
    """Return the names of all Detections label fields on the dataset."""
    fields = []
    for name, field in ctx.dataset.get_field_schema().items():
        if hasattr(field, "document_type"):
            fqn = (
                f"{field.document_type.__module__}"
                f".{field.document_type.__name__}"
            )
            if fqn == "fiftyone.core.labels.Detections":
                fields.append(name)
    return fields


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class FinetuneDetection(foo.Operator):
    """Fine-tune a HuggingFace object-detection model."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="finetune_detection",
            label="Fine-tune Detection Model",
            description=(
                "Fine-tune any HuggingFace AutoModelForObjectDetection model "
                "on your FiftyOne dataset"
            ),
            icon="/icons/adjust-svgrepo-com.svg",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    # ---- dynamic input form -----------------------------------------------

    def resolve_input(self, ctx):
        inputs = types.Object()

        # -- Label field (auto-detect Detections fields) --------------------
        detection_fields = _get_detection_fields(ctx)

        if not detection_fields:
            inputs.view(
                "no_fields_warning",
                types.Warning(
                    label="No detection fields found",
                    description=(
                        "Your dataset needs at least one field of type "
                        "fiftyone.core.labels.Detections"
                    ),
                ),
            )
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Detection"),
            )

        inputs.enum(
            "label_field",
            values=detection_fields,
            required=True,
            label="Label Field",
            description="The Detections field to train on",
        )

        label_field = ctx.params.get("label_field")
        if not label_field:
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Detection"),
            )

        # Show a quick summary of the classes
        classes = ctx.dataset.distinct(f"{label_field}.detections.label")
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

        # -- Bbox format ----------------------------------------------------
        bbox_choices = types.DropdownView()
        for fmt_id, fmt_info in BBOX_FORMATS.items():
            bbox_choices.add_choice(
                fmt_id,
                label=fmt_info["label"],
                description=fmt_info["description"],
            )

        inputs.enum(
            "bbox_format",
            bbox_choices.values(),
            required=True,
            label="Bounding Box Format",
            description=(
                "The bbox format your model expects. "
                "DETR-family models use COCO; YOLOS uses YOLO; "
                "Faster R-CNN uses XYXY."
            ),
            default="coco",
            view=bbox_choices,
        )

        # -- Model name -----------------------------------------------------
        inputs.str(
            "model_name",
            required=True,
            label="Model Name",
            description=(
                "Any HuggingFace AutoModelForObjectDetection model ID. "
                "Examples: facebook/detr-resnet-50, "
                "microsoft/conditional-detr-resnet-50, "
                "hustvl/yolos-small"
            ),
            default="facebook/detr-resnet-50",
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
                description=(
                    "Fraction of data used for training "
                    "(rest is validation)"
                ),
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
            default=4,
            label="Batch Size",
            description="Per-device training and evaluation batch size",
        )
        inputs.float(
            "learning_rate",
            default=1e-5,
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
            "output_parent_dir",
            required=True,
            label="Output Parent Directory",
            description=(
                "Parent directory where a new model folder "
                "will be created"
            ),
            view=file_explorer,
        )
        inputs.str(
            "model_dir_name",
            required=True,
            label="Model Folder Name",
            description=(
                "Name for the new model directory "
                "(will be created if it doesn't exist)"
            ),
            default="finetuned_detection_model",
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
                description=(
                    "Repository ID on the Hub, "
                    "e.g. your-username/model-name"
                ),
            )
            inputs.bool(
                "hub_private",
                default=True,
                label="Private Repository",
                description="Make the Hub repository private",
                view=types.CheckboxView(),
            )

        return types.Property(
            inputs,
            view=types.View(label="Fine-tune Detection Model"),
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
        ``_run_training``.

        The ``StopIteration`` guard is required because FiftyOne's
        delegated executor wraps ``execute()`` in an asyncio Future,
        and ``StopIteration`` cannot be set as a Future exception.
        A ``StopIteration`` can leak from the HuggingFace Trainer's
        internal DataLoader iteration at epoch boundaries.
        """
        hub_token = (
            ctx.secrets.get("HF_TOKEN")
            if ctx.params.get("push_to_hub")
            else None
        )

        # Build output_dir from parent + model folder name.
        # FileExplorerView returns {"absolute_path": "..."} from the App;
        # SDK callers pass a plain string via __call__.
        raw_parent = ctx.params.get("output_parent_dir", ".")
        if isinstance(raw_parent, dict):
            parent_dir = raw_parent.get("absolute_path", ".")
        else:
            parent_dir = raw_parent

        model_dir_name = ctx.params.get(
            "model_dir_name", "finetuned_detection_model",
        )
        output_dir = os.path.join(parent_dir, model_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            return self._run_training(
                target_view=ctx.target_view(),
                label_field=ctx.params["label_field"],
                bbox_format=ctx.params.get("bbox_format", "coco"),
                model_name=ctx.params["model_name"],
                split_strategy=ctx.params.get("split_strategy", "percentage"),
                train_split=ctx.params.get("train_split", 0.8),
                train_tag=ctx.params.get("train_tag", "train"),
                val_tag=ctx.params.get("val_tag", "val"),
                num_epochs=ctx.params.get("num_epochs", 3),
                batch_size=ctx.params.get("batch_size", 4),
                learning_rate=ctx.params.get("learning_rate", 1e-5),
                output_dir=output_dir,
                push_to_hub=ctx.params.get("push_to_hub", False),
                hub_model_id=ctx.params.get("hub_model_id"),
                hub_private=ctx.params.get("hub_private", True),
                hub_token=hub_token,
            )
        except StopIteration as e:
            raise RuntimeError(
                "Training iteration ended unexpectedly. This can "
                "happen when the HuggingFace Trainer's DataLoader "
                "raises StopIteration across an async boundary."
            ) from e

    # ---- core training logic ----------------------------------------------

    def _run_training(
        self,
        target_view,
        label_field,
        bbox_format,
        model_name,
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        output_dir="./finetuned_detection_model",
        push_to_hub=False,
        hub_model_id=None,
        hub_private=True,
        hub_token=None,
    ):
        """Run the full object-detection fine-tuning pipeline.

        This method contains all the training logic and is called by both
        ``execute()`` (App / delegated path) and ``__call__()`` (SDK path).
        """
        # Force single-GPU training to avoid DataParallel.  The HF Trainer
        # wraps models in nn.DataParallel when it sees multiple GPUs, which
        # breaks models whose forward() calls self.device (e.g. DETR) and
        # is slow in general.  For proper multi-GPU, use accelerate/DDP.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        import json

        import numpy as np
        import torch
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            Trainer,
            TrainingArguments,
        )

        output_dir = os.path.expanduser(output_dir)
        convert_fn = _BBOX_CONVERTERS[bbox_format]

        # -- 1. Build class mapping -----------------------------------------
        logger.info("Building class mapping...")

        classes = target_view.distinct(f"{label_field}.detections.label")
        classes = sorted([c for c in classes if c is not None])
        label2id = {c: i for i, c in enumerate(classes)}
        id2label = {i: c for c, i in label2id.items()}
        num_labels = len(classes)

        if num_labels < 1:
            raise ValueError(
                f"Need at least 1 class to fine-tune, but found "
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

        # Filter out samples with no detections
        train_view = train_view.exists(f"{label_field}.detections")
        val_view = val_view.exists(f"{label_field}.detections")

        # Ensure metadata (width/height) is populated — needed for
        # bounding-box coordinate conversion.
        print("Computing metadata (if needed)...")
        train_view.compute_metadata(skip_failures=False, overwrite=False)
        val_view.compute_metadata(skip_failures=False, overwrite=False)

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
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # -- 4. Build PyTorch datasets via GetItem + to_torch() -------------
        #
        # Object detection works at the IMAGE level (not patches).
        # Each sample keeps all of its detections.  The GetItem converts
        # FiftyOne Detections into the annotation-dict format that the
        # HuggingFace image processor expects.
        # -----------------------------------------------------------------
        print("Building datasets...")

        class _DetectionGetItem(GetItem):
            """Bridge from FiftyOne Detections to HF detection format.

            Each sample is transformed into::

                {
                    "pixel_values": Tensor[C, H, W],
                    "pixel_mask":   Tensor[H, W],
                    "labels":       dict with class_labels, boxes, ...
                }
            """

            def __init__(
                self, proc, l2id, bbox_fmt, cvt_fn, field_mapping=None,
            ):
                self.proc = proc
                self.l2id = l2id
                self.bbox_fmt = bbox_fmt
                self.cvt_fn = cvt_fn
                super().__init__(field_mapping=field_mapping)

            @property
            def required_keys(self):
                return ["filepath", "detections", "metadata"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                detections_obj = d.get("detections")
                metadata = d.get("metadata")

                if metadata is not None:
                    img_w = metadata.width
                    img_h = metadata.height
                else:
                    img_w, img_h = image.size

                # Build COCO-style annotation list from all detections
                annotations = []
                for det in detections_obj.detections:
                    bbox = self.cvt_fn(det.bounding_box, img_w, img_h)
                    area = _compute_area(
                        bbox, self.bbox_fmt, img_w, img_h,
                    )
                    annotations.append(
                        {
                            "bbox": bbox,
                            "category_id": self.l2id[det.label],
                            "area": area,
                            "iscrowd": 0,
                        }
                    )

                target = {
                    "image_id": 0,
                    "annotations": annotations,
                }

                # The processor converts the image + annotations into the
                # format expected by the model (pixel_values, labels, etc.)
                inputs = self.proc(
                    images=image,
                    annotations=[target],
                    return_tensors="pt",
                )

                # Squeeze batch dim added by the processor
                result = {}
                result["pixel_values"] = inputs["pixel_values"].squeeze(0)
                if "pixel_mask" in inputs:
                    result["pixel_mask"] = inputs["pixel_mask"].squeeze(0)
                # labels is a list of one dict — unwrap it
                result["labels"] = inputs["labels"][0]

                return result

        field_mapping = {"detections": label_field}

        train_getter = _DetectionGetItem(
            processor,
            label2id,
            bbox_format,
            convert_fn,
            field_mapping=field_mapping,
        )
        val_getter = _DetectionGetItem(
            processor,
            label2id,
            bbox_format,
            convert_fn,
            field_mapping=field_mapping,
        )

        train_dataset = train_view.to_torch(train_getter)
        val_dataset = val_view.to_torch(val_getter)

        # -- 5. Configure Trainer -------------------------------------------
        print("Configuring trainer...")

        # Custom collate: images may differ in spatial size (the processor
        # preserves aspect ratio), so we pad every image in the batch to
        # the largest H and W and build a pixel_mask that marks real pixels.
        def _collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]

            # Determine max spatial dims across the batch
            max_h = max(pv.shape[-2] for pv in pixel_values)
            max_w = max(pv.shape[-1] for pv in pixel_values)

            padded_images = []
            padded_masks = []
            for pv in pixel_values:
                c, h, w = pv.shape
                pad_bottom = max_h - h
                pad_right = max_w - w
                # F.pad expects (left, right, top, bottom)
                padded = torch.nn.functional.pad(
                    pv, (0, pad_right, 0, pad_bottom), value=0,
                )
                padded_images.append(padded)
                # pixel_mask: 1 for real pixels, 0 for padding
                mask = torch.zeros(max_h, max_w, dtype=torch.long)
                mask[:h, :w] = 1
                padded_masks.append(mask)

            return {
                "pixel_values": torch.stack(padded_images),
                "pixel_mask": torch.stack(padded_masks),
                "labels": [item["labels"] for item in batch],
            }

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id if push_to_hub else None,
            hub_private_repo=hub_private,
            hub_token=hub_token,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=_collate_fn,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=processor,
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
                    "bbox_format": bbox_format,
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
            "bbox_format": bbox_format,
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
            f"| **Bbox format** | {results.get('bbox_format', 'N/A')} |\n"
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
        bbox_format="coco",
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        output_dir="./finetuned_detection_model",
        push_to_hub=False,
        hub_model_id=None,
        hub_private=True,
        delegate=False,
    ):
        """Execute detection fine-tuning via the SDK through the operator
        framework.

        In a notebook you must ``await`` the result::

            result = await detection_trainer(
                dataset,
                label_field="ground_truth",
                model_name="facebook/detr-resnet-50",
                bbox_format="coco",
            )

        Args:
            sample_collection: a
                :class:`fiftyone.core.collections.SampleCollection`
                (Dataset or DatasetView) to train on.
            label_field: the name of the ``Detections`` label field.
            model_name: any HuggingFace ``AutoModelForObjectDetection``
                model ID.
            bbox_format: ``"coco"``, ``"yolo"``, or ``"xyxy"``.
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
            hub_private: whether the Hub repository should be private
                (default ``True``).
            delegate: whether to schedule as a delegated operation
                (default ``False`` — runs immediately). Set to ``True``
                to run in the background via an orchestrator.

        Returns:
            an :class:`ExecutionResult`, or an ``asyncio.Task`` in
            notebook contexts.
        """
        # Split output_dir into parent + folder name so the params
        # match what execute() reads from the App form.
        abs_path = os.path.abspath(output_dir)
        output_parent_dir = {
            "absolute_path": os.path.dirname(abs_path),
        }
        model_dir_name = os.path.basename(abs_path)

        params = dict(
            label_field=label_field,
            model_name=model_name,
            bbox_format=bbox_format,
            split_strategy=split_strategy,
            train_split=train_split,
            train_tag=train_tag,
            val_tag=val_tag,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_parent_dir=output_parent_dir,
            model_dir_name=model_dir_name,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_private=hub_private,
        )

        ctx = dict(view=sample_collection.view())
        return foo.execute_operator(
            self.uri,
            ctx,
            params=params,
            request_delegation=delegate,
        )
