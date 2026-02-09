"""Fine-tune a HuggingFace semantic segmentation model on a FiftyOne dataset.

This operator supports:
- Any HuggingFace model compatible with ``AutoModelForSemanticSegmentation``
- Train/val splitting via percentage or existing sample tags
- Delegated (background) execution with progress reporting
- Optional push to HuggingFace Hub
- SDK invocation via ``__call__``

Supported label field types
---------------------------

**``fiftyone.core.labels.Segmentation``** — semantic masks stored as:

- *In-memory* ``mask`` arrays (uint8 ``(H, W)`` of class indices) stored
  directly in the database.
- *On-disk* images referenced by ``mask_path`` (added in FiftyOne 0.19).
  Accepted on-disk formats:

  - 2-D 8-bit or 16-bit grayscale (pixel values are class indices).
  - Palette-mode PNGs (e.g. Pascal VOC) — pixel values are class indices.
  - 3-D 8-bit RGB masks — each unique colour is mapped to a contiguous
    class index automatically (with a log warning so you can verify).

**``fiftyone.core.labels.Detections``** — instance segmentation stored as
``fo.Detection`` objects with per-instance ``mask`` arrays inside an
``fo.Detections`` field.  At training time, the per-instance masks are
composited into a single semantic mask (``(H, W)`` of class indices) using
the detection ``label`` values.  Detections without a ``mask`` fall back to
bounding-box filling.  A ``background`` class at index 0 is added
automatically.

FiftyOne-to-PyTorch conversion
-------------------------------

1. Subclass ``fiftyone.utils.torch.GetItem`` to define what fields to
   extract (``required_keys``) and how to transform them (``__call__``).
2. Use ``field_mapping`` so the GetItem uses a generic key
   (``"label_data"``) while the actual field name can vary.
3. Call ``view.to_torch(get_item)`` to produce a PyTorch Dataset.

Semantic segmentation works at the **image level** — each sample has a
per-pixel mask where each pixel value is a class index.
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


def _get_segmentation_compatible_fields(ctx):
    """Return fields suitable for semantic-segmentation training.

    Detects both ``fiftyone.core.labels.Segmentation`` fields (semantic
    masks) and ``fiftyone.core.labels.Detections`` fields (instance
    masks that will be composited into semantic masks during training).

    Returns:
        dict: mapping of field name to label-type string
        (``"Segmentation"`` or ``"Detections"``).
    """
    fields = {}
    for name, field in ctx.dataset.get_field_schema().items():
        if hasattr(field, "document_type"):
            fqn = (
                f"{field.document_type.__module__}"
                f".{field.document_type.__name__}"
            )
            if fqn == "fiftyone.core.labels.Segmentation":
                fields[name] = "Segmentation"
            elif fqn == "fiftyone.core.labels.Detections":
                fields[name] = "Detections"
    return fields


def _get_field_type(dataset, label_field):
    """Determine the label type for a given field name.

    Returns ``"Segmentation"``, ``"Detections"``, or ``None``.
    """
    schema = dataset.get_field_schema()
    field = schema.get(label_field)
    if field is None or not hasattr(field, "document_type"):
        return None
    fqn = (
        f"{field.document_type.__module__}"
        f".{field.document_type.__name__}"
    )
    if fqn == "fiftyone.core.labels.Segmentation":
        return "Segmentation"
    if fqn == "fiftyone.core.labels.Detections":
        return "Detections"
    return None


def _get_mask_targets(ctx, label_field):
    """Return the mask targets dict for the given field.

    FiftyOne stores class-to-pixel-value mappings in either
    ``dataset.mask_targets[field]`` or ``dataset.default_mask_targets``.
    Returns ``None`` if no mapping is found.
    """
    mask_targets = None

    if ctx.dataset.mask_targets:
        mask_targets = ctx.dataset.mask_targets.get(label_field)

    if mask_targets is None:
        mask_targets = ctx.dataset.default_mask_targets or None

    return mask_targets


def _load_mask_from_segmentation(seg_obj):
    """Extract the class-index mask array from a ``Segmentation`` label.

    Handles both in-memory masks (``seg_obj.mask``) and on-disk masks
    (``seg_obj.mask_path``).  Multi-channel masks loaded from disk are
    reduced to a single-channel class-index array.

    Returns:
        numpy.ndarray: 2-D array of integer class indices ``(H, W)``.

    Raises:
        ValueError: if neither ``mask`` nor ``mask_path`` is available.
    """
    import numpy as np
    from PIL import Image as _PILImage

    mask = seg_obj.mask
    if mask is not None:
        if mask.ndim == 3:
            mask = _collapse_multichannel_mask(mask)
        return mask

    mask_path = getattr(seg_obj, "mask_path", None)
    if mask_path is None:
        raise ValueError(
            "Segmentation label has neither a 'mask' array nor a "
            "'mask_path'.  Cannot load mask data."
        )

    mask_img = _PILImage.open(mask_path)

    # Palette-mode PNGs (e.g. Pascal VOC) — pixel values are already
    # class indices; do NOT convert to RGB.
    if mask_img.mode == "P":
        return np.array(mask_img)

    mask = np.array(mask_img)

    if mask.ndim == 2:
        return mask

    return _collapse_multichannel_mask(mask)


def _collapse_multichannel_mask(mask):
    """Reduce a 3-D ``(H, W, C)`` mask to 2-D class indices.

    Strategy:
      1. Single channel → squeeze.
      2. All RGB channels identical → use the first channel.
      3. Otherwise → encode each unique RGB colour as a contiguous
         integer index.  (A warning is logged so the caller can verify
         the mapping is consistent with their class labels.)
    """
    import numpy as np

    if mask.shape[2] == 1:
        return mask[:, :, 0]

    rgb = mask[:, :, :3]

    if (
        np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
    ):
        return rgb[:, :, 0]

    # Encode each unique colour as a flat 32-bit int
    flat = (
        rgb[:, :, 0].astype(np.int32) * 65536
        + rgb[:, :, 1].astype(np.int32) * 256
        + rgb[:, :, 2].astype(np.int32)
    )
    unique_vals, inverse = np.unique(flat, return_inverse=True)
    logger.warning(
        "RGB segmentation mask detected with %d unique colours.  "
        "Mapping to contiguous class indices 0..%d.  Verify that "
        "this is consistent with your class mapping.",
        len(unique_vals),
        len(unique_vals) - 1,
    )
    return inverse.reshape(mask.shape[:2]).astype(np.int64)


def _detections_to_semantic_mask(detections, image_hw, label2id):
    """Composite a list of ``Detection`` objects into a semantic mask.

    Each detection's per-instance mask (or bounding box, if no mask is
    present) is painted onto a canvas at the class index looked up
    from *label2id*.  Later detections overwrite earlier ones at
    overlapping pixels.

    Args:
        detections: list of ``fiftyone.core.labels.Detection``.
        image_hw: ``(H, W)`` of the full image.
        label2id: ``dict`` mapping class name → integer index.

    Returns:
        numpy.ndarray: ``(H, W)`` int64 array of class indices.
            Index 0 is background (no detection coverage).
    """
    import numpy as np
    from PIL import Image as _PILImage

    h, w = image_hw
    canvas = np.zeros((h, w), dtype=np.int64)

    for det in detections:
        cls_idx = label2id.get(det.label)
        if cls_idx is None:
            continue

        x, y, bw, bh = det.bounding_box  # relative coords [0, 1]
        x1 = max(int(round(x * w)), 0)
        y1 = max(int(round(y * h)), 0)
        x2 = min(int(round((x + bw) * w)), w)
        y2 = min(int(round((y + bh) * h)), h)

        box_h, box_w = y2 - y1, x2 - x1
        if box_h <= 0 or box_w <= 0:
            continue

        det_mask = det.mask
        if det_mask is not None:
            # det.mask is (box_h, box_w) boolean/uint8 within the bbox.
            # Resize if it doesn't match the absolute bbox dimensions.
            if det_mask.shape != (box_h, box_w):
                mask_img = _PILImage.fromarray(
                    det_mask.astype(np.uint8) * 255,
                )
                mask_img = mask_img.resize(
                    (box_w, box_h),
                    _PILImage.NEAREST,
                )
                det_mask = np.array(mask_img) > 0
            else:
                det_mask = det_mask > 0

            canvas[y1:y2, x1:x2][det_mask] = cls_idx
        else:
            # No instance mask — fill entire bounding box.
            canvas[y1:y2, x1:x2] = cls_idx

    return canvas


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class FinetuneSegmentation(foo.Operator):
    """Fine-tune a HuggingFace semantic-segmentation model."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="finetune_segmentation",
            label="Fine-tune Segmentation Model",
            description=(
                "Fine-tune any HuggingFace "
                "AutoModelForSemanticSegmentation model "
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

        inputs.view(
            "experimental_warning",
            types.Warning(
                label="Experimental",
                description=(
                    "Semantic segmentation fine-tuning is experimental. "
                    "Results may vary depending on your dataset and "
                    "model combination. Please report any issues."
                ),
            ),
        )

        # -- Label field (auto-detect compatible fields) --------------------
        compatible_fields = _get_segmentation_compatible_fields(ctx)

        if not compatible_fields:
            inputs.view(
                "no_fields_warning",
                types.Warning(
                    label="No compatible fields found",
                    description=(
                        "Your dataset needs at least one field of "
                        "type fiftyone.core.labels.Segmentation or "
                        "fiftyone.core.labels.Detections"
                    ),
                ),
            )
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Segmentation"),
            )

        # Build display labels so the user can see the field type
        field_choices = list(compatible_fields.keys())

        inputs.enum(
            "label_field",
            values=field_choices,
            required=True,
            label="Label Field",
            description=(
                "Segmentation or Detections field to train on. "
                "Detections fields will be composited into per-pixel "
                "semantic masks automatically."
            ),
        )

        label_field = ctx.params.get("label_field")
        if not label_field:
            return types.Property(
                inputs,
                view=types.View(label="Fine-tune Segmentation"),
            )

        field_type = compatible_fields.get(label_field)

        # -- Class info (depends on field type) -----------------------------
        if field_type == "Detections":
            # Discover classes from the detection labels
            distinct_labels = sorted(
                ctx.dataset.distinct(
                    f"{label_field}.detections.label",
                )
            )
            if distinct_labels:
                num_classes = len(distinct_labels) + 1  # +1 background
                preview = ", ".join(distinct_labels[:10])
                if len(distinct_labels) > 10:
                    preview += (
                        f" ... (+{len(distinct_labels) - 10} more)"
                    )

                inputs.view(
                    "class_info",
                    types.Notice(
                        label=(
                            f"Detections field — found "
                            f"{len(distinct_labels)} class(es): "
                            f"{preview}.  A 'background' class will "
                            f"be added at index 0 "
                            f"({num_classes} total)."
                        ),
                    ),
                )
            else:
                inputs.view(
                    "no_labels_warning",
                    types.Warning(
                        label="No detection labels found",
                        description=(
                            f"Field '{label_field}' has no "
                            "detections with labels."
                        ),
                    ),
                )
        else:
            # Segmentation field — use mask_targets as before
            mask_targets = _get_mask_targets(ctx, label_field)
            if mask_targets:
                num_classes = len(mask_targets)
                preview_items = list(mask_targets.items())[:10]
                preview = ", ".join(
                    f"{v} ({k})" for k, v in preview_items
                )
                if num_classes > 10:
                    preview += (
                        f" ... (+{num_classes - 10} more)"
                    )

                inputs.view(
                    "class_info",
                    types.Notice(
                        label=(
                            f"Found {num_classes} classes: {preview}"
                        ),
                    ),
                )
            else:
                inputs.view(
                    "no_targets_warning",
                    types.Warning(
                        label="No mask_targets found",
                        description=(
                            "Your dataset has no mask_targets set "
                            f"for '{label_field}'. You must provide "
                            "num_classes manually, and pixel values "
                            "in the mask will be used as class "
                            "indices directly."
                        ),
                    ),
                )
                inputs.int(
                    "num_classes",
                    required=True,
                    label="Number of Classes",
                    description=(
                        "Total number of semantic classes in your "
                        "masks (including background if applicable)"
                    ),
                )

        # -- Model name -----------------------------------------------------
        inputs.str(
            "model_name",
            required=True,
            label="Model Name",
            description=(
                "Any HuggingFace "
                "AutoModelForSemanticSegmentation model ID. "
                "Examples: nvidia/mit-b0, "
                "nvidia/segformer-b0-finetuned-ade-512-512, "
                "facebook/mask2former-swin-tiny-ade-semantic"
            ),
            default="nvidia/mit-b0",
        )

        inputs.bool(
            "do_reduce_labels",
            default=False,
            label="Reduce Labels",
            description=(
                "Shift all label IDs down by 1 and treat pixel "
                "value 0 as background (255 = ignore). Enable "
                "this for datasets like ADE20K where 0 = "
                "background."
            ),
            view=types.CheckboxView(),
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
                description=(
                    "Sample tag that identifies training samples"
                ),
            )
            inputs.str(
                "val_tag",
                default="val",
                label="Validation Tag",
                description=(
                    "Sample tag that identifies validation samples"
                ),
            )

        # -- Training hyper-parameters --------------------------------------
        inputs.int(
            "num_epochs",
            default=50,
            label="Epochs",
            description="Number of training epochs",
        )
        inputs.int(
            "batch_size",
            default=2,
            label="Batch Size",
            description="Per-device training and evaluation batch size",
        )
        inputs.float(
            "learning_rate",
            default=6e-5,
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
            default="finetuned_segmentation_model",
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
            view=types.View(label="Fine-tune Segmentation Model"),
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
            "model_dir_name", "finetuned_segmentation_model",
        )
        output_dir = os.path.join(parent_dir, model_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            return self._run_training(
                target_view=ctx.target_view(),
                label_field=ctx.params["label_field"],
                model_name=ctx.params["model_name"],
                do_reduce_labels=ctx.params.get(
                    "do_reduce_labels", False,
                ),
                num_classes_override=ctx.params.get("num_classes"),
                split_strategy=ctx.params.get(
                    "split_strategy", "percentage",
                ),
                train_split=ctx.params.get("train_split", 0.8),
                train_tag=ctx.params.get("train_tag", "train"),
                val_tag=ctx.params.get("val_tag", "val"),
                num_epochs=ctx.params.get("num_epochs", 50),
                batch_size=ctx.params.get("batch_size", 2),
                learning_rate=ctx.params.get("learning_rate", 6e-5),
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
        model_name,
        do_reduce_labels=False,
        num_classes_override=None,
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=50,
        batch_size=2,
        learning_rate=6e-5,
        output_dir="./finetuned_segmentation_model",
        push_to_hub=False,
        hub_model_id=None,
        hub_private=True,
        hub_token=None,
    ):
        """Run the full semantic-segmentation fine-tuning pipeline.

        This method contains all the training logic and is called by both
        ``execute()`` (App / delegated path) and ``__call__()`` (SDK
        path).
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
        import torch.nn as nn
        from PIL import Image
        from fiftyone.utils.torch import GetItem
        from transformers import (
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
            Trainer,
            TrainingArguments,
        )

        output_dir = os.path.expanduser(output_dir)

        # -- 1. Detect field type and build class mapping -------------------
        logger.info("Building class mapping...")

        dataset = target_view._dataset
        field_type = _get_field_type(dataset, label_field)

        if field_type is None:
            raise ValueError(
                f"Field '{label_field}' is not a Segmentation or "
                "Detections field."
            )

        logger.info(
            "Label field '%s' has type %s", label_field, field_type,
        )

        if field_type == "Detections":
            # Discover classes from distinct detection labels.
            # Index 0 is reserved for background (unlabeled pixels).
            distinct_labels = sorted(
                target_view.distinct(
                    f"{label_field}.detections.label",
                )
            )
            if not distinct_labels:
                raise ValueError(
                    f"Detections field '{label_field}' has no "
                    "labels.  Cannot build a class mapping."
                )

            id2label = {0: "background"}
            label2id = {"background": 0}
            for i, lbl in enumerate(distinct_labels, start=1):
                id2label[i] = lbl
                label2id[lbl] = i
            num_labels = len(id2label)
        else:
            # Segmentation field — use mask_targets or fallback
            mask_targets = None
            if dataset.mask_targets:
                mask_targets = dataset.mask_targets.get(label_field)
            if mask_targets is None:
                mask_targets = dataset.default_mask_targets or None

            if mask_targets:
                # mask_targets is {pixel_value: class_name}
                id2label = {
                    int(k): v for k, v in mask_targets.items()
                }
                label2id = {
                    v: int(k) for k, v in mask_targets.items()
                }
                num_labels = len(mask_targets)
            elif num_classes_override:
                num_labels = num_classes_override
                id2label = {i: str(i) for i in range(num_labels)}
                label2id = {str(i): i for i in range(num_labels)}
            else:
                raise ValueError(
                    "No mask_targets found on the dataset and "
                    "num_classes was not provided.  Either set "
                    "dataset.mask_targets or provide num_classes."
                )

        if num_labels < 2:
            raise ValueError(
                f"Need at least 2 classes to fine-tune, but found "
                f"{num_labels} in field '{label_field}'"
            )

        logger.info("Classes (%d): %s", num_labels, id2label)
        print(f"Found {num_labels} classes")

        # -- 2. Split into train / val --------------------------------------
        logger.info(
            "Splitting dataset (strategy=%s)...", split_strategy,
        )

        if split_strategy == "tags":
            train_view = target_view.match_tags(train_tag)
            val_view = target_view.match_tags(val_tag)
        else:
            sample_ids = list(target_view.values("id"))
            random.shuffle(sample_ids)
            n_train = int(len(sample_ids) * train_split)
            train_view = target_view.select(sample_ids[:n_train])
            val_view = target_view.select(sample_ids[n_train:])

        # Filter out samples with no label data
        train_view = train_view.exists(label_field)
        val_view = val_view.exists(label_field)

        # For Detections fields, also exclude samples with zero
        # detections (exists() only checks the field is not None).
        if field_type == "Detections":
            from fiftyone import ViewField as F

            _has_dets = (
                F(f"{label_field}.detections").length() > 0
            )
            train_view = train_view.match(_has_dets)
            val_view = val_view.match(_has_dets)

        n_train = len(train_view)
        n_val = len(val_view)

        logger.info(
            "Train samples: %d  |  Val samples: %d", n_train, n_val,
        )
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
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            do_reduce_labels=do_reduce_labels,
        )

        print(f"Loading model {model_name}...")
        model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # -- 4. Build PyTorch datasets via GetItem + to_torch() -------------
        #
        # Semantic segmentation works at the IMAGE level.
        # Each sample has an image and a per-pixel mask.
        # The GetItem loads the image + mask and runs them through the
        # HuggingFace image processor.
        #
        # For Segmentation fields the mask is loaded directly (with
        # support for both in-memory masks and on-disk mask_path).
        # For Detections fields the per-instance masks are composited
        # into a single semantic mask on the fly.
        # -----------------------------------------------------------------
        print("Building datasets...")

        # Capture outer variables for the closure
        _field_type = field_type
        _label2id = label2id

        class _SegmentationGetItem(GetItem):
            """Bridge from FiftyOne labels to HF format.

            Supports both ``Segmentation`` and ``Detections`` fields.

            Each sample is transformed into::

                {
                    "pixel_values": Tensor[C, H, W],
                    "labels":       Tensor[H, W] (class indices),
                }
            """

            def __init__(self, proc, field_mapping=None):
                self.proc = proc
                super().__init__(field_mapping=field_mapping)

            @property
            def required_keys(self):
                return ["filepath", "label_data"]

            def __call__(self, d):
                image = Image.open(d["filepath"]).convert("RGB")
                label_data = d.get("label_data")

                if _field_type == "Detections":
                    # Composite instance masks → semantic mask
                    w_img, h_img = image.size
                    mask = _detections_to_semantic_mask(
                        label_data.detections,
                        (h_img, w_img),
                        _label2id,
                    )
                else:
                    # Segmentation — handles both mask and
                    # mask_path transparently
                    mask = _load_mask_from_segmentation(
                        label_data,
                    )

                seg_map = Image.fromarray(
                    mask.astype(np.uint8),
                )

                inputs = self.proc(
                    images=image,
                    segmentation_maps=seg_map,
                    return_tensors="pt",
                )

                # Squeeze the batch dim the processor adds
                return {
                    k: v.squeeze(0) for k, v in inputs.items()
                }

        field_mapping = {"label_data": label_field}

        train_getter = _SegmentationGetItem(
            processor,
            field_mapping=field_mapping,
        )
        val_getter = _SegmentationGetItem(
            processor,
            field_mapping=field_mapping,
        )

        train_dataset = train_view.to_torch(train_getter)
        val_dataset = val_view.to_torch(val_getter)

        # -- 5. Configure Trainer -------------------------------------------
        print("Configuring trainer...")

        # Custom collate: stack pixel_values and labels tensors.
        # Images are resized to the same size by the processor, so
        # a simple stack works here.
        def _collate_fn(batch):
            return {
                "pixel_values": torch.stack(
                    [item["pixel_values"] for item in batch]
                ),
                "labels": torch.stack(
                    [item["labels"] for item in batch]
                ),
            }

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="mean_iou",
            remove_unused_columns=False,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id if push_to_hub else None,
            hub_private_repo=hub_private,
            hub_token=hub_token,
            logging_steps=1,
            fp16=torch.cuda.is_available(),
        )

        # -- Evaluation metric: mean IoU -----------------------------------
        _num_labels = num_labels

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            # logits shape: (B, num_labels, h, w) — typically lower
            # resolution than labels
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()

            # Compute mean IoU per class
            ious = []
            for cls in range(_num_labels):
                pred_mask = pred_labels == cls
                true_mask = labels == cls
                intersection = (pred_mask & true_mask).sum()
                union = (pred_mask | true_mask).sum()
                if union > 0:
                    ious.append(intersection / union)

            mean_iou = float(np.mean(ious)) if ious else 0.0

            return {"mean_iou": mean_iou}

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=_collate_fn,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics,
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
                    "do_reduce_labels": do_reduce_labels,
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
        epochs_str = (
            str(int(epochs)) if epochs is not None else "N/A"
        )

        summary = (
            "### Training Complete\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| **Model** | `{results.get('model_name', 'N/A')}` |\n"
            f"| **Saved to** | `{results.get('output_dir', 'N/A')}` |\n"
            f"| **Classes** | {results.get('num_classes', 'N/A')} |\n"
            f"| **Train samples** | "
            f"{results.get('train_samples', 'N/A')} |\n"
            f"| **Val samples** | "
            f"{results.get('val_samples', 'N/A')} |\n"
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
        do_reduce_labels=False,
        num_classes=None,
        split_strategy="percentage",
        train_split=0.8,
        train_tag="train",
        val_tag="val",
        num_epochs=50,
        batch_size=2,
        learning_rate=6e-5,
        output_dir="./finetuned_segmentation_model",
        push_to_hub=False,
        hub_model_id=None,
        hub_private=True,
        delegate=False,
    ):
        """Execute segmentation fine-tuning via the SDK through the
        operator framework.

        The *label_field* may be either:

        - A ``Segmentation`` field (in-memory ``mask`` or on-disk
          ``mask_path``).
        - A ``Detections`` field whose per-instance masks will be
          composited into per-pixel semantic masks automatically (a
          ``background`` class is added at index 0).

        In a notebook you must ``await`` the result::

            result = await segmentation_trainer(
                dataset,
                label_field="ground_truth",
                model_name="nvidia/mit-b0",
            )

        Args:
            sample_collection: a
                :class:`fiftyone.core.collections.SampleCollection`
                (Dataset or DatasetView) to train on.
            label_field: the name of the ``Segmentation`` or
                ``Detections`` label field.
            model_name: any HuggingFace
                ``AutoModelForSemanticSegmentation`` model ID.
            do_reduce_labels: whether to shift label IDs down by 1 and
                treat pixel value 0 as background (set to ``True`` for
                ADE20K-style datasets).
            num_classes: total number of classes. Required only when
                using a ``Segmentation`` field and ``mask_targets`` is
                not set on the dataset.  Ignored for ``Detections``
                fields (classes are discovered automatically).
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
            hub_model_id: Hub repository ID (required when
                *push_to_hub* is ``True``).
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
            do_reduce_labels=do_reduce_labels,
            num_classes=num_classes,
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
