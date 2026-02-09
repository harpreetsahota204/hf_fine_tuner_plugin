# HuggingFace Fine-Tuning Plugin for FiftyOne

Fine-tune HuggingFace vision models directly from the FiftyOne App or
programmatically via the Python SDK.

## Supported Tasks

| Task | Operator | Status |
|------|----------|--------|
| Image Classification | `finetune_classification` | Stable |
| Object Detection | `finetune_detection` | Stable |
| Semantic Segmentation | `finetune_segmentation` | Experimental |

## Features

- Fine-tune any compatible HuggingFace model (ie, `AutoModelForImageClassification` or `AutoModelForObjectDetection`) on your FiftyOne dataset
- Train/val splitting via percentage or existing sample tags
- Background (delegated) execution so the App stays responsive
- Optional push to HuggingFace Hub (private repos by default)
- Full SDK support for programmatic / notebook workflows

## Requirements

- FiftyOne >= 1.0
- Python 3.9+
- A CUDA-capable GPU is recommended

## Installation

```bash
fiftyone plugins download https://github.com/harpreetsahota204/hf_fine_tuner_plugin

# Install any requirements for the plugin
fiftyone plugins requirements @harpreetsahota/hf_fine_tuner_plugin --install
```

## Secrets

If you want to push models to the HuggingFace Hub, set your token as an
environment variable before launching FiftyOne:

```bash
export HF_TOKEN="hf_..."
```

Or authenticate via the HuggingFace CLI:

```bash
hf auth login
```

## Usage — App

Launch the FiftyOne App, open a dataset, and find the fine-tuning operators
in the operator browser or the grid actions menu.

<!-- TODO: Add GIFs showing App usage -->

## Usage — SDK

### Classification
![image](icons/ft_classification.gif)

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.utils.huggingface as fouh

# Load a dataset
dataset = fouh.load_from_hub("Voxel51/cats-vs-dogs-sample", max_samples=100)

# Get the operator
classification_trainer = foo.get_operator(
    "@harpreetsahota/hf_fine_tuner_plugin/finetune_classification"
)

# Run fine-tuning
classification_trainer(
    dataset.view(),
    label_field="label",
    model_name="google/vit-base-patch16-224",
    num_epochs=1,
    batch_size=16,
    output_dir="./my_finetuned_vit",
)
```

### Object Detection
![image](icons/ft_detection.gif)

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

# Load a dataset with detections
dataset = foz.load_zoo_dataset("quickstart")

# Get the operator
det_trainer = foo.get_operator(
    "@harpreetsahota/hf_fine_tuner_plugin/finetune_detection"
)

# Run fine-tuning
det_trainer(
    dataset,
    label_field="ground_truth",
    model_name="facebook/detr-resnet-50",
    bbox_format="coco",
    num_epochs=1,
    batch_size=16,
    output_dir="./my_finetuned_detr",
)
```

### Operator Parameters

All three operators share these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_field` | str | *required* | The label field to train on |
| `model_name` | str | *required* | HuggingFace model ID |
| `split_strategy` | str | `"percentage"` | `"percentage"` or `"tags"` |
| `train_split` | float | `0.8` | Fraction for training (percentage mode) |
| `train_tag` | str | `"train"` | Sample tag for training data (tags mode) |
| `val_tag` | str | `"val"` | Sample tag for validation data (tags mode) |
| `num_epochs` | int | varies | Number of training epochs |
| `batch_size` | int | varies | Per-device batch size |
| `learning_rate` | float | varies | Peak learning rate |
| `output_dir` | str | varies | Directory for saving the fine-tuned model |
| `push_to_hub` | bool | `False` | Push model to HuggingFace Hub |
| `hub_model_id` | str | `None` | Hub repo ID (required if pushing) |
| `hub_private` | bool | `True` | Make Hub repo private |
| `delegate` | bool | `False` | Run in background via orchestrator |

The detection operator also accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox_format` | str | `"coco"` | `"coco"`, `"yolo"`, or `"xyxy"` |

### Delegated Execution

Training runs as a delegated operation in the App by default. To use
delegated execution from the SDK, pass `delegate=True`.

Before launching the delegated service, set the following environment
variables in the same terminal:

```bash
# Required for the delegated orchestrator
export FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS=true

# Optional: control which GPU(s) training runs on (defaults to "0")
export CUDA_VISIBLE_DEVICES=0

# Launch the delegated operation service
fiftyone delegated launch
```

Then kick off training from your Python process:

```python
classification_trainer(
    dataset.view(),
    label_field="label",
    model_name="google/vit-base-patch16-224",
    output_dir="./my_finetuned_vit",
    delegate=True,
)
```

## Multi-GPU Note

By default, training runs on a single GPU to avoid issues with PyTorch's
`DataParallel` (which breaks models like DETR). If you want to use a
different GPU, set `CUDA_VISIBLE_DEVICES` before launching the delegated
service (e.g. `export CUDA_VISIBLE_DEVICES=1`). For multi-GPU training,
use an `accelerate`-compatible setup.

## License

Apache 2.0
