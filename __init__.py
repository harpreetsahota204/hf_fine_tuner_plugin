"""FiftyOne HuggingFace Fine-tuning Plugin.

Fine-tune HuggingFace vision models directly from the FiftyOne App
or programmatically via the SDK.
"""

import os

os.environ["FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS"] = "true"

from .classification import FinetuneClassification
from .detection import FinetuneDetection
from .segmentation import FinetuneSegmentation


def register(plugin):
    """Register operators with the plugin."""
    plugin.register(FinetuneClassification)
    plugin.register(FinetuneDetection)
    plugin.register(FinetuneSegmentation)
