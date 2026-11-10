"""Independent one-shot Handwriting-VAR package."""

from .pipeline import HandwritingVARPipeline
from .config import ModelConfig, TrainConfig

__all__ = ["HandwritingVARPipeline", "ModelConfig", "TrainConfig"]
__version__ = "0.1.0"

