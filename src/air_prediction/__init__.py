"""CSAI Project 19: hourly PM2.5 forecasting with Classical ML and Deep Learning Hybrid Models."""

__version__ = "0.2.0"

class _MissingTorch:
    """Placeholder for deep learning components when PyTorch is not installed."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"'{self._name}' requires PyTorch, which is not installed. "
            "Install optional deep learning dependencies with: pip install -e '.[dl]'"
        )

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<Unavailable: {self._name} (requires PyTorch)>"


try:
    from .deep_learning import (
        DLConfig,
        DeepLearningPipeline,
        HybridResidualModel,
        SequenceAirDataset,
        TemporalForecaster,
    )
except ImportError as exc:
    # Only treat as optional if PyTorch itself is what failed to import.
    # If deep_learning has an unrelated broken import or syntax issue, re-raise immediately.
    if "torch" in str(exc) or getattr(exc, "name", None) == "torch":
        DLConfig = _MissingTorch("DLConfig")
        DeepLearningPipeline = _MissingTorch("DeepLearningPipeline")
        HybridResidualModel = _MissingTorch("HybridResidualModel")
        SequenceAirDataset = _MissingTorch("SequenceAirDataset")
        TemporalForecaster = _MissingTorch("TemporalForecaster")
    else:
        raise

__all__ = [
    "DLConfig",
    "DeepLearningPipeline",
    "HybridResidualModel",
    "SequenceAirDataset",
    "TemporalForecaster",
]
