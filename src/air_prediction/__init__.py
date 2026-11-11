"""CSAI Project 19: hourly PM2.5 forecasting with Classical ML and Deep Learning Hybrid Models."""

# Initialize the project CPU policy before NumPy, XGBoost or PyTorch loads.
# Explicit caller settings are preserved; local launchers use the same defaults.
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
del _variable

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


def __getattr__(name):
    """Keep classical data/model imports independent of optional PyTorch."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    try:
        module = import_module(".deep_learning", __name__)
    except ImportError as exc:
        if "torch" in str(exc).lower() or getattr(exc, "name", None) == "torch":
            return _MissingTorch(name)
        raise
    return getattr(module, name)


__all__ = [
    "DLConfig",
    "DeepLearningPipeline",
    "HybridResidualModel",
    "PinballLoss",
    "SequenceAirDataset",
    "TemporalForecaster",
    "empirical_coverage",
]
