"""AI Model Explorer: see any AI model's architecture from its weight manifest, without downloading the weights.

    from ai_model_explorer import inspect, summary
    a = inspect("Qwen/Qwen3-8B")
    print(summary(a))

Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy", 2026.
"""
__version__ = "0.1.0"

from .api import inspect, summary  # noqa: E402

__all__ = ["inspect", "summary", "__version__"]
