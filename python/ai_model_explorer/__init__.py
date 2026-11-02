"""AI Model Explorer: see any AI model's architecture from its weight manifest, without downloading the weights.

    from ai_model_explorer import explore, compare
    m = explore("Qwen/Qwen3-8B")
    print(m.summary())
    m.insights(); m.breakdown("kind"); m.memory(context=32768)
    m.plot("qwen3.png"); m.report("qwen3.html")

Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Recovery of Neural Network Anatomy", 2026.
"""
__version__ = "0.2.0"

from .api import inspect, summary  # noqa: E402
from .model import Model, compare, compare_table, explore  # noqa: E402

__all__ = ["explore", "compare", "compare_table", "Model", "inspect", "summary", "__version__"]
