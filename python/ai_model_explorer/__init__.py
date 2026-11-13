"""AI Model Explorer: recover a model's architecture from its weight manifest, without downloading the weights.

    from ai_model_explorer import explore, compare, fits, diff, ready, census
    m = explore("Qwen/Qwen3-8B")
    print(m.summary())
    m.insights(); m.breakdown("kind"); m.memory(context=32768)
    m.plot("qwen3.png"); m.report("qwen3.html")
    fits("Qwen/Qwen3-8B", "rtx-4090", context=32768)
    diff("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8")
    ready("Qwen/Qwen3-8B"); census("Qwen/Qwen3-8B")

Method: S. Singkul, "Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories", 2026.
"""
__version__ = "1.0.3"

from .api import inspect, summary  # noqa: E402
from .model import Model, compare, compare_table, explore  # noqa: E402
from .tools import DEVICES, census, diff, fits, ready, scan  # noqa: E402

__all__ = ["explore", "compare", "compare_table", "Model", "inspect", "summary", "fits", "diff", "scan", "DEVICES", "ready", "census", "__version__"]
