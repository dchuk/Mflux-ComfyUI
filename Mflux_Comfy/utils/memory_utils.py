"""MLX memory management utilities.

Provides functions to clear GPU cache and monitor memory usage.
Safe to import even when MLX is not available (returns no-ops).
"""
import gc
import os

# Conditional MLX import (same pattern as Mflux_Core.py)
_skip_mlx_import = os.environ.get("MFLUX_COMFY_DISABLE_MLX_IMPORT") == "1"

try:
    if _skip_mlx_import:
        raise ImportError("Skipping MLX import via env var")
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    _MLX_AVAILABLE = False


def clear_mlx_memory(cache_limit_bytes: int = 1_000_000_000) -> None:
    """Clear MLX cache and free memory.

    Call this after each generation to prevent OOM on subsequent runs.

    Args:
        cache_limit_bytes: Maximum cache size (default 1GB)
    """
    gc.collect()
    if _MLX_AVAILABLE and mx is not None:
        mx.set_cache_limit(cache_limit_bytes)
        mx.clear_cache()


def get_memory_stats() -> dict:
    """Get current MLX memory statistics.

    Returns:
        Dict with active_memory, peak_memory, cache_memory in bytes.
        Returns zeros if MLX is not available.
    """
    if not _MLX_AVAILABLE or mx is None:
        return {
            "active_memory": 0,
            "peak_memory": 0,
            "cache_memory": 0,
            "mlx_available": False,
        }

    return {
        "active_memory": mx.get_active_memory(),
        "peak_memory": mx.get_peak_memory(),
        "cache_memory": mx.get_cache_memory(),
        "mlx_available": True,
    }


def reset_memory_tracking() -> None:
    """Reset peak memory counter for fresh measurement."""
    if _MLX_AVAILABLE and mx is not None:
        mx.reset_peak_memory()
