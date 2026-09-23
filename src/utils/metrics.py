"""Metrics and timing facilities for the pipeline.

Centralizes how ``pipeline.run`` measures and reports per-prompt tracking.
``measure_time`` is re-exported from ``src.utils.timer`` so callers depend on
this module (the metrics interface) instead of the timer implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from src.utils.timer import measure_time

__all__ = ["measure_time", "track_prompt"]


@contextmanager
def track_prompt(index: int) -> Iterator[None]:
    """Measure and report wall-clock time of a single prompt generation.

    Args:
        index: Zero-based index of the prompt being generated.

    Prints:
        ``[Started] Prompt <index>`` before the block and
        ``[Timming] Prompt <index>: <elapsed> ms`` after it, keeping the
        timing concerns out of the pipeline loop.
    """
    print(f"[Started] Prompt {index}")
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        print(f"[Timming] Prompt {index}: {elapsed_ms} ms")
