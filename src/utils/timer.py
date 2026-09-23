"""Context manager that measures and prints the wall-clock time of a block."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def measure_time(label: str) -> Iterator[None]:
    """Measure and print the elapsed time of a code block.

    Args:
        label: Human-readable name for the measured block.

    Prints:
        ``[Started] <label>`` before the block and ``[Timming] <label>:
        <elapsed> ms`` after it.
    """
    print(f"[Started] {label}")
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_time = (perf_counter() - start) * 1000.0
        print(f"[Timming] {label}: {elapsed_time} ms")
