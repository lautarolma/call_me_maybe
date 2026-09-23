"""
    para medir el tiempo de ejecucion de un bloque de codigo.
    Con un Context manager  yo le paso este measure_time
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

@contextmanager
def measure_time(labels: str) -> Iterator[None]:
    print(f"[Started] {labels}")
    start = perf_counter()
    try:
        yield

    finally:
        elapsed_time = (perf_counter() - start) * 1000.0
        print(f"[Timming] {labels}: {elapsed_time} ms")