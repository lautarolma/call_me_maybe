"""Bench de scaling de cores para la VM nueva (temporal, no versionado).

Mide el forward del modelo con 1..N threads y reporta ms/step por thread,
para decidir si la CPU alcanza el KPI (<5 min / 11 prompts) o se escala.

Uso:  uv run python .scratch_task42/bench_scaling.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

import torch  # noqa: E402

from llm_sdk import Small_LLM_Model  # noqa: E402

CORES = os.cpu_count() or 1
SEQ = 190  # tokens del prompt típico (medido en el smoke)


def bench_forward(threads: int, seq_len: int) -> float:
    """ms/step promedio para un forward con `threads` hilos."""
    torch.set_num_threads(threads)
    model = Small_LLM_Model()
    ids = model.encode("x" * 50)[0].tolist()
    # usar ids repetidos hasta seq_len para simular ventana real
    ids = (ids * ((seq_len // len(ids)) + 1))[:seq_len]
    # 3 forwards de warmup + 3 medidos
    for _ in range(3):
        model.get_logits_from_input_ids(ids)
    t0 = time.time()
    for _ in range(3):
        model.get_logits_from_input_ids(ids)
    return (time.time() - t0) / 3 * 1000


def main() -> None:
    print(f"CPU cores disponibles: {CORES}")
    print(f"secuencia de bench: {SEQ} tokens\n")
    results = []
    for threads in [1, 2, 4, 6, 8]:
        if threads > CORES:
            break
        ms = bench_forward(threads, SEQ)
        results.append((threads, ms))
        print(f"  threads={threads}: {ms:.0f} ms/step")

    # extrapolación 11 prompts
    print("\n=== Extrapolación KPI (<5 min para 11 prompts) ===")
    for threads, ms in results:
        steps_prompt = 70  # avg estimado del smoke (308s a ~4.4s/step)
        per_prompt = ms * steps_prompt / 1000
        total = per_prompt * 11
        ok = "✅" if total < 300 else "❌"
        print(f"  threads={threads}: ~{per_prompt:.0f}s/prompt → 11 prompts ≈ {total:.0f}s {ok}")


if __name__ == "__main__":
    main()