"""Profiling del token filter (temporal, no versionado).

Mide el costo PURO de compute_allowed_ids (Fase 1-2-3, Python) sin el
modelo: usa logits falsos y estados típicos. Esto separa el costo del
filter (CPU Python) del costo del forward (modelo C++/oneDNN).

Casos:
  A) ROOT: expected_first_chars={'{'} -> pocos candidatos
  B) KEY abierta (wildcard '*'): TODOS los buckets -> peor caso
  C) STRING abierto (wildcard '*'): igual que B pero con string depth

Uso:  uv run python .scratch_task42/profiling_filter.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

import torch  # noqa: E402

torch.set_num_threads(4)

from src.decoder.schema_validator import SchemaContext  # noqa: E402
from src.decoder.state import DecoderState  # noqa: E402
from src.decoder.token_filter import compute_allowed_ids  # noqa: E402
from src.decoder.trie import build_trie  # noqa: E402
from src.loader.function_loader import load_functions  # noqa: E402
from src.loader.vocab_loader import load_vocab  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402

FUNCTIONS_PATH = "data/input/functions_definition.json"


def bench(state: DecoderState, schema: SchemaContext, vocab, trie, label: str) -> None:
    logits = [0.0] * vocab.vocab_size  # falsos, el filter no los consume
    # warmup + 3 medidas
    for _ in range(2):
        compute_allowed_ids(state, schema, vocab, trie, logits)
    times = []
    n_allowed = 0
    for _ in range(3):
        t0 = time.time()
        allowed = compute_allowed_ids(state, schema, vocab, trie, logits)
        times.append(time.time() - t0)
        n_allowed = len(allowed)
    ms = min(times) * 1000
    print(f"  {label:32s} {ms:7.0f} ms/step (min de 3) | allowed={n_allowed}")


def main() -> None:
    t0 = time.time()
    print("[1] Loading model (para vocab) ...")
    model = Small_LLM_Model()
    print(f"    model en {time.time() - t0:.1f}s")
    print("[2] Building vocab ...")
    vocab = load_vocab(model)
    print(f"    vocab_size={vocab.vocab_size} buckets={len(vocab.tokens_starting_with)}")
    functions = load_functions(FUNCTIONS_PATH)
    trie = build_trie([fn.name for fn in functions])

    schema = SchemaContext(functions)

    # Caso A: ROOT -> solo '{'
    print("\n=== FASE 1+2+3 por step ===")
    bench(DecoderState(), SchemaContext(functions), vocab, trie, "A) ROOT ('{')")

    # Caso B: key abierta, esperando el sig valor (wildcard)
    st = DecoderState()
    st.update_from_text('{"name": "fn_greet", "par')  # a medio camino de "parameters"
    schema.update(st)
    bench(st, schema, vocab, trie, "B) key abierta (wildcard)")

    # Caso C: dentro de un string value (wildcard tambien)
    st2 = DecoderState()
    st2.update_from_text('{"name": "fn_greet", "parameters": {"name": "sh')
    schema2 = SchemaContext(functions)
    schema2.update(st2)
    bench(st2, schema2, vocab, trie, "C) string abierto (wildcard)")

    # Distribucion de buckets: cuantos candidatos hay por caso
    total_buckets = sum(len(ids) for k, ids in vocab.tokens_starting_with.items() if k != "<byte>")
    print(f"\ntotal candidatos wildcard: {total_buckets}")
    sizes = sorted((len(ids), k) for k, ids in vocab.tokens_starting_with.items() if k != "<byte>")
    print(f"bucket mas grande: {sizes[-1][1]!r} con {sizes[-1][0]} tokens")
    print(f"bucket mas chico:  {sizes[0][1]!r} con {sizes[0][0]} tokens")
    print(f"tokens promedio por bucket: {total_buckets / len([k for k in vocab.tokens_starting_with if k != '<byte>']):.0f}")


if __name__ == "__main__":
    main()