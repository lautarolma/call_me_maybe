"""Task 4.2 - Single-prompt smoke test (temporal, no versionado).

Corre generate() con el primer prompt real del test file contra el modelo
Qwen3-0.6B REAL. Verifica: output parseable como JSON y contiene
"fn_add_numbers". Mide tiempos de carga, generación y ms/step estimado.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

from src.decoder.constrained_generator import generate
from src.decoder.trie import build_trie
from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import load_vocab
from src.prompt.prompt_builder import build_prompt
from llm_sdk import Small_LLM_Model

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"


def main() -> None:
    t0 = time.time()

    print("[1/5] Loading functions ...")
    functions = load_functions(FUNCTIONS_PATH)

    print("[2/5] Loading prompts ...")
    prompts = load_prompts(PROMPTS_PATH)
    query = prompts[0]
    print(f"      primer prompt: {query[:70]!r}")

    print("[3/5] Initializing model (carga pesada, paciencia) ...")
    model = Small_LLM_Model()
    print(f"      model listo en {time.time() - t0:.1f}s")

    print("[4/5] Building vocab index (decode ~150k tokens, tarda) ...")
    vocab = load_vocab(model)
    print(f"      vocab_size={vocab.vocab_size} buckets={len(vocab.tokens_starting_with)} "
          f"en {time.time() - t0:.1f}s")

    print("[5/5] Building trie ...")
    trie = build_trie([fn.name for fn in functions])

    prompt = build_prompt(functions, query)
    n_prompt_tokens = len(model.encode(prompt)[0].tolist())
    print(f"\n=== SMOKE TEST ===")
    print(f"query        : {query!r}")
    print(f"prompt tokens: {n_prompt_tokens}")

    # Bench de un forward del modelo (ms/step) con ventana de 32 tokens.
    bench_ids = model.encode(prompt)[0].tolist()[:32]
    bench_t = time.time()
    for _ in range(3):
        model.get_logits_from_input_ids(bench_ids)
    bench_ms = (time.time() - bench_t) / 3 * 1000
    print(f"forward bench: ~{bench_ms:.0f} ms/step (32 tokens)")

    t1 = time.time()
    output, ok = generate(model, prompt, vocab, functions, trie)
    gen_s = time.time() - t1
    print(f"\ngenerate() total: {gen_s:.1f}s | ok(COMPLETE)={ok}")
    print(f"output ({len(output)} chars):\n{output[:600]!r}")

    # Validaciones del acceptance criteria.
    try:
        parsed = json.loads(output)
        print(f"\nJSON parseable : OK")
        print(f"name           : {parsed.get('name')!r}")
        print(f"parameters     : {parsed.get('parameters')}")
        assert parsed.get("name") == "fn_add_numbers", (
            f"FALLO: name={parsed.get('name')!r} != 'fn_add_numbers'"
        )
        print("fn_add_numbers : OK")
        print(f"\n>>> SMOKE TEST PASS ({(time.time() - t0) / 60:.1f} min total) ✅")
    except Exception as exc:  # noqa: BLE001 - informe crudo del smoke
        print(f"\n>>> SMOKE TEST FAIL ❌: {exc}")


if __name__ == "__main__":
    main()