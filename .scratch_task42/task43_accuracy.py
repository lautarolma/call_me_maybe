"""Task 4.3 - Accuracy test con los 11 prompts (temporal, no versionado).

Corre generate() con los 11 prompts de function_calling_tests.json contra el
modelo REAL Qwen3-0.6B y mide:
  1. accuracy de función (nombre correcto)
  2. accuracy full (función + argumentos correctos)  <- quality bar M14 del subject
  3. timing total y por prompt

Scoring: el nombre DEBE coincidir. Los args deben coincidir en key/value.
Para fn_substitute_string_with_regex el regex admite variantes equivalentes
(\\d+, [0-9]+) y replacement según intención del prompt.

Resultados parciales persistidos en .scratch_task42/results_43.json por si
el proceso se corta.

Uso:  uv run python .scratch_task42/task43_accuracy.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

import torch  # noqa: E402

# Óptimo del bench_scaling en esta máquina (i7-7700HQ, 6 vCPU):
# threads=4: 2611 ms/step | threads=6: 5123 ms/step (PEOR: bandwidth-bound).
torch.set_num_threads(4)

from src.decoder.constrained_generator import generate  # noqa: E402
from src.decoder.trie import build_trie  # noqa: E402
from src.loader.function_loader import load_functions  # noqa: E402
from src.loader.input_loader import load_prompts  # noqa: E402
from src.loader.vocab_loader import load_vocab  # noqa: E402
from src.prompt.prompt_builder import build_prompt  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"
RESULTS_PATH = Path(".scratch_task42/results_43.json")

# Ground truth esperado para los 11 prompts (construido a mano contra
# functions_definition.json). "regex" admite variantes -> lista de aceptados.
EXPECTED: list[dict] = [
    {"fn": "fn_add_numbers", "args": {"a": 2, "b": 3}},
    {"fn": "fn_add_numbers", "args": {"a": 265, "b": 345}},
    {"fn": "fn_greet", "args": {"name": "shrek"}},
    {"fn": "fn_greet", "args": {"name": "john"}},
    {"fn": "fn_reverse_string", "args": {"s": "hello"}},
    {"fn": "fn_reverse_string", "args": {"s": "world"}},
    {"fn": "fn_get_square_root", "args": {"a": 16}},
    {"fn": "fn_get_square_root", "args": {"a": 144}},
    {
        "fn": "fn_substitute_string_with_regex",
        "args": {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": ["\\d+", "[0-9]+"],
            "replacement": "NUMBERS",
        },
    },
    {
        "fn": "fn_substitute_string_with_regex",
        "args": {
            "source_string": "Programming is fun",
            "regex": ["[aeiouAEIOU]", "[aeiou]"],
            "replacement": "*",
        },
    },
    {
        "fn": "fn_substitute_string_with_regex",
        "args": {
            "source_string": "The cat sat on the mat with another cat",
            "regex": ["cat", "\\bcat\\b"],
            "replacement": "dog",
        },
    },
]


def _match_value(got: object, expected: object) -> bool:
    """Compara un valor; si expected es lista, acepta cualquiera de sus items."""
    if isinstance(expected, list):
        return any(_match_value(got, item) for item in expected)
    return type(got) is type(expected) and got == expected


def check_call(call: dict, expected: dict) -> tuple[bool, list[str]]:
    """Valida un JSON parseado contra el ground truth. Retorna (ok, errores)."""
    errors: list[str] = []
    if call.get("name") != expected["fn"]:
        return False, [f"fn={call.get('name')!r} != {expected['fn']!r}"]
    for key, want in expected["args"].items():
        if key not in call.get("parameters", {}):
            errors.append(f"falta param {key!r}")
        elif not _match_value(call["parameters"][key], want):
            errors.append(f"{key}={call['parameters'][key]!r} !~ {want!r}")
    return not errors, errors


def main() -> None:
    t0 = time.time()
    print(f"[0] threads={torch.get_num_threads()} | todo se persistira en {RESULTS_PATH}")

    print("[1] Loading functions ...")
    functions = load_functions(FUNCTIONS_PATH)
    print("[2] Loading prompts ...")
    prompts = load_prompts(PROMPTS_PATH)
    assert len(prompts) == 11, f"esperaba 11 prompts, hay {len(prompts)}"

    print("[3] Initializing model (carga pesada, paciencia) ...")
    model = Small_LLM_Model()
    print(f"    model listo en {time.time() - t0:.1f}s")

    print("[4] Building vocab index (~150k decodes, tarda) ...")
    vocab = load_vocab(model)
    print(f"    vocab_size={vocab.vocab_size} en {time.time() - t0:.1f}s")

    print("[5] Building trie ...")
    trie = build_trie([fn.name for fn in functions])

    results: list[dict] = []
    fn_ok = full_ok = 0
    total_gen_s = 0.0
    for i, (query, expected) in enumerate(zip(prompts, EXPECTED), start=1):
        prompt = build_prompt(functions, query)
        t1 = time.time()
        output, ok = generate(model, prompt, vocab, functions, trie)
        gen_s = time.time() - t1
        total_gen_s += gen_s

        errors: list[str] = []
        try:
            call = json.loads(output)
            fn_correct = call.get("name") == expected["fn"]
            full, errors = check_call(call, expected)
        except Exception as exc:  # noqa: BLE001 - output no parseable
            call, fn_correct, full = None, False, False
            errors = [f"json: {exc}"]

        fn_ok += int(fn_correct)
        full_ok += int(full)
        status = "OK" if full else ("FN" if fn_correct else "XX")
        print(f"[{i:>2}/11] {status} {gen_s:6.1f}s | {query[:60]!r}")
        for err in errors:
            print(f"        - {err}")

        entry = {
            "i": i,
            "query": query,
            "ok_full": full,
            "ok_fn": fn_correct,
            "errors": errors,
            "gen_s": round(gen_s, 1),
            "output": output if not full else None,
        }
        results.append(entry)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    print(f"\n=== RESULTADO TASK 4.3 ===")
    print(f"accuracy fn : {fn_ok}/11 ({fn_ok / 11 * 100:.0f}%)")
    print(f"accuracy full: {full_ok}/11 ({full_ok / 11 * 100:.0f}%)  <- quality bar M14")
    print(f"timing generación: {total_gen_s / 60:.1f} min | total con carga: {elapsed / 60:.1f} min")
    print(f"KPI <5 min: {'OK' if total_gen_s < 300 else 'INCUMPLIDO en CPU'}")
    print(f"resultados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()