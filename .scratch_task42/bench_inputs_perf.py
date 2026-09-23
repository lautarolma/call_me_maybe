"""Bench de performance para inputs inventados v1/v2 (temporal, no versionado).

Corre generate() con el modelo REAL Qwen3-0.6B contra una version de inputs
inventados (.scratch_task42/bench_inputs/<version>/) y mide:
  1. accuracy de funcion (nombre correcto)
  2. accuracy full (funcion + argumentos correctos)
  3. timing por prompt y total

WATCHDOG: 240s (4 min) por ejecucion — si se excede, el proceso se cancela con
exit 124 (semantica igual a `timeout`) para revisar la causa raiz.

Uso:  uv run python .scratch_task42/bench_inputs_perf.py v1|v2
"""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

import torch  # noqa: E402

# Optimo del bench_scaling en esta maquina (i7-7700HQ, 6 vCPU):
# threads=4: 2611 ms/step | threads=6: 5123 ms/step (PEOR: bandwidth-bound).
torch.set_num_threads(4)

from src.decoder.constrained_generator import generate  # noqa: E402
from src.decoder.trie import build_trie  # noqa: E402
from src.loader.function_loader import load_functions  # noqa: E402
from src.loader.input_loader import load_prompts  # noqa: E402
from src.loader.vocab_loader import load_vocab  # noqa: E402
from src.prompt.prompt_builder import build_prompt  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402

TIMEOUT_S = 240  # 4 min por ejecucion (watchdog del usuario)

CURRENT: dict = {"prompt": "loading", "gen_s": 0.0}


def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
    """Watchdog: cancela la ejecucion con exit 124 para revisar la causa."""
    print(f"\n[WATCHDOG] TIMEOUT {TIMEOUT_S}s EXCEDIDO en: {CURRENT}")
    print("[WATCHDOG] Cancelando ejecucion (exit 124). Revisar causa raiz.")
    sys.exit(124)


# Ground truth por version (construido a mano contra los inputs inventados).
EXPECTED: dict[str, list[dict]] = {
    "v1": [
        {"fn": "fn_multiply", "args": {"a": 6, "b": 7}},
        {"fn": "fn_power", "args": {"base": 2, "exponent": 10}},
        {"fn": "fn_concatenate", "args": {"left": "foo", "right": "bar"}},
        {"fn": "fn_reverse_words", "args": {"text": "quick brown fox"}},
        {"fn": "fn_greater_than", "args": {"x": 15, "y": 12}},
        {"fn": "fn_multiply", "args": {"a": 4, "b": 8}},
    ],
    "v2": [
        {"fn": "fn_format_date", "args": {"day": 25, "month": 12, "year": 2024}},
        {
            "fn": "fn_count_occurrences",
            "args": {"text": "the cat and the dog and the cat chased the cat", "word": "cat"},
        },
        {"fn": "fn_repeat_string", "args": {"text": "ha", "times": 3}},
        {"fn": "fn_celsius_to_fahrenheit", "args": {"celsius": 100}},
        {"fn": "fn_contains_substring", "args": {"text": "hello world", "substring": "world"}},
        {
            "fn": "fn_count_occurrences",
            "args": {"text": "the quick the fox the dog", "word": "the"},
        },
    ],
}


def _match_value(got: object, expected: object) -> bool:
    """Compara un valor; si expected es lista, acepta cualquiera de sus items."""
    if isinstance(expected, list):
        return any(_match_value(got, item) for item in expected)
    if isinstance(expected, bool) or isinstance(got, bool):
        return type(got) is type(expected) and got == expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return float(got) == float(expected)
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
    global CURRENT

    if len(sys.argv) != 2 or sys.argv[1] not in ("v1", "v2"):
        print("uso: uv run python .scratch_task42/bench_inputs_perf.py v1|v2")
        sys.exit(2)
    version = sys.argv[1]

    base = Path(".scratch_task42/bench_inputs") / version
    functions_path = base / "functions_definition.json"
    prompts_path = base / "function_calling_tests.json"
    results_path = Path(".scratch_task42/bench_inputs") / f"results_{version}.json"
    expected = EXPECTED[version]

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)

    t0 = time.time()
    print(f"[0] version={version} | threads={torch.get_num_threads()} | watchdog={TIMEOUT_S}s")

    print("[1] Loading functions ...")
    functions = load_functions(functions_path)
    print("[2] Loading prompts ...")
    prompts = load_prompts(prompts_path)
    assert len(prompts) == len(expected), f"esperaba {len(expected)} prompts, hay {len(prompts)}"

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
    for i, (query, want) in enumerate(zip(prompts, expected), start=1):
        prompt = build_prompt(functions, query)
        CURRENT = {"prompt": query, "i": i, "gen_s": 0.0}
        t1 = time.time()
        output, ok = generate(model, prompt, vocab, functions, trie)
        gen_s = time.time() - t1
        total_gen_s += gen_s
        CURRENT["gen_s"] = round(gen_s, 1)

        errors: list[str] = []
        try:
            call = json.loads(output)
            fn_correct = call.get("name") == want["fn"]
            full, errors = check_call(call, want)
        except Exception as exc:  # noqa: BLE001 - output no parseable
            call, fn_correct, full = None, False, False
            errors = [f"json: {exc}"]

        fn_ok += int(fn_correct)
        full_ok += int(full)
        status = "OK" if full else ("FN" if fn_correct else "XX")
        print(f"[{i:>2}/{len(prompts)}] {status} {gen_s:6.1f}s | {query[:60]!r}")
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
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    signal.alarm(0)  # watchdog apagado: terminamos en tiempo

    n = len(prompts)
    print(f"\n=== RESULTADO BENCH {version} ===")
    print(f"accuracy fn  : {fn_ok}/{n} ({fn_ok / n * 100:.0f}%)")
    print(f"accuracy full: {full_ok}/{n} ({full_ok / n * 100:.0f}%)")
    print(f"timing generacion: {total_gen_s / 60:.1f} min | total con carga: {elapsed / 60:.1f} min")
    print(f"promedio/prompt   : {total_gen_s / n:.1f}s | max: {max(r['gen_s'] for r in results):.1f}s")
    proy = total_gen_s / n * 11
    print(f"proyeccion a 11 prompts: {proy / 60:.1f} min (KPI <5 min: {'OK' if proy < 300 else 'INCUMPLIDO'})")
    print(f"resultados en {results_path}")


if __name__ == "__main__":
    main()