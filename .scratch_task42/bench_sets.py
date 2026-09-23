"""Bench de accuracy + performance contra un set de inputs (temporal, no versionado).

Corre generate() con el modelo REAL Qwen3-0.6B contra un set de inputs
(original | n1 | n2) y mide:
  1. accuracy de funcion (nombre correcto)
  2. accuracy full (funcion + argumentos correctos)
  3. timing por prompt, promedio y total

WATCHDOG: 600s (10 min) por PROMPT — si un prompt lo excede se registra como
timeout en los resultados y el bench CONTINUA (dato medible, no aborto).
Un crash/excepcion NO capturada aborta todo el bench (semantica `timeout`).

Uso:  uv run python .scratch_task42/bench_sets.py original|n1|n2
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

TIMEOUT_S = 600  # 10 min por prompt (max historico 329.9s -> margen 2x)

BASE = Path(".scratch_task42/bench_inputs")

# Ruta de inputs y ground truth por set.
SETS: dict[str, dict] = {
    "original": {
        "functions": Path("data/input/functions_definition.json"),
        "prompts": Path("data/input/function_calling_tests.json"),
        "expected": [
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
        ],
    },
    "n1": {
        "functions": BASE / "n1/functions_definition.json",
        "prompts": BASE / "n1/function_calling_tests.json",
        "expected": [
            {"fn": "fn_calculate_rectangle_area", "args": {"width": 4, "height": 7}},
            {"fn": "fn_calculate_rectangle_area", "args": {"width": 5, "height": 8}},
            {"fn": "fn_count_vowels", "args": {"text": "hello world"}},
            {"fn": "fn_count_vowels", "args": {"text": "programming is fun"}},
            {"fn": "fn_is_palindrome", "args": {"word": "radar"}},
            {"fn": "fn_is_palindrome", "args": {"word": "level"}},
            {"fn": "fn_to_uppercase", "args": {"text": "hello"}},
            {"fn": "fn_to_uppercase", "args": {"text": "good morning"}},
            {"fn": "fn_get_max_of_two", "args": {"a": 15, "b": 22}},
            {"fn": "fn_get_max_of_two", "args": {"a": 3, "b": 8}},
            {"fn": "fn_remove_vowels", "args": {"text": "banana"}},
        ],
    },
    "n2": {
        "functions": BASE / "n2/functions_definition.json",
        "prompts": BASE / "n2/function_calling_tests.json",
        "expected": [
            {"fn": "fn_to_kelvin", "args": {"celsius": 25}},
            {"fn": "fn_to_kelvin", "args": {"celsius": 0}},
            {"fn": "fn_count_words", "args": {"text": "the quick brown fox"}},
            {"fn": "fn_count_words", "args": {"text": "hello world"}},
            {"fn": "fn_starts_with", "args": {"text": "hello", "prefix": "he"}},
            {"fn": "fn_starts_with", "args": {"text": "world", "prefix": "wor"}},
            {"fn": "fn_join_strings", "args": {"left": "foo", "right": "bar"}},
            {"fn": "fn_join_strings", "args": {"left": "open", "right": "source"}},
            {"fn": "fn_get_first_character", "args": {"text": "python"}},
            {"fn": "fn_get_first_character", "args": {"text": "banana"}},
            {"fn": "fn_get_last_word", "args": {"text": "hello brave world"}},
        ],
    },
}


class PromptTimeout(Exception):
    """Un prompt excedio el watchdog de TIMEOUT_S segundos."""


def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
    raise PromptTimeout(f"prompt excedio {TIMEOUT_S}s")


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
    current: dict = {"prompt": "loading", "gen_s": 0.0}

    if len(sys.argv) != 2 or sys.argv[1] not in SETS:
        print("uso: uv run python .scratch_task42/bench_sets.py original|n1|n2")
        sys.exit(2)
    name = sys.argv[1]
    cfg = SETS[name]
    results_path = BASE / f"results_{name}.json"

    signal.signal(signal.SIGALRM, _alarm_handler)

    t0 = time.time()
    print(f"[0] set={name} | threads={torch.get_num_threads()} | watchdog={TIMEOUT_S}s/prompt")

    print("[1] Loading functions ...")
    functions = load_functions(cfg["functions"])
    print(f"    {len(functions)} funciones: {[f.name for f in functions]}")
    print("[2] Loading prompts ...")
    prompts = load_prompts(cfg["prompts"])
    expected = cfg["expected"]
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
    timeouts = 0
    for i, (query, want) in enumerate(zip(prompts, expected), start=1):
        prompt = build_prompt(functions, query)
        current = {"prompt": query, "i": i, "gen_s": 0.0}

        errors: list[str] = []
        timeout = False
        try:
            signal.alarm(TIMEOUT_S)
            t1 = time.time()
            output, ok = generate(model, prompt, vocab, functions, trie)
            gen_s = time.time() - t1
        except PromptTimeout:
            timeout = True
            output, ok, gen_s, call, fn_correct, full = None, False, TIMEOUT_S * 1.0, None, False, False
            errors = [f"WATCHDOG: excedio {TIMEOUT_S}s"]
            timeouts += 1
        finally:
            signal.alarm(0)
        if not timeout:
            total_gen_s += gen_s
            current["gen_s"] = round(gen_s, 1)
            try:
                call = json.loads(output)
                fn_correct = call.get("name") == want["fn"]
                full, errors = check_call(call, want)
            except PromptTimeout:  # pragma: no cover - defensivo
                raise
            except Exception as exc:  # noqa: BLE001 - output no parseable
                call, fn_correct, full = None, False, False
                errors = [f"json: {exc}"]

        fn_ok += int(fn_correct)
        full_ok += int(full)
        status = "TO" if timeout else ("OK" if full else ("FN" if fn_correct else "XX"))
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
            "timeout": timeout,
            "output": output if not full else None,
        }
        results.append(entry)
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    n = len(prompts)
    print(f"\n=== RESULTADO BENCH {name} ===")
    print(f"accuracy fn  : {fn_ok}/{n} ({fn_ok / n * 100:.0f}%)")
    print(f"accuracy full: {full_ok}/{n} ({full_ok / n * 100:.0f}%)  <- quality bar M14")
    print(f"timeouts     : {timeouts}")
    if timeouts == 0:
        print(f"timing generacion: {total_gen_s / 60:.1f} min | total con carga: {elapsed / 60:.1f} min")
        print(f"promedio/prompt   : {total_gen_s / n:.1f}s | max: {max(r['gen_s'] for r in results):.1f}s")
        print(f"proyeccion a 11 prompts: {total_gen_s:.0f}s ({total_gen_s / 60:.1f} min) "
              f"| KPI <5 min: {'OK' if total_gen_s < 300 else 'INCUMPLIDO'}")
    else:
        print(f"timing parcial (sin timeouts): {total_gen_s / 60:.1f} min | total con carga: {elapsed / 60:.1f} min")
        print(f"promedio (sin timeouts): {total_gen_s / (n - timeouts):.1f}s")
    print(f"resultados en {results_path}")


if __name__ == "__main__":
    main()