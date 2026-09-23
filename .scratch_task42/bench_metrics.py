"""Benchmark real por prompt with constrained-decoder metrics.

Usage:
    uv run python .scratch_task42/bench_metrics.py original n1 n2 v1 v2

The runner uses the real model and records wall time plus GenerationMetrics.
It does not change generation semantics or use KV cache.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_sdk import Small_LLM_Model  # noqa: E402
from src.decoder.constrained_generator import (  # noqa: E402
    GenerationMetrics,
    generate,
)
from src.decoder.trie import build_trie  # noqa: E402
from src.loader.function_loader import load_functions  # noqa: E402
from src.loader.input_loader import load_prompts  # noqa: E402
from src.loader.vocab_loader import load_vocab  # noqa: E402
from src.prompt.prompt_builder import build_prompt  # noqa: E402


SCRATCH = ROOT / ".scratch_task42"
BASE = SCRATCH / "bench_inputs"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark definitions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sets = _load_module("bench_sets_defs", SCRATCH / "bench_sets.py")
_inputs = _load_module("bench_inputs_defs", SCRATCH / "bench_inputs_perf.py")

CONFIGS = {
    "original": (
        ROOT / "data/input/functions_definition.json",
        ROOT / "data/input/function_calling_tests.json",
        _sets.SETS["original"]["expected"],
    ),
    "n1": (
        BASE / "n1/functions_definition.json",
        BASE / "n1/function_calling_tests.json",
        _sets.SETS["n1"]["expected"],
    ),
    "n2": (
        BASE / "n2/functions_definition.json",
        BASE / "n2/function_calling_tests.json",
        _sets.SETS["n2"]["expected"],
    ),
    "v1": (
        BASE / "v1/functions_definition.json",
        BASE / "v1/function_calling_tests.json",
        _inputs.EXPECTED["v1"],
    ),
    "v2": (
        BASE / "v2/functions_definition.json",
        BASE / "v2/function_calling_tests.json",
        _inputs.EXPECTED["v2"],
    ),
}


def _match_value(got: object, expected: object) -> bool:
    if isinstance(expected, list):
        return any(_match_value(got, item) for item in expected)
    if isinstance(expected, bool) or isinstance(got, bool):
        return type(got) is type(expected) and got == expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return float(got) == float(expected)
    return type(got) is type(expected) and got == expected


def _score(output: str, expected: dict) -> tuple[bool, bool, list[str]]:
    try:
        call = json.loads(output)
    except Exception as exc:  # noqa: BLE001
        return False, False, [f"json: {exc}"]
    if call.get("name") != expected["fn"]:
        return False, False, [
            f"fn={call.get('name')!r} != {expected['fn']!r}"
        ]
    errors: list[str] = []
    parameters = call.get("parameters", {})
    for key, wanted in expected["args"].items():
        if key not in parameters:
            errors.append(f"missing parameter {key!r}")
        elif not _match_value(parameters[key], wanted):
            errors.append(f"{key}={parameters[key]!r} !~ {wanted!r}")
    return True, not errors, errors


def _run_set(name: str, model: Small_LLM_Model, vocab) -> dict:
    functions_path, prompts_path, expected = CONFIGS[name]
    functions = load_functions(functions_path)
    prompts = load_prompts(prompts_path)
    if len(prompts) != len(expected):
        raise ValueError(
            f"{name}: {len(prompts)} prompts but {len(expected)} expected"
        )
    trie = build_trie([function.name for function in functions])
    results: list[dict] = []
    for index, (query, wanted) in enumerate(zip(prompts, expected), start=1):
        metrics = GenerationMetrics()
        started = time.perf_counter()
        output, complete = generate(
            model,
            build_prompt(functions, query),
            vocab,
            functions,
            trie,
            metrics=metrics,
        )
        wall_seconds = time.perf_counter() - started
        function_ok, full_ok, errors = _score(output, wanted)
        results.append(
            {
                "index": index,
                "prompt": query,
                "complete": complete,
                "function_ok": function_ok,
                "full_ok": full_ok,
                "errors": errors,
                "wall_seconds": wall_seconds,
                "metrics": {
                    "forward_calls": metrics.forward_calls,
                    "forward_seconds": metrics.forward_seconds,
                    "allowed_compute_seconds": metrics.allowed_compute_seconds,
                    "ranked_iterator_seconds": metrics.ranked_iterator_seconds,
                    "fine_validation_seconds": metrics.fine_validation_seconds,
                    "fine_validation_calls": metrics.fine_validation_calls,
                    "fine_validation_rejections": metrics.fine_validation_rejections,
                    "candidates_tested": metrics.candidates_tested,
                    "deterministic_tokens": metrics.deterministic_tokens,
                    "ambiguous_steps": metrics.ambiguous_steps,
                },
            }
        )
        print(
            f"[{name} {index}/{len(prompts)}] "
            f"{'OK' if full_ok else 'FN' if function_ok else 'XX'} "
            f"{wall_seconds:.2f}s forwards={metrics.forward_calls} "
            f"det={metrics.deterministic_tokens} "
            f"amb={metrics.ambiguous_steps} "
            f"candidates={metrics.candidates_tested}"
        )
    total_wall = sum(row["wall_seconds"] for row in results)
    return {
        "set": name,
        "prompts": len(results),
        "function_accuracy": sum(row["function_ok"] for row in results),
        "full_accuracy": sum(row["full_ok"] for row in results),
        "wall_seconds": total_wall,
        "results": results,
    }


def main() -> None:
    names = sys.argv[1:] or list(CONFIGS)
    unknown = set(names) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown sets: {sorted(unknown)}")
    started = time.perf_counter()
    print(f"Loading model, threads={torch.get_num_threads()} ...")
    model = Small_LLM_Model()
    print("Loading vocabulary ...")
    vocab = load_vocab(model)
    all_results = []
    for name in names:
        all_results.append(_run_set(name, model, vocab))
    output_path = SCRATCH / "results_metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "threads": torch.get_num_threads(),
                "elapsed_seconds": time.perf_counter() - started,
                "sets": all_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
