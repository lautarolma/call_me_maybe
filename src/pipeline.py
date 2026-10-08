"""Pipeline orchestration for call_me_maybe.

Loads function definitions, prompts and the model vocabulary, initializes
the model, runs constrained generation over every prompt, validates each
result and persists the output JSON required by the subject.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

from src.models.output import FunctionCall
from src.prompt.prompt_builder import build_prompt
from src.utils.metrics import (
    MetricsRun,
    measure_time,
    report_prompt_metrics,
    track_prompt,
)
from src.decoder.constrained_generator import generate
from src.decoder.trie import build_trie
from src.validator import build_results, find_unsupported_prompts

import argparse

# Import of the SDK provided by the course (it lives in llm_sdk/llm_sdk/__init__.py).
# `Small_LLM_Model` wraps a Hugging Face causal model for inference.
from llm_sdk import Small_LLM_Model

from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import load_vocab


def run(args: argparse.Namespace) -> int:
    """Run the Phase 1 pipeline: load inputs + model, print summary.

    Args:
        args: Parsed CLI arguments (``functions_definition``, ``input``,
            ``output`` paths).

    Returns:
        ``0`` on success, ``1`` on failure.

    HOW IT WORKS (internals):
    - It is the ORCHESTRATOR: it does not know how to load JSONs or run
      tensors, it only coordinates the specialists in order and propagates
      exceptions to `__main__.py`, which turns them into exit code 1.
    """
    print(f"[1/5] Loading function definitions from {args.functions_definition} ...")
    # load_functions(path) -> list[FunctionDef]:
    #   opens the JSON, parses it with json.load, validates each entry against
    #   the pydantic FunctionDef model (types, required fields) and checks for
    #   duplicate names. On failure it raises ValueError with a descriptive
    #   message (see function_loader.py for the details).
    functions = load_functions(args.functions_definition)

    print(f"[2/5] Loading prompts from {args.input} ...")
    # load_prompts(path) -> list[str]:
    #   same JSON parsing mechanism, but accepts two formats: an array of
    #   plain strings or an array of {"prompt": "..."} objects. It rejects
    #   empty lists (running a pipeline without inputs makes no sense).
    #
    # WHY WE KEEP BOTH LISTS: `build_prompt` injects the function definitions
    # and produces the text the MODEL sees. The `prompt` field of the output
    # must be the ORIGINAL request, without the definitions — the grader
    # compares it with `correction["prompt"]` by EXACT string equality. If we
    # only kept the constructed prompts, we would write the definitions-laden
    # prompt into the output, and the comparison would fail on all 11 tests.
    raw_prompts = load_prompts(args.input)
    prompts = [build_prompt(functions, prompt) for prompt in raw_prompts]

    print("[3/5] Initializing model (first run downloads weights from the HF Hub) ...")
    # Small_LLM_Model() with no arguments uses the default Qwen/Qwen3-0.6B.
    # WHAT IT DOES INSIDE (llm_sdk):
    #   1. Picks the device with priority mps > cuda > cpu:
    #      - mps = Metal Performance Shaders (Apple Silicon GPU).
    #      - cuda = NVIDIA GPU.
    #      - cpu = universal fallback, slower but always available.
    #   2. Picks dtype: float16 on GPU/MPS (half the memory, ~1.2 GB for
    #      600M parameters vs ~2.4 GB in float32), float32 on CPU for
    #      numeric compatibility.
    #   3. AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"): downloads (the
    #      first time) or reads from cache (~/.cache/huggingface/hub) the
    #      tokenizer files (vocab.json, merges.txt, tokenizer.json) and
    #      builds the object that maps text <-> token ids.
    #   4. AutoModelForCausalLM.from_pretrained(...): same, but with the
    #      transformer WEIGHTS (safetensors). `torch_dtype=self._dtype`
    #      loads the weights already in float16.
    #   5. .eval(): puts the model in inference mode (disables dropout and
    #      other layers with training-specific behavior).
    #   6. requires_grad=False on all parameters: tells PyTorch autograd
    #      not to record operations for gradient computation, saving memory
    #      and speeding up every forward pass.
    model = Small_LLM_Model()

    print("[4/5] Building vocabulary index ...")
    # load_vocab(model) -> Vocab:
    #   reads the MODEL's vocab.json (via get_path_to_vocab_file, which
    #   resolves the path in the HF cache) and pre-indexes each token by its
    #   FIRST DECODED CHARACTER. That index is the basis of the constrained
    #   decoder: when the generator needs "all tokens starting with `"`", it
    #   answers in O(1) with a set of ids instead of scanning 150k+ tokens.
    vocab = load_vocab(model)
    print("[5/5] Building trie ...")
    trie_node = build_trie([function.name for function in functions])

    print()
    print("=== Phase 1 summary ===")
    print(f"  functions : {len(functions)}")
    print(f"  prompts   : {len(prompts)}")
    print(f"  trie_node : {len(trie_node.children)}")

    print(f"  vocab size: {vocab.vocab_size}")
    print(f"  first-char buckets : {len(vocab.tokens_starting_with)}")
    print(f"  output path: {args.output}")
    print()
    print("All components loaded OK. Starting constrained generation.")

    # Results are accumulated to print them outside the loop: the generation
    # cycle stays free of prints and direct measurements (those live in
    # src/utils/metrics.py).
    generated: list[str] = []
    # One MetricsRun PER PROMPT, not a single accumulator: the forward count
    # is the only magnitude that does NOT depend on the hardware, so it is the
    # one that separates "the code did more work" from "the machine was
    # slower" (see report_prompt_metrics). The loop stays print-free: it only
    # accumulates.
    prompt_metrics: list[MetricsRun] = []
    with measure_time("Complete run"):
        for i, prompt in enumerate(prompts):
            with track_prompt(i):
                prompt_run = MetricsRun()
                generated.append(
                    generate(
                        model,
                        prompt,
                        vocab,
                        functions,
                        trie_node,
                        metrics=prompt_run,
                    )[0]
                )
                prompt_metrics.append(prompt_run)

    # The forward count lives in data/output/ (git-ignored, next to the
    # deliverable) and NOT in /tmp: the measurement evidence must survive a
    # VM restart.
    report_prompt_metrics(prompt_metrics, args.output.parent / "decode_metrics.json")

    # --- Deliverable persistence -------------------------------------------
    # We turn each decoder string into a validated FunctionCall, paired with
    # its ORIGINAL prompt, and write the array to disk. `build_results`
    # preserves the 1:1 order with the input prompts: the grader pairs with
    # `zip()`, which is positional.
    results: list[FunctionCall] = build_results(raw_prompts, generated)

    # The stdout echo comes AFTER `build_results`, not before: what is
    # printed must be what stays in the file. It used to print the decoder's
    # raw string and the console showed `replacement: "****"` and
    # `template: 'Say hello to {name}'` while the JSON on disk already had
    # `*` and `Say "hello" to {name}`. The grader scores the file, so the
    # score was still 11/11 — but a reviewer reading the console saw broken
    # output. It prints `name` + `parameters` (not `prompt`, already printed
    # when reading the input) in the SAME format as before.
    for call in results:
        print(f"  result : {json.dumps(call.echo_view(), indent=2, ensure_ascii=False)}")

    # --- Diagnostics for unmatched prompts (stderr, never the JSON) --------
    # The constrained decoder always emits a valid function, so a prompt that
    # matches no function does not crash: it picks the least ugly one and
    # moves on. Without this warning the output looks correct. It does not
    # alter the results file or the score — it is a sensor for whoever reads
    # the run. See `find_unsupported_prompts` for the criterion.
    for idx in find_unsupported_prompts(raw_prompts, results):
        print(
            f"WARNING: prompt {idx} produced a call with no argument value "
            f"present in the prompt text — the request most likely matches "
            f"none of the available functions: {raw_prompts[idx]!r}",
            file=sys.stderr,
        )

    written = write_results(results, args.output)

    print()
    print(f"  wrote {len(results)} entries -> {written}")
    return 0


def write_results(results: list[FunctionCall], path: Path) -> Path:
    """Serialize the results to the output JSON and write it to disk.

    Args:
        results: Validated entries, in the order of the input prompts.
        path: Destination. The parent is created if missing (the subject
            requires the program to create the output/ directory at run time).

    Returns:
        The path actually written.

    WHY `model_dump()` and not `model_dump_json()`:
    `json.dumps` over a list of dicts is simpler to test (the result is plain
    text, not bytes) and keeps control of the indent. `ensure_ascii=False`
    matters: the prompts contain accents and typographic quotes; without it
    the JSON escapes them as \\uXXXX, which is still valid but unreadable for
    a human reviewer.
    """
    payload = [call.model_dump() for call in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
