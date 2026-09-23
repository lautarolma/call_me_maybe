"""Probe instrumentado: costo real por step del loop de generación (scratch).

Copia el loop de constrained_generator.generate() con timers por step para
medir cuánto cuesta CADA pieza:
  - t_check : M5 check (compute_allowed_ids SIN logits = filter completo)
  - t_fwd   : forward del modelo (get_logits_from_input_ids)
  - t_m12   : M1/M2 (compute_allowed_ids CON logits)
  - len_check: cuántos candidatos devolvió el M5 check
  - wildcard: si la fase pedía '*' (espera cualquier char)

No toca producción: importa solo lo que necesita y replica el loop.

Uso: uv run python .scratch_task42/probe_step_cost.py
"""

from __future__ import annotations

import sys
import time
from copy import copy

sys.path.insert(0, "/home/laviles/Python/call_me_maybe")

import torch  # noqa: E402

torch.set_num_threads(4)

from src.decoder.schema_validator import SchemaContext  # noqa: E402
from src.decoder.state import DecoderPhase, DecoderState  # noqa: E402
from src.decoder.token_filter import compute_allowed_ids  # noqa: E402
from src.decoder.trie import build_trie  # noqa: E402
from src.loader.function_loader import load_functions  # noqa: E402
from src.loader.input_loader import load_prompts  # noqa: E402
from src.loader.vocab_loader import load_vocab  # noqa: E402
from src.prompt.prompt_builder import build_prompt  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402

FUNCTIONS_PATH = ".scratch_task42/bench_inputs/v1/functions_definition.json"
PROMPTS_PATH = ".scratch_task42/bench_inputs/v1/function_calling_tests.json"
N_PROMPTS = 1  # 1 prompt basta para el diagnóstico por step


def main() -> None:
    t0 = time.time()
    functions = load_functions(FUNCTIONS_PATH)
    prompts = load_prompts(PROMPTS_PATH)
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    trie = build_trie([fn.name for fn in functions])

    query = prompts[0]
    prompt = build_prompt(functions, query)
    input_ids = model.encode(prompt)[0].tolist()
    prompt_length = len(input_ids)
    state = DecoderState()
    schema = SchemaContext(functions)

    steps_total = 0
    t_check = t_fwd = t_m12 = 0.0
    wildcard_steps = 0
    skip_used = 0
    len_dist: dict[int, int] = {}

    for _ in range(200):
        # ─── réplica del loop (M5 → forward → M1/M2) ───
        t1 = time.perf_counter()
        allowed_check = compute_allowed_ids(state, schema, vocab, trie)
        t2 = time.perf_counter()
        t_check += t2 - t1
        is_wildcard = "*" in state.expected_first_chars()
        len_dist[len(allowed_check)] = len_dist.get(len(allowed_check), 0) + 1
        steps_total += 1
        if is_wildcard:
            wildcard_steps += 1

        if len(allowed_check) == 1 and not is_wildcard:
            best_id = next(iter(allowed_check))
            token_text = vocab.id2decoded.get(best_id)
            if token_text is None:
                break
            input_ids.append(best_id)
            if not state.update_from_text(token_text):
                break
            schema.update(state)
            skip_used += 1
            if state.phase is DecoderPhase.COMPLETE:
                break
            continue

        # Ambiguous step: forward
        t3 = time.perf_counter()
        logits = model.get_logits_from_input_ids(input_ids)
        t4 = time.perf_counter()
        t_fwd += t4 - t3

        # M1/M2
        t5 = time.perf_counter()
        allowed = compute_allowed_ids(state, schema, vocab, trie, logits)
        t6 = time.perf_counter()
        t_m12 += t6 - t5

        if not allowed:
            print("EMPTY ALLOWED — cortando")
            break

        # argmax + pase fino (replicado)
        while allowed:
            best_id = max(allowed, key=lambda tid: logits[tid])
            token_text = vocab.id2decoded.get(best_id)
            if token_text is not None:
                trial = copy(state)
                fine = SchemaContext(functions)
                fine._params_object_seen = schema.has_seen_params_object()
                fine.update(trial)
                ok_fine = True
                for char in token_text:
                    if not trial.update_from_text(char):
                        ok_fine = False
                        break
                    if not fine.allows_token(char, trial, trie):
                        ok_fine = False
                        break
                    fine.update(trial)
                if ok_fine:
                    break
            allowed.discard(best_id)
        else:
            print("FINE ALL AGOTADO — cortando")
            break

        if token_text is None:
            break
        input_ids.append(best_id)
        if not state.update_from_text(token_text):
            break
        schema.update(state)
        if state.phase is DecoderPhase.COMPLETE:
            break

    elapsed = time.time() - t0
    print(f"\n=== PROBE STEP COST (prompt 1: {query!r}) ===")
    print(f"steps totales        : {steps_total} (wildcard: {wildcard_steps}, {wildcard_steps / steps_total * 100:.0f}%)")
    print(f"skip-if-single usados: {skip_used}")
    print(f"t_check (M5, filter completo sin logits): {t_check:.1f}s  ({t_check / steps_total * 1000:.0f} ms/step)")
    print(f"t_fwd   (forward modelo)               : {t_fwd:.1f}s  ({t_fwd / steps_total * 1000:.0f} ms/step)")
    print(f"t_m12   (M1/M2 con logits)             : {t_m12:.1f}s  ({t_m12 / steps_total * 1000:.0f} ms/step)")
    print(f"total generación                       : {elapsed:.1f}s")
    print(f"distribución len(allowed_check): {dict(sorted(len_dist.items()))}")
    print(f"tiempo teórico sin M5 check  : {elapsed - t_check:.1f}s")


if __name__ == "__main__":
    main()