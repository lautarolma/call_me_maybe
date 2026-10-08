"""Constrained generation loop: the three paths of every decoding step.

HOW TO READ THIS MODULE (start here):
- ``generate()`` is the loop. Everything else is a helper, placed below in
  reverse call order: whatever the loop calls last comes first.
- Each step walks THREE paths, in this order, first match wins:
    1. STATIC TEXT (no forward). If the current state matches a
       deterministic span — the static header at startup, or a closing
       tail — it is injected already tokenized. Cheapest path: no model
       query at all.
    2. M5 SKIP-IF-SINGLE (no forward). If there is no wildcard and the
       full filter returns exactly 1 candidate there is no decision to
       make: commit it without paying a forward.
    3. M1/M2 WITH FORWARD. Query the model and pick by argmax over the
       valid candidates, with the fine pass (see below).
- STRUCTURAL INVARIANT: all three paths end in ONE commit, and that commit
  is the same 4-operation sequence:
      input_ids.append(id) → state.update_from_text(text) → schema.update(state)
      → emitted_parts.append(text)
  The sequence lives encapsulated in ``_commit_token()``, never replicated
  inline. Replicating it per path would let a new commit forget the
  ``emitted_parts`` row, making the fine pass and the oracle's static span
  see an incomplete output — silent breakage of the same class as the
  whitespace-duplication failure documented below.

WHY THIS MODULE EXISTS:
- It is the conveyor belt of the didactic plan: per step it takes the
  model's logits, filters them to the ids that keep the output valid JSON
  (``compute_allowed_ids``), picks the best allowed token by argmax,
  commits the state, and repeats until COMPLETE or the token limit.
- It is the ONLY consumer of ``compute_allowed_ids``: the filter reduces
  the ~151K vocab ids to a small set; the argmax its signature reserved
  lives here (the filter itself does NOT consume logits).

FINE PASS — POST-ARGMAX RE-SIMULATION:
- The schema clauses have documented gaps: the filter validates a token as
  a WHOLE (committed snapshot + simulated snapshot), not its char-by-char
  walk. A token that enters AND leaves structures in one step escapes the
  clauses: full name crossed (clause 1), entry into parameters + first key
  (clause 2 residual), key+value+closing complete (clause 3), entry and
  exit of parameters (clause 4).
- Closed here at NEGLIGIBLE cost: re-simulate ONLY the WINNING token
  char-by-char against a FRESH SchemaContext (never the shared one, which
  would stay contaminated if the candidate fails) and validate EVERY
  character. Along the char walk the clauses fire where the per-token pass
  did not: the current_key reset, the intermediate COLON, the depth
  0→1→0 transitions.
- If the winner fails the fine pass it is dropped and the NEXT best in
  ``allowed`` is tried (repeated argmax). Normally the first one passes
  (~1 re-simulation per step); the worst case is controlled degradation.
  NOTE — the retry only has somewhere to go depending on which filter
  branch produced `allowed` (see ``_pick_best_token``): with M2 (model
  top-k) there are up to 2000 alternatives, so the retry can rescue the
  generation; with M1 (raw argmax, fast path) `allowed` holds a single
  element, so if that one fails there is no alternative and the veto
  applies. The M1 veto has never been observed in the suite, but the
  branch stays open by design (M1 exists to validate nothing extra when
  the decision is already taken).
- It also closes the "output object without parameters" gap: when the
  state reaches COMPLETE, the fine pass requires the walk to have gone
  through PARAMS_OBJECT (``SchemaContext.has_seen_params_object``). The
  subject ALWAYS emits the object (even empty for fn_empty).

DOCUMENTED DEVIATION from the didactic pseudocode:
- The commit uses ``vocab.id2decoded``, NOT ``vocab.id2token``. Same
  deviation the filter documents: the state machine works over DECODED
  text ('Ġthe' → ' the'); id2token would provide a byte-mapheaded token
  that breaks syntax validation. The final decode for the output is done
  with ``model.decode()`` over the ids (the SDK applies the inverse table).

WHAT IT DOES NOT DO (separation of concerns):
- No full-pipeline orchestration (loaders, output file): that lives in
  pipeline.py. This module is only the pure generation loop.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from time import perf_counter

from llm_sdk import Small_LLM_Model

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.token_filter import compute_allowed_ids
from src.decoder.trie import TrieNode
from src.loader.vocab_loader import Vocab
from src.models.function_definition import FunctionDef
from src.utils.metrics import MetricsRun

MAX_TOKENS = 200  # Safety net — expected output is ~30-60 tokens

# 100% schema-deterministic prefix: EVERY valid output starts with this
# structure before the LLM has to choose the real function name. It is
# tokenized ONCE with encode() and injected with no forward and no filter
# (see _commit_static_text): it cuts the structural forwards of
# ROOT/OBJECT_OPEN/KEY_START/IN_KEY/KEY_END/COLON (~8-11 per prompt,
# ~4.5-4.8s each — the forward() cost is uniform per step and does not
# depend on the candidate-set size, so skipping the forward is the only
# real lever here).
# ⚠ The header MUST be BYTE-EXACT to the format the model naturally
# produces (the two leading newlines before '{'). Trimming those newlines
# "because they cost no forward" (injecting more text costs nothing extra:
# the whole header skips the model equally) pushed the model out of
# distribution: it tokenized the name as a lone "f" instead of a natural
# chunk, leaving no valid candidate for the next step (no token among the
# top-2000 by logit kept "f..." as a trie prefix). With the newlines
# restored, that prompt completes again.
STATIC_HEADER = '\n\n{\n  "name": "'

# ─── Per-state oracle (level-1 static spans) ─────────────────────────────
# Generalizes the linear second span into a table of spans per state
# (level 1 of the formal register): each span is a PURE function
# (state, schema, generated_text) -> aligned expected text, or None (None =
# fall through to the normal path: M5 → filter → forward). The oracle
# NEVER forces: it only answers when the span is the only deterministic
# continuation the schema allows — the design itself rules out forcing text
# the model would never produce.
#
# Glossary (each symbol, its practical meaning — verified against state.py):
#   generated_text        Output text generated so far (fed per call by
#                         generate(); the oracle aligns against it).
#   expected_text         The ideal text the oracle wants to inject.
#   trailing_whitespace   Trailing spaces/newlines of generated_text
#                         (ws-blind phases ONLY: VALUE_END/IN_OBJECT/
#                         PARAMS_OBJECT/KEY_END/COLON — in KEY_START/IN_KEY
#                         a space is key content and is not consumed).
#   is_between_name_and_parameters
#                         depth==0 ∧ current_key=="name" ∧ keys_enclosed==∅
#                         → gate for the parameters-opening span. Replaces
#                         ¬has_seen_params_object: that flag is sticky and
#                         only turns on if the token ENDS in PARAMS_OBJECT
#                         (update() runs with the post-token state) → a BPE
#                         token crossing the '{' (e.g. '{"') leaves it off
#                         with parameters already open: a false negative in
#                         the UNSAFE direction (it would duplicate the
#                         span). The gate is case-derivable instead: an
#                         internal param literally named "name" leaves
#                         keys_enclosed≠∅ → false; fn_empty (0 params)
#                         leaves current_key=="parameters" → false; true ⟺
#                         between the value of "name" and the next key.
#   parameter_order       tuple(F.parameters) — original insertion order.
#   missing_required_parameters
#                         Required parameter keys still missing
#                         (schema.required_keys_remaining()).
#   has_missing_required_parameters
#                         Pending-params flag: missing_required_parameters
#                         != ∅.
#   first_parameter_name  parameter_order[0], first parameter key.
#   next_required_parameter_name
#                         First parameter key that is still required.
#   opening_quote_for_value(k)
#                         Opening quote for k's value, only if the param
#                         type is "string" (spans leave the state in COLON
#                         or IN_STRING_VALUE d1 depending on the type).
#   parameter_key_text(k) Key text plus its two dots and value opener:
#                         '"age": ' + opening_quote_for_value(k).
#   indented_parameter_entry(k)
#                         Newline, indentation and parameter_key_text(k) —
#                         the model's natural per-key layout.
#   align_static_text(C)  Strip already-emitted trailing_whitespace from
#                         the expected text; None (and '' → None) when
#                         alignment is impossible → span does not apply.
# FREE SPAN: with an empty parameter_order, T1/T2 return None — the model
# generates "parameters": {} with its FUSED {} token; injecting a lone '{'
# would be a seam over a format the model never produces. No subject
# function hits that path (all have params); fn_empty is covered by the
# real probe (inline {} format).
# Level 2 (T7-T10, fused tokens) and B′ (complete fn_name via trie):
# DEFERRED until the level-1 residue is measured.
_WS = " \t\n\r"
_WS_BLIND_PHASES = frozenset(
    {
        DecoderPhase.VALUE_END,
        DecoderPhase.IN_OBJECT,
        DecoderPhase.PARAMS_OBJECT,
        DecoderPhase.KEY_END,
        DecoderPhase.COLON,
    }
)


def _get_trailing_whitespace(text: str) -> str:
    """Return the trailing whitespace from ``text``."""
    return text[len(text.rstrip(_WS)):]


def _align_static_text(
    expected_text: str, generated_text: str, align_trailing_whitespace: bool
) -> str | None:
    """Align expected static text with already generated text.

    If the model already generated the expected trailing whitespace, return
    only the remaining text to avoid duplicating it. Return ``None`` when
    the generated whitespace cannot align with the expected text.
    """
    trailing_whitespace = (
        _get_trailing_whitespace(generated_text)
        if align_trailing_whitespace
        else ""
    )
    if not expected_text.startswith(trailing_whitespace):
        return None
    remaining_text = expected_text[len(trailing_whitespace):]
    return remaining_text or None


def _is_between_name_and_parameters(state: DecoderState) -> bool:
    """Return whether the output is between ``name`` and ``parameters``."""
    return (
        state.depth == 0
        and state.current_key == "name"
        and not state.keys_enclosed
    )


def _get_value_opening_quote(schema: SchemaContext, key: str) -> str:
    """Return the value's opening quote when the parameter is a string."""
    f = schema.selected_function
    if f is None:
        return ""
    param = f.parameters.get(key)
    return '"' if param is not None and param.type == "string" else ""


def _build_parameter_key_text(schema: SchemaContext, key: str) -> str:
    """Build a parameter key and the opening quote of its value."""
    return f'"{key}": ' + _get_value_opening_quote(schema, key)


def _build_indented_parameter_entry(schema: SchemaContext, key: str) -> str:
    """Build an indented parameter entry in the model's natural format."""
    return "\n    " + _build_parameter_key_text(schema, key)


def _get_next_required_parameter_name(schema: SchemaContext) -> str:
    """Return the first required parameter name that has not been emitted."""
    f = schema.selected_function
    if f is None:
        raise AssertionError(
            "_get_next_required_parameter_name() without selected function"
        )
    missing_required_parameters = schema.required_keys_remaining()
    for parameter_name in tuple(f.parameters):
        if parameter_name in missing_required_parameters:
            return parameter_name
    raise AssertionError(
        "_get_next_required_parameter_name() without missing parameters"
    )


def _inject_parameters_after_name_value(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T1: close ``name``, then inject ``parameters`` and its first key."""
    if not (
        state.phase is DecoderPhase.VALUE_END
        and _is_between_name_and_parameters(state)
    ):
        return None
    f = schema.selected_function
    if f is None:
        return None
    parameter_order = tuple(f.parameters)
    if not parameter_order:
        return None  # no params → forward (model's fused {} token)
    expected_text = ",\n  \"parameters\": {" + _build_indented_parameter_entry(
        schema, parameter_order[0]
    )
    return _align_static_text(expected_text, generated_text, True)


def _inject_parameters_after_name_separator(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T2: after the fused comma, inject ``parameters`` and its first key."""
    if not (
        state.phase is DecoderPhase.IN_OBJECT
        and _is_between_name_and_parameters(state)
    ):
        return None
    f = schema.selected_function
    if f is None:
        return None
    parameter_order = tuple(f.parameters)
    if not parameter_order:
        return None
    expected_text = "\n  \"parameters\": {" + _build_indented_parameter_entry(
        schema, parameter_order[0]
    )
    return _align_static_text(expected_text, generated_text, True)


def _inject_next_required_parameter(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T3: inside ``parameters``, inject the next required parameter key.

    PARAMS_OBJECT is this decoder's "between values" phase (verified:
    VALUE_END + ',' → PARAMS_OBJECT, NOT IN_OBJECT). So T3 covers TWO
    entries: (a) the opening '{' arrived by forward with no fused key, and
    (b) the comma closing the PREVIOUS value — the number case:
    IN_NUMBER_VALUE (number open, no VALUE_END) → the model's comma closes
    the number and lands directly in PARAMS_OBJECT. The trailing-whitespace
    alignment absorbs whatever ws the model emitted after the comma
    ('2.0,' + '\n    ' → trailing_whitespace='\n    ').
    """
    if not (state.phase is DecoderPhase.PARAMS_OBJECT and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if not schema.required_keys_remaining():
        return None
    return _align_static_text(
        _build_indented_parameter_entry(
            schema, _get_next_required_parameter_name(schema)
        ),
        generated_text,
        True,
    )


def _inject_next_parameter_after_value(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T4: after a parameter value, inject the next required parameter key.

    Only reachable with a value that CLOSES in its own token (strings, and
    tokens ending with a closer): numbers/booleans/nulls stay OPEN
    (IN_NUMBER/BOOL/NULL_VALUE) until the next character → that flow
    resumes in T3 (the model's comma leaves PARAMS_OBJECT).
    """
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if not schema.required_keys_remaining():
        return None
    expected_text = ",\n    " + _build_parameter_key_text(
        schema, _get_next_required_parameter_name(schema)
    )
    return _align_static_text(expected_text, generated_text, True)


def _close_parameters_and_root(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T5: with no required parameters left, close ``parameters`` and root."""
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if schema.required_keys_remaining():
        return None
    return _align_static_text("\n  }\n}", generated_text, True)


def _close_root_object(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """T6: after ``parameters``, close the root object (including fn_empty).

    The sticky schema flag does NOT need to have observed ``parameters``:
    the model's fused token '"parameters": {}' carries '{'+'}' in one
    chunk, so the sticky schema flag (update runs POST-token) NEVER sees
    the intermediate PARAMS_OBJECT → has_seen_params_object() stays False
    in the real fn_empty case. Being outside the name-to-parameters
    transition with no pending required parameters already guarantees we
    are after parameters: the only depth-0 VALUE_END before parameters is
    the post-name state, blocked above. Therefore the only thing that can
    close here is the root object.
    """
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 0):
        return None
    if _is_between_name_and_parameters(state):
        return None  # T1/T2 own the name-to-parameters transition.
    if schema.selected_function is None:
        return None
    if schema.required_keys_remaining():
        return None
    return _align_static_text("\n}", generated_text, True)


_STATIC_TEXT_RULES: tuple[
    Callable[[DecoderState, SchemaContext, str], str | None], ...
] = (
    _inject_parameters_after_name_value,
    _inject_parameters_after_name_separator,
    _inject_next_required_parameter,
    _inject_next_parameter_after_value,
    _close_parameters_and_root,
    _close_root_object,
)


def _get_next_static_text(
    state: DecoderState, schema: SchemaContext, generated_text: str
) -> str | None:
    """Return the next deterministic text for ``state``, or ``None``.

    The rules are checked in T1-T6 order. Their domains are disjoint by
    construction, so the order documents the design rather than precedence.
    The function is pure: it does not mutate ``state`` or ``schema`` and
    does not query the model.
    """
    for static_text_rule in _STATIC_TEXT_RULES:
        expected_text = static_text_rule(state, schema, generated_text)
        if expected_text is not None:
            return expected_text
    return None


def generate(
    model: Small_LLM_Model,
    prompt: str,
    vocab: Vocab,
    functions: list[FunctionDef],
    trie: TrieNode,
    max_tokens: int = MAX_TOKENS,
    metrics: MetricsRun | None = None,
) -> tuple[str, bool]:
    """Generate schema-constrained JSON output for a prompt.

    Args:
        model: SDK model (encode/get_logits_from_input_ids/decode).
        prompt: Prompt text to answer with a function call.
        vocab: Pre-indexed vocabulary (id2token + id2decoded + buckets).
        functions: Schema function definitions.
        trie: Function-name trie (build_trie(functions)).
        max_tokens: Safety limit for the loop.
        metrics: Optional per-phase metrics accumulator; when provided it
            records forwards, skips-if-single and time per DecoderPhase.

    Returns:
        (generated_text, success): success is True iff the state reached
        COMPLETE.

    HOW IT WORKS:
    - Same skeleton as the plan's pseudocode: tokenize prompt → initial
      state/schema → per step: logits → compute_allowed_ids → argmax →
      append → commit → COMPLETE?.
    - The fine pass lives inside the argmax: ``_pick_best_token()`` drops
      candidates that fail ``_passes_fine_validation()``.
    """
    # ⚠ The SDK returns a 2D tensor [1, N]; [0].tolist() flattens it to
    # list[int] — the contract get_logits_from_input_ids expects.
    input_ids = model.encode(prompt)[0].tolist()
    # Prompt length BEFORE the loop: input_ids afterwards only grows with
    # generated best_ids, so the final slice separates prompt from
    # generated without re-encoding (re-encoding used to count tensor ROWS).
    prompt_length = len(input_ids)
    state = DecoderState()
    schema = SchemaContext(functions)
    # generated output text (without the prompt), accumulated per iteration.
    # Feeds the oracle's trailing-whitespace alignment; it is updated at
    # EVERY commit point (header, oracle, M5, forward) — never reset nor
    # derived from other structures.
    emitted_parts: list[str] = []
    _commit_static_text(
        STATIC_HEADER, model, vocab, input_ids, state, schema, emitted_parts
    )

    for _ in range(max_tokens):
        step_phase = state.phase
        step_start = perf_counter()
        try:
            # ─── Per-state oracle: level-1 static spans ───
            # BEFORE the filter and the forward: if the state matches a
            # deterministic span (_STATIC_TEXT_RULES), inject it pre-tokenized.
            # If the injection does not advance the state, fall through to
            # the normal path — the continue ONLY happens on real advance
            # (otherwise re-matching the same span next step would loop).
            tail = _get_next_static_text(state, schema, "".join(emitted_parts))
            if tail is not None and _commit_static_text(
                tail, model, vocab, input_ids, state, schema, emitted_parts
            ):
                if state.phase is DecoderPhase.COMPLETE:
                    break
                continue

            # ─── M5: Skip-if-single ───
            # Check first WITHOUT model call. In non-wildcard phases, the
            # candidate set is small (~10-100 tokens) so the full filter is
            # fast. If exactly 1 candidate exists, we can skip the forward
            # call entirely.

            expected_chars = state.expected_first_chars()
            if "*" not in expected_chars:
                allowed_check = compute_allowed_ids(state, schema, vocab, trie)
                if len(allowed_check) == 1:
                    if metrics is not None:
                        metrics.add_skips(step_phase, 1)
                    single_id = next(iter(allowed_check))
                    token_text = vocab.id2decoded.get(single_id)
                    if token_text is None:
                        break
                    _inject_float_tail(
                        state, schema, token_text, model, vocab,
                        input_ids, emitted_parts,
                    )
                    if not _commit_token(
                        single_id, token_text, state, schema,
                        input_ids, emitted_parts,
                    ):
                        break
                    if state.phase is DecoderPhase.COMPLETE:
                        break
                    continue

            # ─── Ambiguous step: consult model ───
            logits = model.get_logits_from_input_ids(input_ids)
            if metrics is not None:
                metrics.add_forward(step_phase)
            # M1/M2: Top-1 opportunistic + Top-K masking via logits
            allowed = compute_allowed_ids(state, schema, vocab, trie, logits)

            if not allowed:
                # Empty set handling: the plan says "attempt repair or
                # break". MVP: break — output stays truncated, success=False.
                break

            # Argmax over allowed + fine pass: drop the best candidate if
            # it fails the char-by-char re-simulation.
            best_id, token_text = _pick_best_token(
                allowed, logits, state, schema, functions, vocab, trie
            )
            if best_id is None:
                # No candidate in allowed passed the fine pass.
                break

            # A "number" closing as an integer ('2,') completes to '2.0'
            # BEFORE committing the chosen closer (no extra forward).
            _inject_float_tail(
                state, schema, token_text, model, vocab, input_ids, emitted_parts
            )

            # ⚠ Documented deviation (see module docstring): commit with
            # DECODED text — the same text the state machine saw. The 4
            # operations travel together in _commit_token: if
            # update_from_text failed (it should not: the filter already
            # validated this token) the loop breaks and the state stays
            # atomic.
            if not _commit_token(
                best_id, token_text, state, schema, input_ids, emitted_parts
            ):
                break

            if state.phase is DecoderPhase.COMPLETE:
                break
        finally:
            # Step timing accrues in the phase it STARTED in (step_phase);
            # running on still honors the continue/break contract.
            if metrics is not None:
                metrics.add_elapsed(
                    step_phase, (perf_counter() - step_start) * 1000.0
                )

    # Only GENERATED tokens (not the prompt): prompt_length was computed
    # before the loop over the real prompt ids. Without [0].tolist(), len()
    # would count tensor ROWS and the slice would drag tokens along.
    generated_ids = input_ids[prompt_length:]
    generated = model.decode(generated_ids)

    return generated, state.phase is DecoderPhase.COMPLETE


def _commit_token(
    token_id: int,
    token_text: str,
    state: DecoderState,
    schema: SchemaContext,
    input_ids: list[int],
    emitted_parts: list[str],
) -> bool:
    """Commit ONE generated token: the 4 operations, always together.

    WHY IT EXISTS:
    - The loop has 2 paths committing a filter-chosen token (M5
      skip-if-single and M1/M2). Both must ALWAYS run the same sequence:
      append the id, advance the state machine, sync the schema to the new
      state, and only then accumulate the text in emitted_parts.
    - emitted_parts is what the fine pass and the oracle's static span use
      to see the output emitted so far. A path forgetting the last row
      would make both see an incomplete output and break silently — that is
      exactly what happened in the whitespace-duplication failure. So the
      sequence lives here, in one place, not replicated per path.
    - INVARIANT ORDER: the id is appended to input_ids BEFORE advancing the
      state. If update_from_text rejects the text (it should not: the
      filter already validated this token) the caller breaks the loop.
      Note: in that case input_ids already holds the id while state and
      emitted_parts do not — "atomic state" in these comments refers to
      state/schema, not input_ids. It is a defensive branch, not the
      normal path.

    Returns:
        True if the token was committed; False if update_from_text rejected
        it (the caller must break the loop).
    """
    input_ids.append(token_id)
    if not state.update_from_text(token_text):
        return False
    schema.update(state)
    emitted_parts.append(token_text)
    return True


_NUMBER_CLOSERS = frozenset(_WS + ",}")


def _inject_float_tail(
    state: DecoderState,
    schema: SchemaContext,
    token_text: str,
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    emitted_parts: list[str],
) -> bool:
    """Complete a "number" the model is about to close as an integer.

    ⚠ WHY A HELPER AND NOT AN if IN THE LOOP: the decoder has TWO commit
    paths — the ambiguous one (argmax over `allowed`) and M5 skip-if-single
    (`len(allowed) == 1`). Both can commit the closing token, so if the
    rule lived in only one of them, the invariant "a 'number' always closes
    as a float" would depend on which path the token took. In
    IN_NUMBER_VALUE, `state.expected_first_chars()` has no wildcard '*'
    (`_number_next_chars` returns literal chars), so the M5 guard DOES hold
    there: the gap is reachable, not theoretical.

    Returns True if injected (state advanced); False if it did not apply.
    """
    tail = _float_tail(state, schema, token_text)
    if tail is None:
        return False
    return _commit_static_text(
        tail, model, vocab, input_ids, state, schema, emitted_parts
    )


def _float_tail(
    state: DecoderState, schema: SchemaContext, token_text: str
) -> str | None:
    """'.0' if ``token_text`` closes an integer literal of a "number" param.

    The grader requires `isinstance(a, float)` for "number": `2` scores 0
    points, `2.0` passes. It only fires when the model ALREADY chose to
    close (the token starts with a closer) and the buffer is a pure integer
    (no '.' or exponent), so a `2.5` or `0.0375` is never touched and the
    numeric value does not change. An "integer" param never enters: its
    expected type is "integer".
    """
    if state.phase is not DecoderPhase.IN_NUMBER_VALUE or state.depth != 1:
        return None
    if schema.current_expected_type() != "number":
        return None
    if not token_text or token_text[0] not in _NUMBER_CLOSERS:
        return None
    if not state.number_buffer.lstrip("-").isdigit():
        return None
    return ".0"


def _commit_static_text(
    static_text: str,
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    state: DecoderState,
    schema: SchemaContext,
    emitted_parts: list[str] | None = None,
) -> bool:
    """Commit pre-tokenized static text without calling the model.

    Despite the old name ("inject_static_header"), this is NOT only the
    header: it is the SINGLE commit point of the static path, and three
    distinct texts flow through it — the static header at startup, the
    closing tail at the end, and the float suffix _inject_float_tail
    delegates. It is called _commit_static_text because what it does is not
    "inject a header" but "commit an oracle decision that needs no model
    query".

    HOW IT WORKS:
    - `model.encode(static_text)` tokenizes the prefix ONCE; each resulting
      id is committed with the same `state.update_from_text` as the main
      loop (same contract: atomic, advances the state machine char by char).
    - `emitted_parts` (optional): when passed, each committed decoded text
      accumulates there. The oracle uses it to compute the trailing
      whitespace already emitted and align its expected texts — without it,
      `_get_next_static_text` could not know which ws already came by
      forward and would duplicate whitespace.
    - Defensive best-effort: the text is valid JSON by construction (same
      grammar `state.py` validates), so it should not fail. If some id does
      not decode or `update_from_text` rejects the text (e.g. an unexpected
      tokenizer split), the injection stops right there — the state stays
      atomic (without that id) and the main loop resumes generating that
      span by normal forward, without crashing.

    Returns:
        True if AT LEAST ONE id was injected (state advanced); False if the
        text contributed no ids or all were rejected. The caller of an
        oracle span MUST check this before `continue`: if it does not
        advance and you continue, the next step sees the SAME state →
        matches the same trigger → failed injection again → infinite loop.
        (The pre-loop header ignores the return: it is called once, outside
        the loop — no such risk.)
    """
    static_ids = model.encode(static_text)[0].tolist()
    advanced = False
    for token_id in static_ids:
        decoded = vocab.id2decoded.get(token_id)
        if decoded is None or not state.update_from_text(decoded):
            break
        input_ids.append(token_id)
        schema.update(state)
        if emitted_parts is not None:
            emitted_parts.append(decoded)
        advanced = True
    return advanced


def _pick_best_token(
    allowed: set[int],
    logits: list[float],
    state: DecoderState,
    schema: SchemaContext,
    functions: list[FunctionDef],
    vocab: Vocab,
    trie: TrieNode,
) -> tuple[int | None, str]:
    """Restricted argmax over `allowed`, with fine pass and retry.

    WHAT IT DOES:
    - Picks the token in `allowed` with the highest logit (restricted
      argmax: the best token the model prefers WITHIN what is allowed). If
      it passes the fine pass, return it.
    - If it does NOT, drop it from the set and repeat with the next best.
      The loop ends two ways: we find a token that passes the fine pass, or
      `allowed` runs out → returns (None, "").

    THE CATCH — the retry only has somewhere to go when `allowed` held more
    than one element, and that depends on which filter branch produced the
    set:
    - M2 (model top-k): up to 2000 candidates → the retry has real
      alternatives and can rescue the generation.
    - M1 (raw argmax, fast path): a single candidate → if that one fails
      the fine pass the loop exhausts the set in one round and returns the
      veto. Not a bug: M1 exists to skip validation when the decision is
      already taken, and its cost is having no plan B. That is why the
      filter accumulates (does not cut) in M2, and why the asymmetry is
      deliberate.

    INVARIANT that makes the loop safe: _passes_fine_validation() mutates
    NEITHER `state` NOR `schema` — it copies the state and validates on a
    fresh SchemaContext. If it contaminated them, the first failed
    candidate would leave a corrupt context and later attempts would
    validate against garbage.

    SIDE EFFECT: this loop consumes `allowed` (discarding as it goes). The
    caller must not reuse that set afterwards — pass a copy if it needs it
    intact.
    """
    while allowed:
        best_id = max(allowed, key=logits.__getitem__)
        token_text = vocab.id2decoded.get(best_id)
        if token_text is not None and _passes_fine_validation(
            functions, schema, state, trie, token_text
        ):
            return best_id, token_text
        allowed.discard(best_id)
    return None, ""


def _passes_fine_validation(
    functions: list[FunctionDef],
    schema: SchemaContext,
    state: DecoderState,
    trie: TrieNode,
    token_text: str,
) -> bool:
    """Char-by-char re-simulation of the WINNING token on a fresh schema.

    WHAT IT IS:
    - The committed state is copied (cheap copy, slots) and a FRESH
      SchemaContext is built (the shared one is NOT touched: if the
      candidate failed halfway it would stay contaminated). The
      _params_object_seen flag is SEEDED from the real schema: the history
      of previous tokens ("did we open parameters already?") cannot be
      re-derived from this token alone.
    - CRITICAL ORDER per character (the same contract as the filter):
      advance the machine (mutates trial) → allows_token(char, trial) with
      the fine schema still synced to the PRE-char state → only after that
      fine.update(trial). If the update ran first, self.* == new_state.*
      inside allows_token and the change/depth clauses (2 and 4) would
      never fire — the fine pass would be a no-op.
    - Why this closes the gaps: the per-TOKEN pass only sees pre
      (committed) and post (simulated); here it sees EVERY intermediate
      state, so the clauses fire where they did not before:
        * Clause 2: the current_key reset ("a" → "" → "b") shows up as a
          change and the prefix check blocks nonexistent/duplicate keys.
        * Clause 3: the intermediate COLON exposes the expected type before
          the value opens ('", "b": "x",' → a string for a number blocks
          at the '"' that opens the string).
        * Clause 1: closing the name inside the SAME token fires the
          is_complete_name branch (the per-token pass lost it when the
          token started outside).
        * Clause 4: depth 0→1→0 inside one token triggers the ⊆ of
          required against keys_enclosed of the intermediate state.
    - COMPLETE without having gone through PARAMS_OBJECT → False (the
      output ALWAYS carries "parameters", even empty for fn_empty).
    """
    trial = copy(state)
    fine = SchemaContext(functions)
    # Previous-token history (was parameters opened?) comes from the real
    # schema; the current token alone cannot decide COMPLETE.
    fine._params_object_seen = schema.has_seen_params_object()
    fine.update(trial)

    for char in token_text:
        if not trial.update_from_text(char):
            # Syntax: the filter already validated the whole token; a char
            # failing here would be a filter bug. Fail closed.
            return False
        # allows_token with PRE-char schema + POST-char state (filter contract)
        if not fine.allows_token(char, trial, trie):
            return False
        # Only now does the fine schema advance to this character's state.
        fine.update(trial)

    if trial.phase is DecoderPhase.COMPLETE and not fine.has_seen_params_object():
        return False
    return True
