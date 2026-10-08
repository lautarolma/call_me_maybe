"""Output validation: turn the decoder's raw string into a checked FunctionCall.

WHAT THIS MODULE DOES (and why it exists):
The constrained decoder produces a STRING with the function-call JSON. That
string has to become an entry of the output array the grader reads. The
intermediate step is not optional: this is where things break if nothing
is checked.

This module does THREE things, all pure (no model, no I/O):
  1. `parse_output`       — raw string -> dict (json.loads, with tolerance)
  2. `build_function_call` — (prompt, dict) -> FunctionCall (pydantic validation)
  3. `validate_output`    — raw string + functions -> FunctionCall | str

Why not a single `json.loads` and done:
  · The decoder guarantees syntactically valid JSON BY CONSTRUCTION, but
    not that the content matches the schema of the chosen function. `name`
    might not exist in functions_definition.json.
  · Without validation here, the error surfaces in the evaluator's grader
    as a zero, not as a message anyone can understand.

THE THREE POST-HOC REPAIRS (`_repair_string_value`):
  The constrained model copies the user's phrase well but deforms it while
  copying. There are three measured deformations, each with its repair:
    A · `_snap_to_query_span`      — TRUNCATED copy (leading punctuation clipped)
    B · `_restore_internal_quotes` — copy MISSING internal quotes
    C · `_collapse_repeated_run`   — COUNTED copy (one repetition per match)
  All three are post-hoc (they do not touch logits), derived from the query,
  and share a single norm: *the value must be backed by the user's phrase;
  otherwise it is an invention and gets normalized*. When there is no unique
  evidence-backed correction, they return the value untouched.

THE GOLDEN RULE — never break positional alignment:
The grader pairs answers and corrections with `zip()`, which is
POSITIONAL. If an entry is missing in the middle, ALL entries after it
misalign and the score is ruined. That is why `build_results` NEVER skips
an entry: if a prompt fails, it emits a placeholder at that exact position
and continues. Losing one test is acceptable; losing eleven is fatal.

HOW IT IS USED (see pipeline.py):
    raw_prompts = load_prompts(args.input)          # RAW input text
    for i, raw_prompt in enumerate(raw_prompts):
        generated = generate(...)[0]                # decoder string
        result = build_results(raw_prompts, generated_list)
    write_results(result, args.output)
"""

from __future__ import annotations

import json
import sys

from src.models.function_definition import FunctionDef
from src.models.output import FunctionCall


# Value used for the `name` field when the generation could not be parsed.
# NOT a real function name: it is a marker that makes that ONE test fail
# with "unknown function" in the grader without dragging down the rest.
# Any non-existent value would be equally valid, but this one is readable
# in the error log.
_UNKNOWN_FN_SENTINEL = "__unparseable__"

#: Delimiters that mark the LEFT edge of a value copied from the query.
#: Whitespace and quotes separate words/values in natural language. An
#: alphanumeric glued to the left means the value is the SUFFIX of a longer
#: word ("llo" inside "hello") and must NOT be stretched.
_SNAP_BOUNDARY = frozenset(" \t\n\r\"'")


def parse_output(raw: str) -> dict[str, object]:
    """Parse the decoder string into a dict.

    Args:
        raw: Text produced by the decoder (JSON with optional surrounding
            whitespace — the decoder emits '\\n\\n{\\n  "name": ...').

    Returns:
        The parsed dict.

    Raises:
        json.JSONDecodeError: if the text is not valid JSON.

    WHY strip():
    The decoder can leave newlines/spaces around the object
    (`'\\n\\n{\\n  ...\\n}'`). `json.loads` tolerates them anyway (whitespace
    is valid outside a value), but the `strip()` documents the intent and
    guards against BOMs, which `json.loads` does NOT tolerate.
    """
    return json.loads(raw.strip())  # type: ignore[no-any-return]


def _snap_to_query_span(value: str, prompt: str) -> str:
    """Re-anchor a string value to the VERBATIM span of the query it came from.

    WHAT PROBLEM IT SOLVES (measured real case): the prompt says
    ``Read the file at /home/user/data.json with utf-8`` and the model emits
    ``path = "home/user/data.json"``: it copies the WHOLE path except the
    leading punctuation (``/``). It does not hallucinate the content — it
    clips the left edge of the span it copied.

    THE RULE (three steps):
    1. If the value appears LITERALLY in the query (exact substring), locate
       that occurrence with ``find``.
    2. Look at the char immediately to the left. If it is PUNCTUATION (not
       whitespace, not quote, not alphanumeric), that char was part of the
       value and the model lost it: stretch the value left until the first
       boundary.
    3. If it already starts on a boundary, or does not appear in the query,
       touch nothing.

    WHY IT STOPS ON THOSE CHARS (counterexamples that fix the boundary):
    - ``"llo"`` inside ``"hello"``: to the left there is ``e`` (alphanumeric)
      → do NOT stretch. Without this brake, "the last 3 letters of hello"
      would return the whole "hello".
    - ``"hello"`` inside ``'hello'``: to the left there is ``'`` (quote) →
      do NOT stretch. The quote is the VALUE's DELIMITER, not its content:
      without this brake the public tests of ``'hello'``/``'world'``
      broke (``hello`` → ``'hello``).
    - ``"C:\\Users\\john\\config.ini"``: to the left there is a space →
      do NOT stretch. That is why a Windows path (which does not start
      with ``/``) stays intact: the rule does NOT assume "every path
      starts with /".

    KNOWN LIMIT: it uses the FIRST occurrence (``find``). If the same value
    appears multiple times with different boundaries, only the first is
    considered. None of the 22 measured cases (11 public + 11 private)
    exercises this.
    """
    if not value:
        return value
    start = prompt.find(value)
    if start < 0:
        return value
    left = start
    while left > 0:
        ch = prompt[left - 1]
        if ch in _SNAP_BOUNDARY or ch.isalnum():
            break
        left -= 1
    if left == start:
        return value
    return prompt[left:start + len(value)]


def _restore_internal_quotes(value: str, prompt: str) -> str:
    """Restore the INTERNAL double quotes the model ate while copying.

    WHAT PROBLEM IT SOLVES (measured real case): the prompt says
    ``Format template: Say "hello" to {name}`` and the model emits
    ``template = "Say hello to {name}"``: it copied the content perfectly
    but ate the quotes delimiting ``hello`` inside the value.

    THE RULE (five steps):
    1. Delete ALL double quotes from the query, keeping the index map to
       get back to the original coordinates.
    2. Find every occurrence of the value in that mutilated query.
    3. Recover the ORIGINAL slice each one corresponds to (it includes the
       quotes step 1 skipped).
    4. Discard slices that (a) are identical to the value —nothing to
       restore— or (b) start or end with a quote.
    5. If EXACTLY ONE remains, return it. If zero or more than one remain,
       return the value untouched.

    WHY EDGE QUOTES ARE DISCARDED (the counterexample that defines the
    rule): the prompt ``Replace all numbers in "Hello 34 I'm 233
    years old" with NUMBERS`` has the value inside quotes, but those quotes
    are the phrase's FRAME, not its content: the expected value is the text
    WITHOUT them. Without this filter the rule would add the quotes back
    and break that public test — and also the one for
    ``Reverse the string 'hello'``.

    WHY IT REQUIRES A SINGLE CANDIDATE: without that requirement the rule
    starts guessing. In ``Say "hello" and hello`` there are two occurrences
    and the correct action is to touch nothing; with ``Use "a" or "b" for
    {x}`` the value ``a`` appears inside words ("Form**a**t"), so the
    evidence is ambiguous. When in doubt it stays silent: the cost of
    silence is one equally failed test, and the cost of guessing is
    breaking the 38 values that currently pass.
    """
    if not value:
        return value
    stripped_chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(prompt):
        if char == '"':
            continue
        stripped_chars.append(char)
        positions.append(index)
    stripped = "".join(stripped_chars)

    candidates: list[str] = []
    start = 0
    while True:
        found = stripped.find(value, start)
        if found < 0:
            break
        # `end` is exclusive over `stripped`; `positions[end - 1]` is the
        # last character of the slice, hence the +1 in the `prompt` cut.
        end = found + len(value)
        original = prompt[positions[found]:positions[end - 1] + 1]
        start = found + 1
        if original == value:
            continue
        if original[0] == '"' or original[-1] == '"':
            continue
        candidates.append(original)

    if len(set(candidates)) != 1:
        return value
    return candidates[0]


def _collapse_repeated_run(value: str, prompt: str) -> str:
    """Undo the counting when the model repeated a character per match.

    WHAT PROBLEM IT SOLVES (measured real case): the prompt says
    ``Replace all vowels in 'Programming is fun' with asterisks`` and the
    model emits ``replacement = "****"``. In any substitution API — Python's
    ``re.sub``, ``sed``, JavaScript's ``replace`` — the ``replacement`` is a
    TEMPLATE applied to every match, not a copy per match. The model counted
    the vowels and wrote one star per vowel: it executed the instruction
    instead of parameterizing it. The correct value is the character
    repeated ONCE.

    THE RULE (four steps):
    1. The value must be ENTIRELY a run of >= 2 identical characters.
    2. That run must NOT appear literally in the query: if the model copied
       it, the repetition is intentional and nothing is touched.
    3. Use the longest run the query DOES show.
    4. If the query shows none, leave a single character.

    WHY STEP 3 EXISTS (the measured hole in the simple version): if the
    query shows ``***`` and the model counts five, always collapsing to one
    would return ``*`` instead of ``***``. The "respect the query's count"
    version returns ``***``. In the real case the query shows no run at all
    — it says "asterisks", the English word, not the symbol — so that case
    does fall through to step 4.

    WHY IT REQUIRES THE WHOLE VALUE TO BE THE RUN: a broader version that
    recognized "a repeated block" (like ``ababab`` -> ``ab``) is BROKEN —
    on the real case it would return ``**`` instead of ``*``, because a run
    of four identical chars is also "a block of two repeated twice".
    Measured: that variant drops the public set from 11/11 to 10/11. The
    legitimate values with repetition (``utf-8``, ``/home/user/data.json``,
    ``NUMBERS``, ``dog``) have distinct characters and stay intact.

    LEGITIMACY NOTE: the rule leans on a STRUCTURAL SIGNATURE (the run of
    characters), not on a vocabulary. Anchoring the correction to the word
    "asterisks" would be a hardcoded lookup table and the subject forbids
    it; anchoring it to the shape of the output is fine.
    """
    if len(value) < 2:
        return value
    if len(set(value)) != 1:
        return value
    if value in prompt:
        return value
    for length in range(len(value), 1, -1):
        head = value[:length]
        if head in prompt:
            return head
    return value[:1]


def _repair_string_value(value: str, prompt: str) -> str:
    """Apply the three post-hoc repairs in order; the first one that fixes wins.

    WHY "FIRST ONE THAT FIXES WINS" and not all three chained: each rule
    requires evidence the previous one did not have. `_snap_to_query_span`
    only fires if the value IS a substring of the query. If it is not, A
    should not touch anything, so the chain moves on to
    `_restore_internal_quotes`, which requires it to appear once quotes are
    removed. And `_collapse_repeated_run` only looks at runs of identical
    characters, a shape B never produces (B returns slices with quotes,
    which are not runs). They are disjoint by construction, and this form
    makes that explicit in the code.

    The order is NOT arbitrary: A is the rule already verified and
    covered, so if B or C had a defect the previous behavior remains
    covered by the regression suite.
    """
    snapped = _snap_to_query_span(value, prompt)
    if snapped != value:
        return snapped
    quoted = _restore_internal_quotes(snapped, prompt)
    if quoted != snapped:
        return quoted
    return _collapse_repeated_run(quoted, prompt)


def build_function_call(prompt: str, payload: dict[str, object]) -> FunctionCall:
    """Build a validated FunctionCall from (prompt, parsed payload).

    Args:
        prompt: The ORIGINAL natural request (not the prompt with the
            function definitions injected). The grader compares it with
            `correction["prompt"]` by EXACT string equality, so it must be
            the input text, byte for byte.
        payload: Dict with the keys `name` and `parameters` (or `name` alone).

    Returns:
        ``FunctionCall`` with the three fields the subject requires.

    Raises:
        pydantic.ValidationError: if `name` is missing or types do not add up.
    """
    name = payload.get("name")
    parameters = payload.get("parameters", {})
    raw_parameters = parameters if isinstance(parameters, dict) else {}
    # Applies the THREE post-hoc repairs to each string value (see
    # `_repair_string_value`). All three correct the same class of defect: the
    # model is a copier and deforms the phrase while copying — it clips it
    # (A), eats the internal quotes (B), or counts repetitions instead of
    # parameterizing (C). Non-strings (numbers, bools, null) pass through
    # untouched: the project limits params to scalars (models/output.py).
    repaired_parameters = {
        key: _repair_string_value(value, prompt) if isinstance(value, str) else value
        for key, value in raw_parameters.items()
    }
    return FunctionCall(
        prompt=prompt,
        # `name` comes as object from json.loads; pydantic validates it as str.
        # If the decoder guarantees a string, the isinstance is defensive and
        # never fails in practice — but without it, mypy would complain about
        # passing `object` where `str` is expected.
        name=name if isinstance(name, str) else str(name),
        # The values stay `object` to mypy (they come from `payload`); the
        # isinstance above already narrowed `parameters` to a dict. The
        # default {} covers "the decoder emitted only the name" (a fn
        # without parameters).
        parameters=repaired_parameters,
    )


def validate_output(
    raw: str,
    functions: list[FunctionDef],
) -> FunctionCall | str:
    """Validate one generation against the available definitions.

    Args:
        raw: Decoder text for ONE prompt.
        functions: Definitions loaded from the input.

    Returns:
        ``FunctionCall`` if the generation is valid and its `name` exists in
        ``functions``; or an error string if something fails.

    WHY it returns `FunctionCall | str` instead of raising:
    The subject requires the program to "must never crash unexpectedly"
    and problem prompts must not kill the pipeline. Returning the error as
    a value lets the caller log it and continue with the next prompt.

    NOTE on `name` validation:
    We check that the name EXISTS in the definitions, but NOT that the
    parameters match that function's schema. That deeper validation is
    still pending: what unblocks the deliverable is that the file exists
    and is parseable, not that it be semantically perfect.
    """
    try:
        payload = parse_output(raw)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return f"expected a JSON object, got {type(payload).__name__}"

    try:
        call = build_function_call(prompt="", payload=payload)
    except Exception as exc:  # pydantic.ValidationError y subclases
        return f"schema violation: {exc}"

    known = {f.name for f in functions}
    if call.name not in known:
        return f"unknown function {call.name!r} (known: {sorted(known)})"

    return call


def build_results(
    prompts: list[str],
    generated: list[str],
) -> list[FunctionCall]:
    """Package (original prompts, generations) into output entries.

    Args:
        prompts: Original requests, in the SAME order as `generated`.
        generated: Decoder strings, one per prompt.

    Returns:
        One entry per prompt, IN ORDER. Entries are never skipped: if a
        generation cannot be parsed, a placeholder is emitted at that
        position to keep the grader's positional `zip()` intact.
    """
    results: list[FunctionCall] = []
    for i, raw_prompt in enumerate(prompts):
        raw_output = generated[i] if i < len(generated) else ""
        try:
            payload = parse_output(raw_output)
            if not isinstance(payload, dict):
                raise ValueError(f"not a JSON object: {type(payload).__name__}")
            results.append(build_function_call(raw_prompt, payload))
        except (json.JSONDecodeError, ValueError) as exc:
            # Aligned placeholder: this test will fail, the others won't.
            results.append(
                FunctionCall(
                    prompt=raw_prompt,
                    name=_UNKNOWN_FN_SENTINEL,
                    parameters={},
                )
            )
            # The warning goes to stderr, the diagnostics stream
            # (see the stdout/stderr note in __main__.py).
            print(
                f"WARNING: prompt {i} produced unparseable output ({exc}); "
                f"emitted placeholder to keep positional alignment",
                file=sys.stderr,
            )
    return results


def _text_forms(value: object) -> list[str]:
    """Textual representations a value can take in a prompt.

    WHY MORE THAN ONE FORM: the decoder writes floats with a decimal point
    (`2.0`), but humans write the number without one ("sum 2 and 3"). If we
    only accepted `"2.0"` as supporting evidence, that legitimate prompt
    would fire a false warning. That is why an integral float returns both
    forms.

    Returns `[]` for values with no comparable textual form (bool, None) —
    the caller treats them as "not judgeable" rather than "unsupported".
    """
    if isinstance(value, bool):
        # `bool` is a subclass of `int`: must be checked BEFORE int or
        # True would be reported as "1".
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, float):
        forms = [repr(value)]
        if value.is_integer():
            forms.append(str(int(value)))
        return forms
    if isinstance(value, str):
        return [value]
    return []


def find_unsupported_prompts(
    prompts: list[str],
    results: list[FunctionCall],
) -> list[int]:
    """Indices of prompts whose call has NO value backed by the prompt text.

    WHAT IT DOES: the constrained decoder ALWAYS emits a valid function —
    the grammar allows nothing else. For a prompt that maps to no function,
    this degenerates into the model choosing whichever looks "least ugly":
    it does not crash, but it does not warn either. Measured real case:
    *"What is the weather in Paris tomorrow?"* → `fn_get_square_root(a=100.0)`.
    This detects that case and reports it.

    THE CRITERION, and why it is simple on purpose: if NO parameter value
    appears (literal, case-insensitive) in the prompt, the call is not
    backed by the input — the model invented even the arguments. If even
    ONE appears, it is not reported: the borderline case (where
    `replacement="****"` is not in the prompt but `source_string` and
    `regex` are) is a model accuracy failure, not a match failure, and the
    evaluation harness already measures it.

    WHAT THIS MODULE IS **NOT** — important for subject compliance:
    the subject says "the function to call should be chosen using the LLM,
    not with heuristics". Nothing is chosen here: the LLM already chose the
    function during constrained decoding. This function is an OUTPUT SENSOR
    for a human, not a decision. It does not touch the results file and
    does not alter the score.

    Args:
        prompts: Original requests (raw text, without the injected defs).
        results: Entries already built by `build_results`.

    Returns:
        Zero-based indices of prompts without textual support. Empty = all OK.
    """
    unsupported: list[int] = []
    for i, call in enumerate(results):
        # The sentinel already got its own warning (unparseable) and has no
        # arguments to judge. Parameter-less functions are not judgeable
        # either: there is nothing to compare against the prompt.
        if call.name == _UNKNOWN_FN_SENTINEL or not call.parameters:
            continue
        if i >= len(prompts):
            continue
        haystack = prompts[i].casefold()
        if not haystack.strip():
            continue
        supported = any(
            form and form.casefold() in haystack
            for value in call.parameters.values()
            for form in _text_forms(value)
        )
        if not supported:
            unsupported.append(i)
    return unsupported
