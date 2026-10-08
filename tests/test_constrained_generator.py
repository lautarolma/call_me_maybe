"""Tests for the constrained generator + fine pass.

INTENT OF THESE TESTS (in a nutshell):
- They test the generator with a FakeModel that advances a fixed token
  sequence (high logits for the next token of the sequence, low for the
  rest): the argmax picks the target when it is in allowed.
- HALF of the tests are for the FINE PASS: the gaps the filter lets through
  by design (abstention on boundary states) MUST be rejected by the
  char-by-char re-simulation of the winner:
    * gap 2 residual 2: a token that enters parameters + first key in one
      token ('", "parameters": {"a') — a valid key passes, an invalid key is
      blocked (before, the invalid one slipped through).
    * gap 3: key+value+closing complete in one token (', "b": "x",' with
      "b": number) — the fine pass blocks it; the filter allowed it.
    * gap 4: enter AND exit parameters in one token — without all the
      required keys it is blocked; with all of them, it passes.
    * exact-duplicate slip ('"a": 4' with "a" already emitted) — blocked.
    * plan gap: output COMPLETE without going through PARAMS_OBJECT
      ('}' right after the name) — blocked.
- Each token in the mock vocab exists ONLY for the case it exercises; the
  real Qwen vocab has thousands of such tokens (~20-30 chars are rare in
  BPE, which is why the filter tolerates them and the fine pass covers them
  at the cost of 1 re-simulation per step).
"""

from __future__ import annotations

from src.decoder.constrained_generator import (
    _float_tail,
    _inject_float_tail,
    _commit_static_text,
    _get_next_static_text,
    _passes_fine_validation,
    generate,
)
from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.trie import TrieNode, build_trie
from src.loader.vocab_loader import (
    BYTE_CATEGORY,
    Vocab,
    _STATIC_PHASE_FIRST_CHARS,
)
from src.models.function_definition import FunctionDef, ParameterDef

# Replica of data/input/functions_definition.json (subset used in tests).
FUNCTIONS = [
    FunctionDef(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": ParameterDef(type="number"),
            "b": ParameterDef(type="number"),
        },
        returns={"type": "number"},
    ),
    FunctionDef(
        name="fn_greet",
        description="Greet someone by name.",
        parameters={
            "name": ParameterDef(type="string"),
        },
        returns={"type": "string"},
    ),
    FunctionDef(
        name="fn_empty",
        description="Function without parameters.",
        parameters={},
        returns={"type": "null"},
    ),
]

# Mock vocab: id -> DECODED text (what the state machine sees).
VOCAB: dict[int, str] = {
    # Structure / literals
    1: "{",
    2: " ",
    3: "}",
    4: ",",
    5: ":",
    6: '"',
    7: "\n  ",  # newline + indent 2 (natural format of the static span)
    # Keys of the output object
    10: "name",
    11: "parameters",
    # Function names
    20: "fn_",
    27: "fn_add_numbers",
    28: "fn_greet",
    24: "empty",
    29: "fn_empty",
    # Mixed structure+content tokens
    30: '": "',
    32: ', "b": 3',
    33: '"a": 2.0',
    # Keys of parameters
    40: '"a"',
    41: '"b"',
    43: '"a": 4',              # EXACT duplicate of "a" (documented slip)
    # Values
    50: '"x"',
    51: "2.0",
    52: '"Javier"',
    53: "true",
    # Keys + structure
    70: '"name": ',
    71: '"parameters": {',
    72: ', "parameters": {',
    # FINE PASS cases — multi-phase tokens of ~19-28 chars
    73: ', "parameters": {"a',       # enters params + 1st key "a" (valid)
    74: ', "parameters": {"zz',      # same with a NONEXISTENT key (gap 2)
    75: ', "b": "x",',               # key+value string+closing for b:number (gap 3)
    76: ', "b": 4,',                 # same with correct type -> fine pass accepts
    77: ', "parameters": {"a": 1}',   # enters AND exits params, "b" missing (gap 4)
    78: ', "parameters": {"a": 1, "b": 2}',  # same with all required keys -> passes
    79: ", ",
    # ORACLE spans: canonicals + standalone value of the E2E
    80: "\n  }\n}",          # T5: closes parameters + closes the ROOT
    81: "3",                 # value of "b" in the E2E (T4 already emitted '"b": ')
    82: "\n}",               # T6: closes the ROOT (fn_empty, INLINE {} format)
}

IDS: dict[str, int] = {text: tid for tid, text in VOCAB.items()}

BYTE_IDS: frozenset[int] = frozenset()  # this mock does not model <byte> tokens


def build_vocab() -> Vocab:
    """Build the mock Vocab mimicking the indexing of vocab_loader.py."""
    id2decoded: dict[int, str] = {}
    starting: dict[str, set[int]] = {}
    for tid, text in VOCAB.items():
        if tid not in BYTE_IDS:
            id2decoded[tid] = text
            starting.setdefault(text[0], set()).add(tid)
    starting.setdefault(BYTE_CATEGORY, set()).add(60)
    valid_by_phase = {
        phase: {
            tid
            for ch in phase_chars
            for tid in starting.get(ch, set())
        }
        for phase, phase_chars in _STATIC_PHASE_FIRST_CHARS.items()
    }
    return Vocab(
        token2id={text: tid for tid, text in VOCAB.items()},
        id2token={tid: text for tid, text in VOCAB.items()},
        id2decoded=id2decoded,
        tokens_starting_with=starting,
        vocab_size=max(VOCAB) + 1,  # arbitrary ids in the mock: cover the maximum
        valid_by_phase=valid_by_phase,
    )


def make_pipeline() -> tuple[DecoderState, SchemaContext, Vocab, TrieNode]:
    vocab = build_vocab()
    trie = build_trie([fn.name for fn in FUNCTIONS])
    return DecoderState(), SchemaContext(FUNCTIONS), vocab, trie


def step(state: DecoderState, schema: SchemaContext, text: str) -> None:
    """Advance the state with a token and sync the schema (generator contract)."""
    assert state.update_from_text(text), f"state machine rejected {text!r}"
    schema.update(state)


def at_params(
    fn_name: str = "fn_add_numbers",
) -> tuple[DecoderState, SchemaContext, Vocab, TrieNode]:
    """State at VALUE_END after the name (depth 0, function selected)."""
    state, schema, vocab, trie = make_pipeline()
    for t in ("{", '"name": ', '"', fn_name, '"'):
        step(state, schema, t)
    return state, schema, vocab, trie


def set_params_ctx() -> tuple[DecoderState, SchemaContext, Vocab, TrieNode]:
    """State PARAMS_OBJECT with selected_function (fn_add_numbers) and empty keys."""
    state, schema, vocab, trie = at_params("fn_add_numbers")
    step(state, schema, ', "parameters": {')
    return state, schema, vocab, trie


class TestFineValidationGap2EnterParamsWithFirstKey:
    """Fine pass: token that enters parameters + reads the 1st key (gap 2)."""

    def test_valid_first_key_passes(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # "a" enters parameters in the SAME token that opens the object: the
        # fine pass validates it (before, the filter let it slip through).
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "parameters": {"a')

    def test_invalid_first_key_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # "zz" is not a prefix of any available key: the prefix check of the
        # char-by-char walk blocks it (gap 2 residual 2 CLOSED).
        assert not _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"zz'
        )


class TestFineValidationGap3CompleteKeyValue:
    """Fine pass: key+value+closing complete in one token (gap 3)."""

    def test_wrong_type_blocked(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')  # IN_NUMBER_VALUE, "a" open
        # ', "b": "x",' opens a string for b:number: the intermediate COLON
        # exposes the type and the '"' that opens the string is blocked.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "b": "x",')

    def test_right_type_passes(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "b": 4,')


class TestFineValidationGap4EnterAndExitParams:
    """Fine pass: enter AND exit parameters in one token (gap 4)."""

    def test_missing_required_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # The whole parameters object in one token (starting from VALUE_END
        # requires the comma: from here only ',' or '}' are valid), without "b":
        # the closing '}' (intermediate depth 1->0) triggers clause 4 -> "b"
        # missing.
        assert not _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"a": 1}'
        )

    def test_all_required_passes(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        assert _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"a": 1, "b": 2}'
        )


class TestFineValidationDuplicateSlip:
    """Fine pass: the exact-duplicate slip is closed in the fine pass."""

    def test_identical_duplicate_key_blocked(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")  # closes "a" -> PARAMS_OBJECT
        # The filter allowed it (documented slip); the char-by-char walk sees
        # the reset ""->"a" -> available no longer contains "a" -> blocked.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, '"a": 4')


class TestFineValidationMissingParamsObject:
    """Fine pass: COMPLETE without going through PARAMS_OBJECT."""

    def test_close_without_params_object_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_empty")
        # '}' closes the output object directly: we never saw the parameters '{'.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, "}")

    def test_full_empty_params_close_passes(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_empty", '"', ', "parameters": {'):
            step(state, schema, t)
        step(state, schema, "}")  # closes empty parameters (depth 1->0)
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, "}")  # closes output


class TestFineValidationNameEscapeRejected:
    """An escape inside the "name" value must not slip through forever.
    name_buffer does not touch it (it is skipped in state.py), so without the
    explicit guard in _allows_name_value the trie keeps seeing a valid prefix
    and the generator never closes the string (real repro: 'Greet shrek' ->
    200 forwards on '\\n\\t\\t\\t...' without completing)."""

    def test_escape_inside_name_value_blocked(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "f"):
            step(state, schema, t)
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, "\\n")

    def test_plain_continuation_of_name_still_passes(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "f"):
            step(state, schema, t)
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, "n_greet")


class _FakeTensor:
    """Dummy that REPLICATES the 2D shape [1, N] of the real encode() tensor.

    Small_LLM_Model.encode() builds torch.tensor([ids]) -> 2D; its .tolist()
    returns list[list[int]]. Before, this dummy returned a FLAT list (1D) and
    the dimension bug went unnoticed in the suite.
    """

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __getitem__(self, idx: int) -> _FakeRow:
        # t[0] of a 2D tensor [1, N] -> 1D view [N]
        return _FakeRow(self._ids)

    def tolist(self) -> list[list[int]]:
        return [list(self._ids)]


class _FakeRow:
    """1D view of a tensor row (t[0].tolist() -> flat list[int])."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return list(self._ids)


class FakeModel:
    """Mock of Small_LLM_Model: pushes a fixed token sequence.

    encode(prompt) returns a single dummy id (prompt_length = 1); each
    get_logits_from_input_ids assigns a high logit to the next token of the
    expected sequence and -100 to the rest. That way the argmax picks the
    target ONLY if the target is in allowed (otherwise it picks another token
    and the generation diverges — the test detects it).
    """

    def __init__(self, vocab: Vocab, sequence: list[str]) -> None:
        self._vocab = vocab
        self._sequence = sequence

    def encode(self, text: str) -> _FakeTensor:
        return _FakeTensor([999])  # dummy 1-id prompt

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab.id2decoded[tid] for tid in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        # FORMS contract with the real SDK: get_logits expects a FLAT list[int].
        # If generate() stopped flattening ([0].tolist()), a list[list[int]]
        # would arrive here and this assert would paint the suite red.
        assert all(isinstance(x, int) for x in input_ids), (
            "get_logits_from_input_ids must receive a flat list[int], "
            f"not {type(input_ids[0]).__name__}"
        )
        step = len(input_ids) - 1  # prompt_length == N of encode
        logits = [-100.0] * self._vocab.vocab_size
        if step < len(self._sequence):
            logits[IDS[self._sequence[step]]] = 100.0
        return logits


class TestGenerator:
    def test_fn_add_numbers_end_to_end(self) -> None:
        """Acceptance criteria: prompt -> JSON with "fn_add_numbers", success."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = FakeModel(
            vocab,
            [
                "{",
                '"name": ',
                '"',
                "fn_add_numbers",
                '"',
                ', "parameters": {',
                '"a": 2.0',
                ', "b": 3',
                "}",
                "}",
            ],
        )
        generated, ok = generate(model, "What is 2+3?", vocab, FUNCTIONS, trie)
        assert ok
        assert "fn_add_numbers" in generated

    def test_zero_max_tokens_fails_cleanly(self) -> None:
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = FakeModel(vocab, ["{"])
        generated, ok = generate(model, "hi", vocab, FUNCTIONS, trie, max_tokens=0)
        assert not ok
        assert generated == ""

    def test_garbage_target_is_replaced_by_second_best(self) -> None:
        """If the model's target is NOT allowed, a valid token is still
        generated (argmax over allowed) or it stops — never broken JSON."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        # Target "fn_" at ROOT: not allowed (only '{' and ws) -> '{' is picked.
        model = FakeModel(vocab, ["fn_"])
        generated, ok = generate(model, "boo", vocab, FUNCTIONS, trie, max_tokens=3)
        assert not ok  # no COMPLETE within 3 tokens
        assert generated.startswith("{")  # the winning token was '{'

    def test_n_token_prompt_does_not_leak_into_generated(self) -> None:
        """Regression: an N-token prompt does NOT leak into the output.

        encode() returns a 2D tensor with ids that do NOT exist in the test
        vocab (999/777/555). If generated_ids included prompt tokens (wrong
        prompt_length), FakeModel.decode would raise KeyError — a noisy
        failure. With the fix, prompt_length == 3 and generated only has '{'.
        """
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])

        class _PromptfulModel(FakeModel):
            """encode() of 3 tokens — none exists in VOCAB (KeyError if any
            reached generated_ids)."""

            def encode(self, text: str) -> _FakeTensor:  # noqa: D102
                return _FakeTensor([999, 777, 555])

        generated, ok = generate(
            _PromptfulModel(vocab, ["{"]),
            "long prompt", vocab, FUNCTIONS, trie, max_tokens=1,
        )
        assert not ok  # only 1 token generated: no COMPLETE
        # The only generated token was '{' (id 1): no prompt in the output.
        assert generated == "{"


# ─── Oracle by state (level-1 static spans) ───────────────────────────────
# _get_next_static_text(state, schema, generated_text) walks
# _STATIC_TEXT_RULES (T1-T6, PURE functions) and returns the canonical of the
# first matching domain, aligned to emitted's trailing ws (E). These tests
# verify:
#   * the canonical of EACH span;
#   * the DISJOINT domains (N gate verified against state.py);
#   * the E alignment (never duplicate ws);
#   * the post-injection state with state.simulate(C) as the source of truth
#     (NO hand-written tables — a desync with state.py is a bug).

T1_CANON_ADD = ',\n  "parameters": {\n    "a": '
T1_CANON_GREET = ',\n  "parameters": {\n    "name": "'
T2_CANON_ADD = '\n  "parameters": {\n    "a": '
T3_CANON_FIRST = '\n    "a": '
T3_CANON_NEXT = '\n    "b": '
T4_CANON_NEXT = ',\n    "b": '
T5_CANON = "\n  }\n}"
T6_CANON = "\n}"

# Old canonicals (linear span) — they can ONLY come from the oracle.
_OLD_TAIL = ',\n  "parameters": {'

# id2decoded of the mock vocab (split chosen for the test; the real Qwen
# encode produces its own split and best-effort handles it anyway).
_ORACLE_IDS: dict[str, list[int]] = {
    T1_CANON_ADD: [4, 7, 71, 7, 2, 2, 40, 5, 2],
    T3_CANON_NEXT: [7, 2, 2, 41, 5, 2],
    T4_CANON_NEXT: [4, 7, 2, 2, 41, 5, 2],
    T5_CANON: [80],
    T6_CANON: [82],
}


class TestLevel1Oracle:
    """Level-1 table: each span answers ITS canonical from its domain."""

    def test_t1_post_name_value_end(self) -> None:
        # Nominal: "name" value closed -> VALUE_END d0, N=1 -> comma +
        # parameters opening + the FIRST key (fusion T1+ENTRY(K1)).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _get_next_static_text(state, schema, "") == T1_CANON_ADD

    def test_t1_string_param_opens_with_quote(self) -> None:
        # fn_greet: its ONLY param is "name" of type string -> OP(k)='"'.
        # (The param has the same name as the output object key: N=1 still
        # holds because it is the d0 VALUE_END — the param lives at depth 1.)
        state, schema, _vocab, _trie = at_params("fn_greet")
        assert _get_next_static_text(state, schema, "") == T1_CANON_GREET

    def test_t2_fused_comma_token(self) -> None:
        # Fused BPE token ('fn_add_numbers",') -> IN_OBJECT directly (without
        # going through VALUE_END): T2 covers the jump (N=1 holds).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        step(state, schema, ",")
        assert state.phase is DecoderPhase.IN_OBJECT
        assert _get_next_static_text(state, schema, "") == T2_CANON_ADD

    def test_empty_params_return_none(self) -> None:
        # Free span: ord=∅ -> T1/T2 return None — the model generates
        # "parameters": {} with its FUSED token (tid 6257); injecting a loose
        # '{' would be a seam-type defect. fn_empty is the only case in the
        # subject; the real probe confirmed the INLINE {} format.
        state, schema, _vocab, _trie = at_params("fn_empty")
        assert _get_next_static_text(state, schema, "") is None

    def test_t3_first_required_key(self) -> None:
        # PARAMS_OBJECT d1 with pending keys -> ENTRY(Knext).
        state, schema, _vocab, _trie = set_params_ctx()
        assert _get_next_static_text(state, schema, "") == T3_CANON_FIRST

    def test_t3_after_comma(self) -> None:
        # The model emitted the comma after "a" -> PARAMS_OBJECT again (in this
        # decoder VALUE_END + ',' -> PARAMS_OBJECT, NOT IN_OBJECT) -> T3 gives
        # the NEXT required key. This is the flow of NUMBER values:
        # IN_NUMBER_VALUE (open number) -> the model pays the comma -> T3.
        state, schema, _vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")
        assert state.phase is DecoderPhase.PARAMS_OBJECT
        assert _get_next_static_text(state, schema, "") == T3_CANON_NEXT

    def test_t4_next_key_after_string_value(self) -> None:
        # VALUE_END d1 with pending keys -> comma + next key. Only a value that
        # CLOSES in its token reaches VALUE_END d1 (string); a number leaves
        # IN_NUMBER_VALUE and the flow resumes at T3 after the comma.
        fn = FunctionDef(
            name="fn_text",
            description="",
            parameters={
                "x": ParameterDef(type="string"),
                "y": ParameterDef(type="string"),
            },
            returns={"type": "string"},
        )
        state, _schema, _vocab, _trie = make_pipeline()
        schema = SchemaContext([fn])
        for t in ("{", '"name": ', '"', "fn_text", '"', ', "parameters": {'):
            step(state, schema, t)
        step(state, schema, '"x": "Javier"')  # closes string -> VALUE_END d1
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _get_next_static_text(state, schema, "") == ',\n    "y": "'

    def test_t5_close_after_last_string_value(self) -> None:
        # VALUE_END d1 with no pending keys -> closes parameters + the ROOT.
        # fn_greet: its ONLY param is string -> T5 gives the close.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _get_next_static_text(state, schema, "") == T5_CANON

    def test_t6_close_root_after_empty_params(self) -> None:
        # fn_empty: after the '}' of empty parameters (INLINE format of the
        # real probe) the ROOT is still to close. Without T6 the model must
        # emit that '}' by forward (probe: SUCCESS=False). T6 gives it.
        state, schema, _vocab, _trie = at_params("fn_empty")
        step(state, schema, ', "parameters": {')
        step(state, schema, "}")
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _get_next_static_text(state, schema, "") == T6_CANON

    def test_t6_close_root_after_full_params(self) -> None:
        # Post-parameters with all keys: VALUE_END d0, N=0 (current_key is the
        # last params key, not "name"), ρ=0, P=1 -> closes the ROOT.
        state, schema, _vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        step(state, schema, "}")  # closes parameters -> d0
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _get_next_static_text(state, schema, "") == T6_CANON

    def test_t6_blocked_while_name_open(self) -> None:
        # N=1 (post-name, parameters not yet open) -> T6 None: the ROOT does
        # NOT close without parameters (the fine pass blocks it if the model
        # tried by forward). fn_empty at VALUE_END post-name: T1 already gave
        # None (ord=∅) and T6 requires N=0 -> all None -> continues by forward.
        state, schema, _vocab, _trie = at_params("fn_empty")
        assert _get_next_static_text(state, schema, "") is None

    def test_t6_fused_empty_params_token(self) -> None:
        # Finding of the real probe: the FUSED token ', "parameters": {}'
        # brings '{'+'}' in ONE chunk -> has_seen_params_object() stays False
        # (schema.update runs POST-token and never sees the intermediate
        # PARAMS_OBJECT). P cannot gate T6 — with N=0 ∧ ρ=0 the ROOT can only
        # be closed (the fine pass of the normal path covers the edge).
        state, schema, _vocab, _trie = at_params("fn_empty")
        step(state, schema, ', "parameters": {}')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert not schema.has_seen_params_object()  # the flag did NOT see the '{'
        assert _get_next_static_text(state, schema, "") == T6_CANON

    def test_bug013_inner_param_name_not_an_opener(self) -> None:
        # Regression: the inner "name" param of fn_greet (depth 1) does NOT
        # trigger T1/T2 (the OPENING spans require N=1: depth==0 ∧
        # keys_enclosed==∅ — on closing the param, keys_enclosed={'name'}).
        # And since it is the ONLY param, ρ=0 -> T5 does answer (the object is
        # COMPLETE and must be closed): the oracle NEVER duplicates the opening
        # span — that was the bug class of the old trigger.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _get_next_static_text(state, schema, "") == T5_CANON

    def test_initial_root_does_not_match(self) -> None:
        state, schema, _vocab, _trie = make_pipeline()
        assert _get_next_static_text(state, schema, "") is None

    def test_insertion_order_of_parameters(self) -> None:
        # ord = tuple(F.parameters): the JSON order is the dict order
        # (insertion order), NOT alphabetical — "b" declared first comes out
        # first.
        fn = FunctionDef(
            name="fn_ordered",
            description="",
            parameters={
                "b": ParameterDef(type="number"),
                "a": ParameterDef(type="number"),
            },
            returns={"type": "number"},
        )
        state, _schema, _vocab, _trie = make_pipeline()
        schema = SchemaContext([fn])
        for t in ("{", '"name": ', '"', "fn_ordered", '"'):
            step(state, schema, t)
        assert _get_next_static_text(state, schema, "") == ',\n  "parameters": {\n    "b": '

    def test_align_trims_emitted_ws(self) -> None:
        # E: if emitted already ends in the canonical's ws (the model put it),
        # the span does NOT duplicate it (never duplicate ws) -> injects only
        # the rest. T5 with emitted='\n  ' -> e='\n  ' -> subtracts '}\n}'.
        # The 1st char of emitted is part of the T5 canon -> the model is
        # ALREADY where the span is.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')  # only param (string): ρ=0 -> VALUE_END d1
        assert _get_next_static_text(state, schema, "\n  ") == "}\n}"

    def test_t2_aligns_fused_comma_indent(self) -> None:
        # The fused token '",\n  ' (comma + model indent in ONE token) leaves
        # IN_OBJECT d0 N=1 with emitted=',\n  ' -> T2 aligns and trims the
        # '\n  ' of the canonical: injects ONLY '"parameters": {\n    "a": '.
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        step(state, schema, ",")  # IN_OBJECT d0 (the ',' does not reset the key)
        assert state.phase is DecoderPhase.IN_OBJECT and state.depth == 0
        assert _get_next_static_text(state, schema, ",\n  ") == (
            '"parameters": {\n    "a": '
        )

    def test_align_unmatchable_ws_falls_to_forward(self) -> None:
        # emitted ends in ws that is NOT a prefix of the canonical (e.g. the
        # model already emitted the IN_OBJECT indent and the T1 comma is next):
        # alignment impossible -> None -> normal forward (safe, without forcing
        # spans).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert _get_next_static_text(state, schema, "\n  ") is None


class TestOracleDomainDisjunction:
    """The span domains are disjoint by construction (phase × depth × gates):
    for a given state, ONE span answers. These are the pairs that shared a
    phase and are distinguished only by the gates."""

    def test_t1_vs_t6_same_phase_different_gate(self) -> None:
        # VALUE_END d0: T1 answers ONLY with N=1 (post-name, params to open);
        # T6 ONLY with N=0 ∧ ρ=0 ∧ P=1 (post-parameters).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert _get_next_static_text(state, schema, "") == T1_CANON_ADD
        state2, schema2, _v2, _t2 = set_params_ctx()
        step(state2, schema2, '"a": 2.0')
        step(state2, schema2, ', "b": 3')
        step(state2, schema2, "}")
        assert _get_next_static_text(state2, schema2, "") == T6_CANON

    def test_t3_vs_t6_pending_keys_gate(self) -> None:
        # PARAMS_OBJECT d1 is T3 (ρ=1); with ρ=0 it does not match T3 -> if the
        # close comes from the model, T6 complements it only at d0.
        state, schema, _vocab, _trie = set_params_ctx()
        assert _get_next_static_text(state, schema, "") == T3_CANON_FIRST


class TailFakeModel(FakeModel):
    """FakeModel whose encode() tokenizes ONLY the oracle canonicals.

    The prompt and the STATIC_HEADER (header 1) return the dummy id 999 like
    the base FakeModel — 999 does not exist in the mock vocab id2decoded, so
    header 1 is never injected (same behavior as the existing tests). The
    _ORACLE_IDS canonicals are returned with mock vocab ids so they can be
    injected through the same path as in production.
    """

    def __init__(self, vocab: Vocab, sequence: list[str]) -> None:
        super().__init__(vocab, sequence)
        self.calls = 0

    def encode(self, text: str) -> _FakeTensor:
        if text in _ORACLE_IDS:
            return _FakeTensor(list(_ORACLE_IDS[text]))
        return _FakeTensor([999])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.calls += 1
        return super().get_logits_from_input_ids(input_ids)


class TestInjectOracleText:
    """_commit_static_text with an oracle canonical: the post state must match
    EXACTLY state.simulate(C) (source of truth; a hand-written table desynced
    with state.py would be a silent bug)."""

    def _state_tuple(self, s: DecoderState) -> tuple[object, object, object, object]:
        return (s.phase, s.depth, s.current_key, s.keys_enclosed)

    def test_t1_injection_advances_to_colon(self) -> None:
        state, schema, vocab, _trie = at_params("fn_add_numbers")
        model = TailFakeModel(vocab, [])
        ids: list[int] = []
        emitted: list[str] = []
        assert _commit_static_text(
            T1_CANON_ADD, model, vocab, ids, state, schema, emitted
        )
        assert ids == _ORACLE_IDS[T1_CANON_ADD]
        _ok, sim = state.simulate(T1_CANON_ADD)  # source of truth
        assert self._state_tuple(state) == self._state_tuple(sim)
        # COLON d1 waiting for the numeric value of "a" — the canonical ended
        # in ' ' (the value opening of a number carries no quote).
        assert state.phase is DecoderPhase.COLON and state.depth == 1
        assert "".join(emitted) == T1_CANON_ADD

    def test_t5_injection_closes_both_objects(self) -> None:
        state, schema, vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        model = TailFakeModel(vocab, [])
        ids: list[int] = []
        assert _commit_static_text(T5_CANON, model, vocab, ids, state, schema)
        _ok, sim = state.simulate(T5_CANON)
        assert self._state_tuple(state) == self._state_tuple(sim)
        assert state.phase is DecoderPhase.COMPLETE

    def test_inject_returns_false_without_advance(self) -> None:
        # encode() only understands oracle canonicals: a text outside the
        # registry -> id 999 does not exist -> injection does not advance ->
        # False (the caller falls back to the normal path; without this
        # contract, a blind continue after the failure would re-match the same
        # span -> infinite loop).
        state, schema, vocab, _trie = at_params("fn_add_numbers")
        model = FakeModel(vocab, [])
        ids: list[int] = []
        before = (state.phase, state.depth, state.current_key)
        assert not _commit_static_text(
            _OLD_TAIL, model, vocab, ids, state, schema
        )
        assert not ids
        assert (state.phase, state.depth, state.current_key) == before


class TestOracleEndToEnd:
    """Oracle end-to-end: canonicals are injected without forwards.

    The model sequence does NOT contain the canonical chars ('\n
    "parameters": {\n    "a": ' does not exist as text in any vocab entry): if
    they appear in the output, they could ONLY have come from the injection.
    The placeholders (" ") occupy the indices each canonical consumes without a
    forward (FakeModel indexes the step by len(input_ids)). Each test fixes the
    EXACT forward count: a regression (a span that does not match, duplicates
    ws, or falls back to forward) diverts the sequence or changes calls.
    """

    def test_fn_add_numbers_full_oracle(self) -> None:
        """T1 + T3 + T6 consumed; the model pays 9 forwards (5 name + '2.0',
        ',', '3', '}'). The real number flow: '2.0' leaves IN_NUMBER_VALUE
        (open number) -> the model pays the comma -> T3 gives the ws + '"b": '
        -> '3' open -> the model pays the closing '}' of parameters -> T6 gives
        the ROOT '\n}'."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        # 5 name + 9 placeholders (T1: 9 ids) + '2.0' + ',' + 6 placeholders
        # (T3: 6 ids) + '3' + '}' (T6 closes without forward).
        model = TailFakeModel(
            vocab,
            [
                "{", '"name": ', '"', "fn_add_numbers", '"',
                *([" "] * 9),
                "2.0", ",",
                *([" "] * 6),
                "3", "}",
            ],
        )
        generated, ok = generate(model, "What is 2+3?", vocab, FUNCTIONS, trie)
        assert ok
        assert T1_CANON_ADD in generated
        assert T3_CANON_NEXT in generated  # ',\n    "b": ' in the real output
        assert generated.endswith(T6_CANON)  # the closing ROOT '\n}'
        assert model.calls == 9

    def test_fn_empty_gets_root_close(self) -> None:
        """fn_empty: ord=∅ -> T1 None -> the model emits ',
        "parameters": {}' by forward; T6 gives the missing '\n}' (real probe:
        SUCCESS=False because of that close)."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = TailFakeModel(
            vocab,
            [
                "{", '"name": ', '"', "fn_empty", '"',
                ', "parameters": {', "}",
            ],
        )
        generated, ok = generate(model, "hi", vocab, FUNCTIONS, trie)
        assert ok
        # Real Qwen INLINE format for empty parameters ('{"' directly):
        # T6 contributes the '\n}' that closed the ROOT.
        assert generated.endswith('"fn_empty", "parameters": {}\n}')
        assert model.calls == 7  # 5 name + parameters '{'+'}' (T6 free)


# ─────────────────────────────────────────────────────────────────────────
# number == float: the other half of the fix (the one that is NOT SchemaContext)
# ─────────────────────────────────────────────────────────────────────────

#: PRIVATE definition that mixes number and integer in the SAME call: this is
#: where a lax fix shows itself (if the decimal is accepted for either of the
#: two, the grader's `isinstance` blows up).
MIXED_FUNCTIONS = [
    FunctionDef(
        name="fn_calc",
        description="Compute compound interest.",
        parameters={
            "principal": ParameterDef(type="number"),
            "years": ParameterDef(type="integer"),
        },
        returns={"type": "number"},
    ),
]

#: Synthetic id for ".0" (the shared mock vocab does not have it and is NOT
#: touched: adding it to the global VOCAB would change `starting["."]` and
#: `valid_by_phase` for all fine-pass tests).
TAIL_ID = 200


class FloatTailModel:
    """Fake that only knows how to tokenize the tail ".0" (what the fix injects)."""

    def __init__(self, vocab: Vocab) -> None:
        self._vocab = vocab
        # Local vocab patch for THIS test (the mock is per-test).
        vocab.id2decoded[TAIL_ID] = ".0"
        vocab.id2token[TAIL_ID] = ".0"
        vocab.token2id[".0"] = TAIL_ID
        vocab.tokens_starting_with.setdefault(".", set()).add(TAIL_ID)

    def encode(self, text: str) -> _FakeTensor:
        if text == ".0":
            return _FakeTensor([TAIL_ID])
        return _FakeTensor([999])


def _at_number(
    key: str, buffer: str, functions: list[FunctionDef] = MIXED_FUNCTIONS
) -> tuple[DecoderState, SchemaContext]:
    """Real state at IN_NUMBER_VALUE with ``buffer`` already committed in the param."""
    state = DecoderState()
    schema = SchemaContext(functions)
    prefix = (
        "{", '"name": ', '"', functions[0].name, '"',
        ', "parameters": {', f'"{key}":', buffer,
    )
    for t in prefix:
        step(state, schema, t)
    assert state.phase is DecoderPhase.IN_NUMBER_VALUE, state.phase
    return state, schema


class TestFloatTailTrigger:
    """`_float_tail` returns '.0' ONLY when an integer must be turned to float.

    WHY THIS HALF EXISTS: the grader runs `assert isinstance(a, float)` for a
    "number" param — `2` scores 0 points. The correction cannot be left to the
    output boundary (output_validator) because `2` and `2.0` are DIFFERENT
    TOKEN SEQUENCES: the decoder already emitted them, and '2.0' != '2' is a
    different string. The literal's shape must be guaranteed during decoding;
    the WHERE (SchemaContext for `integer`, generator for `number`) is fixed by
    the architecture, not by whim.
    """

    def test_fires_on_comma_closer(self) -> None:
        state, schema = _at_number("principal", "2")
        assert _float_tail(state, schema, ",") == ".0"

    def test_fires_on_brace_closer(self) -> None:
        state, schema = _at_number("principal", "2")
        assert _float_tail(state, schema, "}") == ".0"

    def test_fires_on_whitespace_closer(self) -> None:
        """ws is a valid terminator: '2 }' also closes the value."""
        state, schema = _at_number("principal", "2")
        assert _float_tail(state, schema, " }") == ".0"

    def test_fires_on_negative_integer(self) -> None:
        state, schema = _at_number("principal", "-7")
        assert _float_tail(state, schema, ",") == ".0"

    def test_injection_lands_before_the_closer(self) -> None:
        """End-to-end of the helper: after injecting, the buffer is a float and
        the state REMAINS at IN_NUMBER_VALUE, ready to receive the close."""
        vocab = build_vocab()
        state, schema = _at_number("principal", "2")
        ids: list[int] = []
        emitted: list[str] = []
        model = FloatTailModel(vocab)
        assert _inject_float_tail(state, schema, ",", model, vocab, ids, emitted)
        assert state.number_buffer == "2.0", state.number_buffer
        assert state.phase is DecoderPhase.IN_NUMBER_VALUE
        assert emitted == [".0"], emitted
        # Only now the close is valid (before, "2," did not close a float).
        assert state.update_from_text(",")


class TestFloatTailNoTrigger:
    """Everything that must NOT trigger the injection.

    Each case is its own failure mode: if any triggered, we would be CORRUPTING
    a value (2.5 -> 2.5.0) or touching a parameter that does not correspond.
    """

    def test_no_trigger_on_integer_param(self) -> None:
        """An "integer" is NOT touched: coercing 4 to 4.0 breaks the assert."""
        state, schema = _at_number("years", "4")
        assert _float_tail(state, schema, ",") is None

    def test_no_trigger_when_fraction_already_present(self) -> None:
        """'2.5' is already float: injecting would produce '2.5.0' (invalid JSON)."""
        state, schema = _at_number("principal", "2.5")
        assert _float_tail(state, schema, ",") is None

    def test_no_trigger_on_exponent(self) -> None:
        """'1e3' is a Python float: nothing is missing."""
        state, schema = _at_number("principal", "1e3")
        assert _float_tail(state, schema, ",") is None

    def test_no_trigger_on_digit_continuation(self) -> None:
        """Only fires on CLOSE. A token continuing the number does not close."""
        state, schema = _at_number("principal", "2")
        assert _float_tail(state, schema, "5") is None
        assert _float_tail(state, schema, ".") is None

    def test_no_trigger_after_number_closed(self) -> None:
        """Outside IN_NUMBER_VALUE there is no literal to complete."""
        state, schema = _at_number("principal", "2")
        step(state, schema, ",")
        assert state.phase is not DecoderPhase.IN_NUMBER_VALUE
        assert _float_tail(state, schema, "}") is None

    def test_no_trigger_at_depth_zero_name(self) -> None:
        """The "name" value is depth 0 and its expected type is string.

        Even if the grammar does not know it and accepts `{"name": 2`, the
        expected type is NOT "number" -> nothing to complete.
        """
        state = DecoderState()
        schema = SchemaContext(MIXED_FUNCTIONS)
        for t in ("{", '"name": ', "2"):
            step(state, schema, t)
        assert state.phase is DecoderPhase.IN_NUMBER_VALUE
        assert _float_tail(state, schema, ",") is None
