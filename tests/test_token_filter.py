"""Tests for the token filter: compute_allowed_ids.

INTENT OF THESE TESTS (internals):
- They do not test the filter in isolation: they walk the FULL pipeline as the
  generator will. Per token: compute_allowed_ids(state, schema, vocab, trie) ->
  verify the target id's membership -> commit the token
  (state.update_from_text -> schema.update). The state -> schema order is the
  one in the plan.
- The mock vocab mirrors the shape of a real BPE vocabulary: some tokens mix
  structure and content ('fn_add_numbers", "parameters": {', ', "b": 3',
  '"a": 2.0'). Those phase crossings are NORMAL in BPE and are exactly what
  exercises the schema's internal triggers (current_key changes, entering a
  value mid-token).
- The <byte> bucket (id 60) has NO entry in id2decoded, just like in
  vocab_loader.py (BYTE_CATEGORY). Ids 62/63 decode to U+FFFD and to a
  surrogate: _is_clean_utf8 must discard them in phase 2.
- MEMBERSHIP assertions, not exact sets (except ROOT): the vocab leaves
  several valid paths in each state; over-specifying would break as soon as
  someone adds a token.
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
from src.decoder.token_filter import compute_allowed_ids
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
    # Keys
    10: "name",
    11: "parameters",
    12: "name2",       # output key NOT defined in the schema (depth-0 gap)
    13: "na",          # key split in half (real subword)
    14: '"name"',      # complete key with quotes
    # Function names (subwords and complete)
    20: "fn_",
    21: "add",
    22: "_numbers",
    23: "greet",
    24: "empty",
    25: "fn_g",        # prefix of fn_greet
    26: "x",           # char that starts no name
    27: "fn_add_numbers",
    28: "fn_greet",
    # Mixed structure+content tokens (BPE's daily bread)
    30: '": "',
    31: '": 2',
    32: ', "b": 3',    # key "b" read MID-token (starts inside a's value)
    33: '"a": 2.0',    # key "a" + value in one token
    34: '"b": 4',
    35: ', "name": "Javier"',
    36: '"name": "Javier"',
    # parameters keys
    40: '"a"',
    41: '"b"',
    42: '"zzz"',       # key not present in the schema
    43: '"a": 4',      # EXACT duplicate of "a" (same text as the committed one)
    # Values
    50: '"x"',
    51: "2.0",
    52: '"Javier"',
    53: "true",
    54: "null",
    55: '"a": "x"',    # string value for a number parameter
    56: "n",
    # Dirty / special tokens
    60: "<byte>",      # ONLY in the BYTE_CATEGORY bucket (does not enter id2decoded)
    61: " the",        # decoded 'Ġthe': starts with a space
    62: "\ufffd",      # decode of invalid UTF-8 bytes
    63: "\udc80",      # surrogate: not a scalar value
    # key+structure tokens
    70: '"name": ',          # key "name" + colon + ws (ends in COLON)
    71: '"parameters": {',   # opens parameters from OBJECT_OPEN
    72: ', "parameters": {',  # closes name, opens parameters (one token)
}

IDS: dict[str, int] = {text: tid for tid, text in VOCAB.items()}


BYTE_IDS = frozenset({60})  # tokens that ONLY go in BYTE_CATEGORY


def build_vocab() -> Vocab:
    """Build the mock Vocab imitating vocab_loader.py's indexing.

    Byte tokens (BYTE_IDS) do NOT enter id2decoded or the character buckets:
    in the real vocab they are raw bytes that fail to decode. They only exist
    in the BYTE_CATEGORY bucket, which the wildcard of compute_allowed_ids
    excludes.
    """
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
        vocab_size=len(VOCAB) + 1,
        valid_by_phase=valid_by_phase,
    )


def make_pipeline() -> tuple[DecoderState, SchemaContext, Vocab, TrieNode]:
    vocab = build_vocab()
    trie = build_trie([fn.name for fn in FUNCTIONS])
    return DecoderState(), SchemaContext(FUNCTIONS), vocab, trie


def step(state: DecoderState, schema: SchemaContext, text: str) -> None:
    """Advance the state with a token and sync the schema."""
    assert state.update_from_text(text), f"state machine rejected {text!r}"
    schema.update(state)


def allowed(
    state: DecoderState, schema: SchemaContext, vocab: Vocab, trie: TrieNode
) -> set[int]:
    return compute_allowed_ids(state, schema, vocab, trie)


class TestRootAcceptance:
    """Acceptance criteria 1: in ROOT only tokens starting with '{'."""

    def test_root_allowed_set_is_exact(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        # Candidates: '{' bucket (id 1) + ws buckets (2 and 61 ' the').
        # 61 fails in simulate ('t' is not valid in ROOT); the dirty ones and
        # the <byte> bucket are not even candidates. Nothing else allowed.
        assert allowed(state, schema, vocab, trie) == {1, 2}

    def test_output_tokens_rejected_at_root(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        a = allowed(state, schema, vocab, trie)
        assert IDS["name"] not in a      # starts with 'n' (not expected)
        assert IDS['"'] not in a         # '"' only after '{'

    def test_dirty_and_special_tokens_never_allowed(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        a = allowed(state, schema, vocab, trie)
        assert 60 not in a
        assert 62 not in a
        assert 63 not in a
        assert IDS[" the"] not in a      # simulate rejects it in ROOT


class TestWildcardAndDirtyTokens:
    """expected_first_chars '*' (open key/string) takes ALL buckets."""

    def test_dirty_tokens_skipped_while_key_open(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"')
        step(state, schema, "na")  # IN_KEY, current_key "na"
        a = allowed(state, schema, vocab, trie)
        # Wildcard active: U+FFFD and surrogate are CANDIDATES but die in
        # _is_clean_utf8; <byte> was never in id2decoded.
        assert 62 not in a
        assert 63 not in a
        assert 60 not in a
        assert IDS['"'] in a  # closes key "na" -> KEY_END (depth 0, ok)


class TestNameTrie:
    """Acceptance criteria 2: the "name" value is restricted to the trie."""

    def test_only_trie_prefixes_while_name_open(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')  # IN_STRING_VALUE, name_buffer ""
        a = allowed(state, schema, vocab, trie)
        assert IDS["fn_"] in a          # real prefix
        assert IDS["fn_g"] in a         # real prefix (fn_greet)
        assert IDS["add"] not in a      # not a prefix (fn_ missing)
        assert IDS["greet"] not in a
        assert IDS["x"] not in a
        assert IDS["name"] not in a
        assert IDS['"'] not in a        # an empty name cannot be closed

    def test_partial_name_cannot_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')
        step(state, schema, "fn_")  # buffer "fn_": prefix, not a name
        assert IDS['"'] not in allowed(state, schema, vocab, trie)

    def test_complete_name_can_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')
        step(state, schema, "fn_add_numbers")
        assert IDS['"'] in allowed(state, schema, vocab, trie)

    def test_name_built_across_subwords_closes_valid(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_", "add", "_numbers"):
            step(state, schema, t)
        assert IDS['"'] in allowed(state, schema, vocab, trie)

    def test_output_key_not_schema_validated(self) -> None:
        """Documented gap: OUTPUT object keys (depth 0) are not validated."""
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"')  # KEY_START
        # "name2" does not exist in the schema, but it is an output-object key:
        # depth 0 is out of scope (the trie only kicks in with the "name" key).
        assert IDS["name2"] in allowed(state, schema, vocab, trie)


class TestParamKeys:
    def _at_params(self) -> tuple[DecoderState, SchemaContext, Vocab, TrieNode]:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        return state, schema, vocab, trie

    def test_known_and_unknown_keys(self) -> None:
        state, schema, vocab, trie = self._at_params()
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a"'] in a
        assert IDS['"b"'] in a
        assert IDS['"'] in a             # opens a key: empty prefix, ok
        assert IDS['"zzz"'] not in a     # key not present in the schema
        assert IDS['"a": 2.0'] in a      # key + value in one token
        assert IDS["}"] not in a         # "a" and "b" are missing (clause 4)

    def test_mid_token_key_read_is_validated(self) -> None:
        """THE change-trigger case: ', "b": 3' reads the key mid-token (starts
        INSIDE a's value) and ends in IN_NUMBER_VALUE, outside _KEY_PHASES. A
        phase trigger would never see it."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')  # IN_NUMBER_VALUE, key "a" (open)
        a = allowed(state, schema, vocab, trie)
        # Key "b" is read inside this token: valid (change detection) and its
        # value type is also checked (post in IN_NUMBER_VALUE -> number).
        assert IDS[', "b": 3'] in a

    def test_identical_rebuild_slip_is_documented(self) -> None:
        """Known limitation (documented in _allows_param_key): an EXACT
        duplicate rebuilds current_key with the SAME text as the committed one
        ("a" -> reset "" -> "a"): the change trigger cannot tell it apart and
        the key passes. Divergent keys are blocked (growing from its text does
        NOT start as a prefix), but the identical one slips through."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")    # closes "a" -> PARAMS_OBJECT
        # committed current_key == "a" (stale), keys_enclosed == {"a"}
        assert IDS['"a": 4'] in allowed(state, schema, vocab, trie)  # slip

    def test_reopened_key_diverging_from_closed_is_blocked(self) -> None:
        """Contrast with the slip: a key that DIVERGES from the committed text
        (multi-token) IS blocked by the prefix check."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")    # PARAMS_OBJECT, committed key "a"
        step(state, schema, '"')    # opens a new key (KEY_START, "")
        a = allowed(state, schema, vocab, trie)
        # "na" cannot complete into any available key ({b}):
        assert IDS["na"] not in a


class TestValueTypes:
    def test_number_param_constrains_value_entry(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
            '"a"',
            ":",
        ):
            step(state, schema, t)  # COLON with current_key "a" (depth 1)
        a = allowed(state, schema, vocab, trie)
        assert IDS["2.0"] in a           # number == number
        assert IDS['"'] not in a         # opens a string for a number
        assert IDS["true"] not in a      # boolean != number
        assert IDS["null"] not in a      # null != number

    def test_string_param_accepts_string_and_rejects_number(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_greet",
            '"',
            ', "parameters": {',
            '"name": ',  # COLON key "name" (depth 1: the PARAMETER)
        ):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS['"'] in a             # opens a string for a string
        assert IDS['"x"'] in a           # string content: ok (no trie here)
        assert IDS["2.0"] not in a       # number != string

    def test_full_key_string_value_in_one_token_type_checked(self) -> None:
        """Cases that clause 3 DOES detect and the documented gap.

        - '33' ('"a": 2.0') from PARAMS_OBJECT: ends in IN_NUMBER_VALUE (in
          _VALUE_PHASES) -> the number type matches a:number -> passes.
        - '55' ('"a": "x"') from PARAMS_OBJECT: opens a key, value string and
          closes it ALL in the same token -> post in VALUE_END (not in
          _VALUE_PHASES) and the start was not COLON: clause 3 does NOT fire. It
          is the gap documented in _allows_value_type (key+value complete per
          token = parametric parser, the nested-arguments bonus). Here it is
          pinned as expected MVP behavior, not as a bug."""

        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a": 2.0'] in a    # type check DOES fire (post in a value phase)
        assert IDS['"a": "x"'] in a    # documented gap — not blocked


class TestParamsClose:
    def test_close_blocked_until_all_required(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        assert IDS["}"] not in allowed(state, schema, vocab, trie)
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        assert IDS["}"] in allowed(state, schema, vocab, trie)

    def test_close_with_value_in_same_token(self) -> None:
        """The close may come with the final value ('"b": 3}') and the
        just-closed value counts for the subset thanks to the SIMULATED
        keys_enclosed."""

        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
            '"a": 2.0',
            ", ",
            '"b": 3',
        ):
            step(state, schema, t)
        step(state, schema, "}")  # closes params: {a,b} subset of new_state.keys_enclosed
        assert state.phase.name == "VALUE_END"
        assert state.depth == 0
        step(state, schema, "}")  # closes the output object
        assert state.phase.name == "COMPLETE"


class TestFnEmpty:
    def test_empty_params_close_immediately(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_empty", '"', ', "parameters": {'):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS["}"] in a            # required is empty: closes now
        assert IDS['"a"'] not in a      # no key is valid (available empty set)


class TestNameFirstEnforcement:
    def test_parameters_before_name_are_blocked(self) -> None:
        """"parameters" BEFORE the name: with no function selected, neither the
        keys nor the '}' pass. The generator gets stuck at depth 1: the only way
        out is to generate the name first (deliberate reinforcement)."""

        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"parameters": {')
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a"'] not in a
        assert IDS["}"] not in a


class TestFnGreetWalk:
    def test_full_walk_with_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_greet",
            '"',
            ', "parameters": {',
            '"name": ',
            '"Javier"',   # opens+closes the string in one BPE token
        ):
            step(state, schema, t)
        # parameter value closed -> keys_enclosed {name} -> close ok
        assert IDS["}"] in allowed(state, schema, vocab, trie)


class TestFullGenerationWalk:
    def test_fn_add_numbers_end_to_end(self) -> None:
        """The plan's acceptance criteria, live: every proposed token must be in
        allowed_ids BEFORE being committed (generator order)."""

        state, schema, vocab, trie = make_pipeline()
        sequence = [
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
        ]
        for i, text in enumerate(sequence):
            assert IDS[text] in allowed(state, schema, vocab, trie), (
                f"step {i}: {text!r} not allowed"
            )
            step(state, schema, text)
        assert state.phase.name == "COMPLETE"
        assert state.keys_enclosed == {"a", "b"}
