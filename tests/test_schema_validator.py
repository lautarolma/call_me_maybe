"""Tests for SchemaContext.

INTENT OF THESE TESTS (in a nutshell):
- They do not test SchemaContext in isolation: they walk the REAL state
  machine (DecoderState.update_from_text) and sync the schema after each
  token, exactly as the generator does (order fixed by the plan:
  state.update_from_text(token) -> schema.update(state)). If the state
  machine changes its phases, these tests fail and the contract must be
  revisited.
- The data fixture replicates the real functions in
  data/input/functions_definition.json (project convention).
- Resolving the selected function depends on state.name_buffer
  (documented deviation in state.py): the name is accumulated by THE STATE
  MACHINE, not by the schema. The mixed-token case (name + structure in the
  SAME token) is what justifies that design.
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
from src.decoder.trie import TrieNode, build_trie
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

FN_ADD = next(fn for fn in FUNCTIONS if fn.name == "fn_add_numbers")
FN_GREET = next(fn for fn in FUNCTIONS if fn.name == "fn_greet")


def step(schema: SchemaContext, state: DecoderState, token: str) -> None:
    """Advance the state with a token and sync the schema (generator contract)."""
    assert state.update_from_text(token), f"state machine rejected {token!r}"
    schema.update(state)


class TestNameResolution:
    """Resolution of selected_function from the "name" value."""

    def test_resolves_function_when_name_closes(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        assert schema.selected_function is None  # buffer still empty
        step(schema, state, "fn_add_numbers")
        assert schema.selected_function is FN_ADD
        step(schema, state, '"')
        assert schema.selected_function is FN_ADD  # already resolved: no change

    def test_resolution_survives_mid_token_transition(self) -> None:
        """THE case that justifies the design: the name closes AND the same
        token opens "parameters". The post-token state has current_key
        "parameters"; without the state machine's name_buffer, resolution
        would be impossible."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, 'fn_add_numbers", "parameters": {')
        assert schema.selected_function is FN_ADD

    def test_partial_name_does_not_resolve(self) -> None:
        """A partial prefix is not in the index: wait until it is complete."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, "fn_gr")
        assert schema.selected_function is None
        step(schema, state, "eet")
        assert schema.selected_function is FN_GREET

    def test_param_named_name_does_not_contaminate_resolution(self) -> None:
        """fn_greet has a "name" parameter (depth 1): it does not re-resolve."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_greet", "parameters": {')
        assert schema.selected_function is FN_GREET
        step(schema, state, '"name": "Javier"')
        assert schema.selected_function is FN_GREET  # still the same
        assert schema.required_keys_remaining() == set()  # "name" already emitted

    def test_unknown_name_leaves_no_selection(self) -> None:
        """Name that does not exist in the index (defensive; the trie blocks it)."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, 'fn_does_not_exist"')
        assert schema.selected_function is None


class TestCurrentExpectedType:
    def test_name_value_is_string(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name":')  # COLON with current_key "name"
        assert schema.current_expected_type() == "string"

    def test_parameters_key_has_no_scalar_type(self) -> None:
        """"parameters" is an object: no scalar type to constrain."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers"')
        step(schema, state, ', "parameters":')  # COLON, depth 0
        assert schema.current_expected_type() is None

    def test_number_param_expected_type(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        assert schema.current_expected_type() == "number"

    def test_string_param_named_name_type(self) -> None:
        """The "name" parameter of fn_greet (depth 1) is string, not the output."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_greet", "parameters": {"name":')
        assert schema.current_expected_type() == "string"

    def test_unknown_param_has_no_type(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"zzz":')
        assert schema.current_expected_type() is None

    def test_params_before_name_unconstrained(self) -> None:
        """If "parameters" appears BEFORE the name, the type is unknown
        (conservative: the filter still blocks by syntax and trie)."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"parameters": {"a":')
        assert schema.current_expected_type() is None

    def test_no_expected_type_outside_value_phases(self) -> None:
        """In IN_KEY the FUTURE value's type does not apply yet."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a')
        assert schema.current_expected_type() is None  # IN_KEY


class TestRequiredKeys:
    def test_remaining_before_any_param(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {')
        assert schema.required_keys_remaining() == {"a", "b"}

    def test_remaining_after_one_param_closed(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0')
        step(schema, state, " }")  # closes "a" with ws + '}' of params
        assert schema.required_keys_remaining() == {"b"}

    def test_no_selection_means_unknown(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"x": 1}')
        assert schema.required_keys_remaining() == set()


class TestCanCloseParams:
    def test_cannot_close_with_missing_required(self) -> None:
        """THE key test: the syntax PERMITS '}' but the schema blocks it."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0')
        step(schema, state, " }")  # syntactically valid: params closed
        assert state.phase.name == "VALUE_END"  # the machine accepted it
        assert state.depth == 0
        assert not schema.can_close_params()  # but "b" is missing

    def test_can_close_after_all_required(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0')
        step(schema, state, " }")
        assert schema.all_required_present()
        assert schema.can_close_params()

    def test_function_without_params_closes_immediately(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_empty", "parameters": {')
        assert schema.can_close_params()  # zero required keys

    def test_no_selection_is_conservative(self) -> None:
        """With no known function, one cannot assert it can close."""

        schema = SchemaContext(FUNCTIONS)
        assert schema.required_keys_remaining() == set()
        assert not schema.all_required_present()
        assert not schema.can_close_params()


class TestEndToEnd:
    def test_full_json_walk(self) -> None:
        """Walk the COMPLETE fn_add_numbers JSON with checkpoints."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0')
        step(schema, state, " }")
        assert schema.selected_function is FN_ADD
        assert schema.required_keys_remaining() == set()
        assert schema.can_close_params()
        step(schema, state, "}")  # final '}' of the output object
        assert state.phase.name == "COMPLETE"


#: Replica of `fn_is_even` (a PRIVATE definition of the grader):
#: `{"n": {"type": "integer"}}`. This is the case that used to break the decoder.
INTEGER_FUNCTIONS = [
    FunctionDef(
        name="fn_is_even",
        description="Check if a number is even.",
        parameters={"n": ParameterDef(type="integer")},
        returns={"type": "boolean"},
    ),
]


class TestIntegerParamIsNumeric:
    """REGRESSION for integer parameters: an "integer" MUST accept numeric tokens.

    BUG CONTEXT (why this class exists):
    "integer" is not a JSON type: it is the grader's type for a "Python int",
    and it only appears in PRIVATE definitions
    (`fn_is_even.n`, `fn_calculate_compound_interest.years`).

    Clause 3 compared `kind == param.type` with EQUALITY. Every numeric token
    declares kind "number" (_PHASE_KIND / _VALUE_START_KINDS), so with equality
    a declared "integer" rejected EVERY candidate token: the allowed set ended
    up empty and the decoder hung without generating anything.

    These tests use `SchemaContext.allows_token`, NOT the `step()` helper above.
    `step()` only walks the state machine (syntax) and therefore NEVER touched
    the type clause: `allows_token` coverage was ZERO, which is exactly why the
    bug survived to a green suite. The type is decided by clause 3, which is
    only consulted when filtering candidates.
    """

    def _at_param_start(self) -> tuple[SchemaContext, DecoderState, TrieNode]:
        """Real state right after the ':' that opens the "n" value."""
        schema = SchemaContext(INTEGER_FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_is_even", "parameters": {"n":')
        trie = build_trie([f.name for f in INTEGER_FUNCTIONS])
        return schema, state, trie

    def test_integer_param_accepts_digit(self) -> None:
        """The token '1' has kind "number" and MUST be valid for "integer"."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("1")
        assert ok, "1 is valid syntax"
        assert schema.allows_token("1", new_state, trie), (
            "a declared 'integer' must accept a digit: with kind==type "
            "no numeric token passed and the decoder had no candidates"
        )

    def test_integer_param_accepts_negative(self) -> None:
        """'-' is also kind "number": negatives must not hang."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("-")
        assert ok
        assert schema.allows_token("-", new_state, trie)

    def test_integer_param_rejects_decimal_point(self) -> None:
        """Clause 5: an "integer" does NOT allow '.', even if JSON permits it.

        This test used to invert its own premise ("JSON does not distinguish
        int from float"). For the SPEC that is true, but the evaluator is not a
        JSON validator: it runs `fn_is_even(n=2.5)` and blows up with
        `assert isinstance(n, int)`. Syntax is validated by state.py; that the
        LITERAL has int shape is a semantic layer rule (SchemaContext), which
        is why it lives here and not in the grammar.
        """
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("2.5")
        assert ok, "the syntax of '2.5' is valid JSON: the reject comes from the schema"
        assert not schema.allows_token("2.5", new_state, trie), (
            "an 'integer' cannot materialize as float: the grader runs "
            "assert isinstance(n, int) over the parsed value"
        )

    def test_integer_param_rejects_exponent(self) -> None:
        """Same with exponent notation: '1e3' parses to float in Python."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("1e3")
        assert ok
        assert not schema.allows_token("1e3", new_state, trie), (
            "1e3 is a Python float even when written without a dot"
        )

    def test_number_param_accepts_decimal_point(self) -> None:
        """Counterpart: a "number" MUST accept the float form (2.5)."""
        schema = SchemaContext(FUNCTIONS)  # fn_add_numbers.a: number
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("2.5")
        assert ok
        assert schema.allows_token("2.5", new_state, trie), (
            "clause 5 only restricts 'integer': it must not close the "
            "float path of a 'number'"
        )

    def test_integer_param_rejects_string(self) -> None:
        """Relaxing the numeric axis does NOT open the door to strings."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate('"hello"')
        assert ok, "the string syntax is valid"
        assert not schema.allows_token('"hello"', new_state, trie), (
            "a string cannot satisfy an 'integer' parameter"
        )

    def test_integer_param_rejects_boolean(self) -> None:
        """Same with boolean: kind 'boolean' does not belong to the numeric family."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("true")
        assert ok
        assert not schema.allows_token("true", new_state, trie)

    def test_integer_param_rejects_null(self) -> None:
        """Same with null."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("null")
        assert ok
        assert not schema.allows_token("null", new_state, trie)

    def test_integer_param_walks_to_complete(self) -> None:
        """Walk the whole JSON: the fix does not only permit the token, it terminates."""
        schema = SchemaContext(INTEGER_FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_is_even", "parameters": {"n": 4')
        step(schema, state, "}")
        step(schema, state, "}")
        assert schema.all_required_present()
        assert state.phase.name == "COMPLETE"

    def test_number_still_rejects_boolean(self) -> None:
        """Guard against regression: relaxing 'integer' did not relax 'number'."""
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("false")
        assert ok
        assert not schema.allows_token("false", new_state, trie)

    def test_number_param_accepts_digit(self) -> None:
        """Guard against regression: 'number' still accepts digits."""
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("1")
        assert ok
        assert schema.allows_token("1", new_state, trie)
