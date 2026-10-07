"""Schema-aware validation for the constrained JSON decoder.

This module validates the semantic constraints on top of the FSM syntax:
which function name is allowed (by the trie), which parameter keys exist
and are not duplicated, whether a value matches the declared parameter
type, whether parameters can be closed with all required keys present,
and the exact form of integer literals.

The schema does not inject syntax. It only decides whether a model-proposed
token is allowed given the current semantic context.
"""

from __future__ import annotations

from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.trie import TrieNode, find_node, is_complete_name
from src.models.function_definition import FunctionDef

# Phases in which a value is being read, or about to start at COLON.
# In these phases, the current key is defined and an expected type may apply.
_VALUE_READ_PHASES = (
    DecoderPhase.COLON,
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.IN_NUMBER_VALUE,
    DecoderPhase.IN_BOOL_VALUE,
    DecoderPhase.IN_NULL_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)

# Phases that represent an active scalar value. Used to detect when a token
# enters a value in the current step (pre-state not in this set, post-state is).
_VALUE_PHASES = (
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.IN_NUMBER_VALUE,
    DecoderPhase.IN_BOOL_VALUE,
    DecoderPhase.IN_NULL_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)

# Phases in which the decoder is reading the value of the output object's
# "name" key (depth 0), or about to start it (COLON already set the key).
_NAME_READ_PHASES = (
    DecoderPhase.COLON,
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)

# JSON kind declared by the current value phase (for the type gate).
_PHASE_KIND: dict[DecoderPhase, str] = {
    DecoderPhase.IN_STRING_VALUE: "string",
    DecoderPhase.ESCAPE_IN_STRING: "string",
    DecoderPhase.IN_NUMBER_VALUE: "number",
    DecoderPhase.IN_BOOL_VALUE: "boolean",
    DecoderPhase.IN_NULL_VALUE: "null",
}

# JSON kind inferred from the first character of a value that opens at COLON
# and is closed within the same token (e.g. '2,', 'true}', '"x",').
_VALUE_START_KINDS: dict[str, str] = {
    '"': "string",
    "-": "number",
    **{ch: "number" for ch in "0123456789"},
    "t": "boolean",
    "f": "boolean",
    "n": "null",
}

# Parameter types treated as numeric. JSON has only "number"; the "integer"
# type is a project-specific constraint (the evaluator expects int values).
_NUMERIC_PARAM_TYPES = frozenset({"number", "integer"})


def _declared_type_accepts(kind: str, declared: str) -> bool:
    """Return True if a token kind is acceptable for a declared parameter type.

    Numeric kinds are treated as a family: both "number" and "integer"
    accept a numeric token kind. The exact literal form for "integer"
    (no '.', 'e', 'E') is enforced separately.

    Args:
        kind: The JSON kind inferred from the token/value phase.
        declared: The parameter type declared in the schema.

    Returns:
        True if the kind matches the declared type under the numeric family rule.
    """
    if kind == "number":
        return declared in _NUMERIC_PARAM_TYPES
    return kind == declared


# Terminators that close a numeric literal.
_NUMBER_TERMINATORS = frozenset(",} \t\n\r")


class SchemaContext:
    """Semantic context for constrained decoding.

    Tracks the selected function and parameter context (current key, depth,
    and which parameter keys have been enclosed). This object is updated
    once per generation step (not copied per candidate), so it uses
    explicit slots rather than a dataclass.
    """

    __slots__ = (
        "_index",
        "_current_key",
        "_depth",
        "_keys_enclosed",
        "_phase",
        "_params_object_seen",
        "selected_function",
    )

    def __init__(self, functions: list[FunctionDef]) -> None:
        """Initialize the schema context with function definitions.

        Args:
            functions: List of available functions; names must be unique.
        """
        self._index = {fn.name: fn for fn in functions}
        self._current_key = ""
        self._depth = 0
        self._keys_enclosed: set[str] = set()
        self._phase = DecoderPhase.ROOT
        # True if the path has passed through the "parameters" object.
        self._params_object_seen = False
        # Selected function resolved from the "name" value.
        self.selected_function: FunctionDef | None = None

    # ------------------------------------------------------------------ API

    def update(self, state: DecoderState) -> None:
        """Refresh the context from the committed decoder state.

        Called once per generation step (not per candidate). The filter
        reads the schema; mutations occur only here.

        Args:
            state: The decoder state to read from.
        """
        self._phase = state.phase
        self._current_key = state.current_key
        self._keys_enclosed = set(state.keys_enclosed)
        self._depth = state.depth
        if state.phase is DecoderPhase.PARAMS_OBJECT:
            self._params_object_seen = True
        self._resolve_function(state)

    def current_expected_type(self) -> str | None:
        """Return the expected JSON type for the current value, or None if none.

        - Output object key "name" (depth 0): "string".
        - Parameter of the selected function (depth 1): the declared parameter type.
        - Otherwise (parameters object, unknown key, no function selected,
          or not reading a value): None (no constraint).

        Returns:
            The expected type name, or None.
        """
        if self._phase not in _VALUE_READ_PHASES:
            return None
        if self._depth == 0:
            return "string" if self._current_key == "name" else None
        if self.selected_function is None:
            return None
        param = self.selected_function.parameters.get(self._current_key)
        return param.type if param is not None else None

    def required_keys_remaining(self) -> set[str]:
        """Return required parameter keys not yet enclosed.

        In this MVP all parameters are required. This set also defines
        valid candidate keys (those that exist and have not been duplicated).

        Returns:
            Set of remaining required parameter names.
        """
        if self.selected_function is None:
            return set()
        return set(self.selected_function.parameters) - self._keys_enclosed

    def all_required_present(self) -> bool:
        """Return True if all required keys have been enclosed.

        Returns False if no function is selected (conservative).
        """
        return self.selected_function is not None and not self.required_keys_remaining()

    def can_close_params(self) -> bool:
        """Return True if the closing '}' of the parameters object is allowed."""
        return self.all_required_present()

    def has_seen_params_object(self) -> bool:
        """Return True if the decoder has seen the '{' of the "parameters" object.

        Used by the fine pass: when the state reaches COMPLETE, the generator
        may require this flag to be set. It is set in update() when the phase
        becomes PARAMS_OBJECT.
        """
        return self._params_object_seen

    # ------------------------------------------------ Fase 3 del filter (3.4)

    def allows_token(
        self, token_text: str, new_state: DecoderState, trie: TrieNode
    ) -> bool:
        """Return True if the schema allows applying/adopting the candidate token.

        This is a pure predicate: it does not mutate the schema context.
        The filter calls it once per candidate using the last committed
        snapshot (self) and the simulated post-state (new_state).

        The check is the AND of five clauses:
        1. _allows_name_value: the "name" value must be a valid function name.
        2. _allows_param_key: parameter keys must exist and not be duplicated.
        3. _allows_value_type: a parameter value must match the declared type.
        4. _allows_params_close: closing '}' of parameters requires all required keys.
        5. _allows_integer_form: integer parameters forbid '.', 'e', 'E'.

        Args:
            token_text: Text of the candidate token.
            new_state: Simulated decoder state after applying the token.
            trie: Trie of allowed function names.

        Returns:
            True if the token is allowed by schema constraints.
        """
        if not self._allows_name_value(new_state, trie):
            return False
        if not self._allows_param_key(new_state):
            return False
        if not self._allows_value_type(token_text, new_state):
            return False
        if not self._allows_params_close(new_state):
            return False
        if not self._allows_integer_form(token_text):
            return False
        return True

    # ------------------------------------------------------ name resolution

    def _resolve_function(self, state: DecoderState) -> None:
        """Resolve selected_function when the output object's "name" value is known.

        Resolution is based on the name buffer (not phase): any non-empty
        name_buffer from the FSM is the "name" value at depth 0. The buffer
        persists even if the same token moves to subsequent structure.

        Args:
            state: Decoder state containing name_buffer.
        """
        if self.selected_function is not None:
            return
        if not state.name_buffer:
            return
        self.selected_function = self._index.get(state.name_buffer)

    # ------------------------------------------------ cláusulas de Fase 3

    def _allows_name_value(
        self, new_state: DecoderState, trie: TrieNode
    ) -> bool:
        """Clause 1: the "name" value must be a valid function name in the trie.

        Uses the FSM name_buffer (decoded, escapes skipped). Two branches:
        - Post-state still inside the "name" string (depth 0, key "name"): the
          accumulated buffer (including this token) must be a prefix of some name.
        - The token closed or left the "name" string (transitioned out): the
          final buffer must be a complete name.

        Args:
            new_state: Simulated state after the candidate token.
            trie: Trie of allowed function names.

        Returns:
            True if the "name" value is allowed by the trie.
        """
        if (
            new_state.phase in _NAME_READ_PHASES
            and new_state.current_key == "name"
            and new_state.depth == 0
        ):
            return find_node(trie, new_state.name_buffer) is not None
        if (
            self._phase in _NAME_READ_PHASES
            and self._current_key == "name"
            and self._depth == 0
        ):
            return is_complete_name(trie, new_state.name_buffer)
        return True

    def _allows_param_key(self, new_state: DecoderState) -> bool:
        """Clause 2: parameter keys (depth 1) must exist and not be duplicated.

        Triggered when current_key changes in this token (covers cases where
        a key is read mid-token across a value boundary). For a key that is
        still being built, any available key that starts with it is allowed;
        once closed, exact membership is required.

        The set of available keys is computed against the committed
        keys_enclosed (self._keys_enclosed), not the simulated one, so that
        a key closed within the same token is not treated as a duplicate.

        Args:
            new_state: Simulated state after the candidate token.

        Returns:
            True if the parameter key transition is allowed.
        """
        if self._depth != 1:
            return True
        if new_state.current_key == self._current_key:
            return True
        if self.selected_function is None:
            return False
        available = set(self.selected_function.parameters) - self._keys_enclosed
        key = new_state.current_key
        if new_state.phase in (DecoderPhase.KEY_START, DecoderPhase.IN_KEY):
            return any(k.startswith(key) for k in available)
        return key in available

    def _allows_value_type(
        self, token_text: str, new_state: DecoderState
    ) -> bool:
        """Clause 3: a parameter value must match the declared type.

        The kind is inferred in two ways:
        - If the post-state is in a value phase, the phase declares the kind.
        - If the committed phase was COLON and the value opens and closes
          within the same token, the first character of the token text
          determines the kind.

        This applies only when reading a parameter value (depth 1) for the
        selected function. Unknown keys default to allow; non-scalar values
        (e.g. objects) produce no kind and are allowed.

        Args:
            token_text: Text of the candidate token.
            new_state: Simulated state after the candidate token.

        Returns:
            True if the value type is allowed.
        """
        kind: str | None = None
        if new_state.phase in _VALUE_PHASES:
            kind = _PHASE_KIND[new_state.phase]
        elif self._phase is DecoderPhase.COLON:
            kind = _VALUE_START_KINDS.get(token_text.lstrip()[:1])
        if kind is None:
            return True
        if new_state.depth != 1 or self.selected_function is None:
            return True
        param = self.selected_function.parameters.get(new_state.current_key)
        if param is None:
            return True
        return _declared_type_accepts(kind, param.type)

    def _allows_integer_form(self, token_text: str) -> bool:
        """Cláusula 5: el literal de un parámetro "integer" no lleva '.', 'e' ni 'E'.

        La moulinette ejecuta `fn(**params)` con `assert isinstance(n, int)`:
        `4.0` o `1e3` (floats en Python) dan 0 puntos aunque sean JSON válido.

        Trigger por estado COMMITEADO: el token arranca en COLON (puede abrir
        el value) o dentro de IN_NUMBER_VALUE (lo continúa), a depth 1 y con
        el parámetro declarado "integer". Solo se inspecta el tramo del token
        previo al primer terminador: lo que viene después ya no es del literal.
        """
        if self._phase not in (DecoderPhase.COLON, DecoderPhase.IN_NUMBER_VALUE):
            return True
        if self._depth != 1 or self.selected_function is None:
            return True
        param = self.selected_function.parameters.get(self._current_key)
        if param is None or param.type != "integer":
            return True
        text = token_text
        if self._phase is DecoderPhase.COLON:
            text = token_text.lstrip()
            if not text or text[0] not in "-0123456789":
                return True  # el token no abre un number
        for char in text:
            if char in _NUMBER_TERMINATORS:
                break
            if char in ".eE":
                return False
        return True

    def _allows_params_close(self, new_state: DecoderState) -> bool:
        """Clause 4: closing '}' of parameters requires all required keys present.

        Triggered when the committed depth is 1 and the simulated depth becomes 0
        (the token closed the parameters object). The simulated keys_enclosed
        is used so that a key/value pair completed within the same token
        counts toward the requirement.

        Args:
            new_state: Simulated state after the candidate token.

        Returns:
            True if closing parameters is allowed.
        """
        if self._depth != 1 or new_state.depth != 0:
            return True
        if self.selected_function is None:
            return False
        return set(self.selected_function.parameters) <= new_state.keys_enclosed
