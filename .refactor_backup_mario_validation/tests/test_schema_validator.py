"""Tests de SchemaContext (Task 3.3).

VOLUNTAD DE ESTOS TESTS (por dentro):
- No testean SchemaContext aislado: caminan la state machine REAL
  (DecoderState.update_from_text) y sincronizan el schema después de cada
  token, como hace el generator en Task 4.1 (orden exacto del plan:
  state.update_from_text(token) → schema.update(state)). Si la state
  machine cambia sus fases, estos tests pinchan y hay que revisar el
  contrato.
- El fixture de datos replica las funciones reales de
  data/input/functions_definition.json (convención del proyecto).
- La resolución de la función seleccionada depende de state.name_buffer
  (⚠ desvío documentado en state.py): el name lo acumula LA STATE MACHINE,
  no el schema. El caso token-mixto (name + estructura en el MISMO token)
  es el que justifica ese diseño.
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
from src.models.function_definition import FunctionDef, ParameterDef

# Replica de data/input/functions_definition.json (subset usado en tests).
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
    """Avanza el estado con un token y sincroniza el schema (contrato Task 4.1)."""
    assert state.update_from_text(token), f"state machine rejected {token!r}"
    schema.update(state)


class TestNameResolution:
    """Resolución de selected_function desde el value de "name"."""

    def test_resolves_function_when_name_closes(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        assert schema.selected_function is None  # el buffer aún está vacío
        step(schema, state, "fn_add_numbers")
        assert schema.selected_function is FN_ADD
        step(schema, state, '"')
        assert schema.selected_function is FN_ADD  # ya resuelto: no cambia

    def test_resolution_survives_mid_token_transition(self) -> None:
        """EL caso que justifica el diseño: el name cierra Y el mismo token
        arranca "parameters". El estado post-token tiene current_key
        "parameters"; sin el name_buffer de la state machine, la resolución
        sería imposible."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, 'fn_add_numbers", "parameters": {')
        assert schema.selected_function is FN_ADD

    def test_partial_name_does_not_resolve(self) -> None:
        """Un prefijo parcial no está en el índice: se espera a completar."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, "fn_gr")
        assert schema.selected_function is None
        step(schema, state, "eet")
        assert schema.selected_function is FN_GREET

    def test_param_named_name_does_not_contaminate_resolution(self) -> None:
        """fn_greet tiene un parámetro "name" (depth 1): no re-resuelve."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_greet", "parameters": {')
        assert schema.selected_function is FN_GREET
        step(schema, state, '"name": "Javier"')
        assert schema.selected_function is FN_GREET  # sigue siendo la misma
        assert schema.required_keys_remaining() == set()  # "name" ya emitido

    def test_unknown_name_leaves_no_selection(self) -> None:
        """Nombre que no existe en el índice (defensivo; el trie lo bloquea)."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "')
        step(schema, state, 'fn_does_not_exist"')
        assert schema.selected_function is None


class TestCurrentExpectedType:
    def test_name_value_is_string(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name":')  # COLON con current_key "name"
        assert schema.current_expected_type() == "string"

    def test_parameters_key_has_no_scalar_type(self) -> None:
        """"parameters" es un objeto: sin tipo escalar que constreñir."""

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
        """El parámetro "name" de fn_greet (depth 1) es string, no el output."""

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
        """Si "parameters" aparece ANTES del name, el tipo no se conoce
        (conservador: el filter igual bloquea por sintaxis y trie)."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"parameters": {"a":')
        assert schema.current_expected_type() is None

    def test_no_expected_type_outside_value_phases(self) -> None:
        """En IN_KEY el tipo del value FUTURO aún no aplica."""

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
        step(schema, state, " }")  # cierra "a" con ws + '}' de params
        assert schema.required_keys_remaining() == {"b"}

    def test_no_selection_means_unknown(self) -> None:
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"x": 1}')
        assert schema.required_keys_remaining() == set()


class TestCanCloseParams:
    def test_cannot_close_with_missing_required(self) -> None:
        """EL test clave: la sintaxis PERMITE '}' pero el schema lo bloquea."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0')
        step(schema, state, " }")  # sintácticamente válido: params cerrado
        assert state.phase.name == "VALUE_END"  # la máquina lo aceptó
        assert state.depth == 0
        assert not schema.can_close_params()  # pero falta "b"

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
        """Sin función conocida no se puede afirmar que puede cerrar."""

        schema = SchemaContext(FUNCTIONS)
        assert schema.required_keys_remaining() == set()
        assert not schema.all_required_present()
        assert not schema.can_close_params()


class TestEndToEnd:
    def test_full_json_walk(self) -> None:
        """Camina el JSON COMPLETO de fn_add_numbers con checkpoints."""

        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0')
        step(schema, state, " }")
        assert schema.selected_function is FN_ADD
        assert schema.required_keys_remaining() == set()
        assert schema.can_close_params()
        step(schema, state, "}")  # '}' final del output object
        assert state.phase.name == "COMPLETE"
