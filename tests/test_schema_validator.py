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
from src.decoder.trie import TrieNode, build_trie
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


#: Réplica de `fn_is_even` (definición PRIVADA de la moulinette):
#: `{"n": {"type": "integer"}}`. Es el caso que rompía el decoder.
INTEGER_FUNCTIONS = [
    FunctionDef(
        name="fn_is_even",
        description="Check if a number is even.",
        parameters={"n": ParameterDef(type="integer")},
        returns={"type": "boolean"},
    ),
]


class TestIntegerParamIsNumeric:
    """REGRESIÓN del P0: un parámetro "integer" DEBE aceptar tokens numéricos.

    CONTEXTO DEL BUG (por qué esta clase existe):
    "integer" no es un tipo de JSON: es el de la moulinette para "int de
    Python", y sólo aparece en las definiciones PRIVADAS
    (`fn_is_even.n`, `fn_calculate_compound_interest.years`).

    La cláusula 3 comparaba `kind == param.type` con IGUALDAD. Todo token
    numérico declara kind "number" (_PHASE_KIND / _VALUE_START_KINDS), así que
    con igualdad un "integer" declarado rechazaba CADA token candidato: el
    allowed set quedaba vacío y el decoder se colgaba sin generar nada.

    ⚠️ Estos tests usan `SchemaContext.allows_token`, NO el helper `step()` de
    arriba. `step()` sólo camina la state machine (sintaxis) y por eso NUNCA
    tocó la cláusula de tipo: la cobertura de `allows_token` era CERO, que es
    exactamente por lo que el bug sobrevivió a 196 tests verdes. El tipo se
    decide en la cláusula 3, que sólo se consulta al filtrar candidatos.
    """

    def _at_param_start(self) -> tuple[SchemaContext, DecoderState, TrieNode]:
        """Estado real justo después del ':' que abre el valor de "n"."""
        schema = SchemaContext(INTEGER_FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_is_even", "parameters": {"n":')
        trie = build_trie([f.name for f in INTEGER_FUNCTIONS])
        return schema, state, trie

    def test_integer_param_accepts_digit(self) -> None:
        """El token '1' tiene kind "number" y DEBE ser válido para "integer"."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("1")
        assert ok, "1 es sintaxis válida"
        assert schema.allows_token("1", new_state, trie), (
            "un 'integer' declarado debe aceptar un dígito: con kind==type "
            "ningún token numérico pasaba y el decoder no tenía candidatos"
        )

    def test_integer_param_accepts_negative(self) -> None:
        """'-' también es kind "number": los negativos no deben quedar colgados."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("-")
        assert ok
        assert schema.allows_token("-", new_state, trie)

    def test_integer_param_rejects_decimal_point(self) -> None:
        """Cláusula 5: un "integer" NO admite '.', aunque JSON lo permita.

        ⚠ Este test invertía su propia premisa ("JSON no distingue int de
        float"). Para el SPEC es cierto, pero el evaluador no es un validador
        de JSON: ejecuta `fn_is_even(n=2.5)` y revienta con
        `assert isinstance(n, int)`. La syntacticidad la valida state.py; que
        el LITERAL tenga forma de int es una regla de capa semántica
        (SchemaContext), y por eso vive acá y no en el gramático.
        """
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("2.5")
        assert ok, "la sintaxis de '2.5' es JSON válido: el reject es del schema"
        assert not schema.allows_token("2.5", new_state, trie), (
            "un 'integer' no puede materializarse como float: la moulinette "
            "corre assert isinstance(n, int) sobre el valor parseado"
        )

    def test_integer_param_rejects_exponent(self) -> None:
        """idem con notación exponencial: '1e3' parsea a float en Python."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("1e3")
        assert ok
        assert not schema.allows_token("1e3", new_state, trie), (
            "1e3 es un float de Python aunque se escriba sin punto"
        )

    def test_number_param_accepts_decimal_point(self) -> None:
        """Contracara: un "number" SÍ debe admitir la forma float (2.5)."""
        schema = SchemaContext(FUNCTIONS)  # fn_add_numbers.a: number
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("2.5")
        assert ok
        assert schema.allows_token("2.5", new_state, trie), (
            "la cláusula 5 sólo restringe a 'integer': no debe cerrar el "
            "camino float de un 'number'"
        )

    def test_integer_param_rejects_string(self) -> None:
        """Relajar el eje numérico NO abre la puerta a strings."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate('"hola"')
        assert ok, "la sintaxis del string es válida"
        assert not schema.allows_token('"hola"', new_state, trie), (
            "un string no puede satisfacer un parámetro 'integer'"
        )

    def test_integer_param_rejects_boolean(self) -> None:
        """idem con boolean: kind 'boolean' no pertenece a la familia numérica."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("true")
        assert ok
        assert not schema.allows_token("true", new_state, trie)

    def test_integer_param_rejects_null(self) -> None:
        """idem con null."""
        schema, state, trie = self._at_param_start()
        ok, new_state = state.simulate("null")
        assert ok
        assert not schema.allows_token("null", new_state, trie)

    def test_integer_param_walks_to_complete(self) -> None:
        """Camina el JSON completo: el fix no sólo permite el token, termina."""
        schema = SchemaContext(INTEGER_FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_is_even", "parameters": {"n": 4')
        step(schema, state, "}")
        step(schema, state, "}")
        assert schema.all_required_present()
        assert state.phase.name == "COMPLETE"

    def test_number_still_rejects_boolean(self) -> None:
        """Guarda contra regresión: relajar 'integer' tampoco relajó 'number'."""
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("false")
        assert ok
        assert not schema.allows_token("false", new_state, trie)

    def test_number_param_accepts_digit(self) -> None:
        """Guarda contra regresión: 'number' sigue aceptando dígitos."""
        schema = SchemaContext(FUNCTIONS)
        state = DecoderState()
        step(schema, state, '{"name": "fn_add_numbers", "parameters": {"a":')
        trie = build_trie([f.name for f in FUNCTIONS])
        ok, new_state = state.simulate("1")
        assert ok
        assert schema.allows_token("1", new_state, trie)
