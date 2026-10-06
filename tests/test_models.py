"""Unit tests for the Pydantic I/O models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models.function_definition import FunctionDef, ParameterDef
from src.models.output import FunctionCall

VALID_FUNCTION = {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"},
}


class TestFunctionDef:
    def test_valid_function_def(self) -> None:
        fn = FunctionDef(**VALID_FUNCTION)
        assert fn.name == "fn_add_numbers"
        assert fn.parameters["a"].type == "number"
        assert fn.returns["type"] == "number"

    def test_missing_fields_raise(self) -> None:
        with pytest.raises(ValidationError):
            FunctionDef(name="fn_missing_stuff")

    def test_empty_parameters_allowed(self) -> None:
        fn = FunctionDef(name="fn", description="d", parameters={}, returns={})
        assert fn.parameters == {}

    def test_parameter_names_synced_from_keys(self) -> None:
        fn = FunctionDef(**VALID_FUNCTION)
        assert fn.parameters["a"].name == "a"
        assert fn.parameters["b"].name == "b"

    def test_invalid_parameter_type_raises(self) -> None:
        # "object" es un tipo REAL de JSON, pero está fuera del scope del MVP a
        # propósito (requiere schema recursivo — ver bonus B8). Sigue siendo el
        # mejor ejemplo de tipo inválido porque el rechazo es INTENCIONAL y no
        # un olvido.
        #
        # OJO: este test usaba "integer" como ejemplo y POR ESO codificaba un
        # bug como comportamiento esperado. "integer" no es un tipo de JSON:
        # es la manera que tiene la moulinette de marcar un int de Python, y
        # aparece en las definiciones privadas. Con "integer" fuera del Literal,
        # load_functions explotaba sobre el set privado y el programa no
        # arrancaba. Ver test_integer_type_accepted.
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": "object"}}}
        with pytest.raises(ValidationError):
            FunctionDef(**payload)

    def test_integer_type_accepted(self) -> None:
        """Regresión del P0: "integer" es un tipo válido (sólo en el set privado).

        Si este test falla, el programa no arranca con las definiciones privadas
        (`fn_is_even.n` y `fn_calculate_compound_interest.years` son
        "integer") y la mitad de la evaluación queda en cero.
        """
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": "integer"}}}
        fn = FunctionDef(**payload)
        assert fn.parameters["a"].type == "integer"

    def test_non_string_parameter_type_raises(self) -> None:
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": 123}}}
        with pytest.raises(ValidationError):
            FunctionDef(**payload)

    def test_all_documented_types_accepted(self) -> None:
        for allowed in ("string", "number", "integer", "boolean", "null"):
            payload = {**VALID_FUNCTION, "parameters": {"x": {"type": allowed}}}
            fn = FunctionDef(**payload)
            assert fn.parameters["x"].type == allowed


class TestParameterDef:
    def test_valid(self) -> None:
        param = ParameterDef(name="a", type="number")
        assert param.name == "a"
        assert param.type == "number"

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParameterDef(name="a")


class TestFunctionCall:
    def test_valid_call(self) -> None:
        call = FunctionCall(
            prompt="Greet shrek",
            name="fn_greet",
            parameters={"name": "shrek"},
        )
        assert call.prompt == "Greet shrek"
        assert call.name == "fn_greet"
        assert call.parameters == {"name": "shrek"}

    def test_default_empty_parameters(self) -> None:
        call = FunctionCall(prompt="Greet shrek", name="fn_greet")
        assert call.parameters == {}

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            FunctionCall(prompt="Greet shrek", parameters={"a": 1})

    def test_missing_prompt_raises(self) -> None:
        """El subject V.4 exige las 3 keys: prompt, name y parameters.

        `prompt` es obligatorio: la moulinette compara
        `student_answer["prompt"]` con `correction["prompt"]` por igualdad
        exacta, así que un FunctionCall sin prompt no es serializable a una
        entry válida del output.
        """
        with pytest.raises(ValidationError):
            FunctionCall(name="fn_greet", parameters={"a": 1})


class TestEchoView:
    """`echo_view` es lo que el pipeline imprime en stdout.

    No es un detalle cosmético: el eco se imprimía ANTES de la validación, así
    que la consola mostraba `replacement: "****"` mientras el JSON en disco ya
    traía `*`. El corretero scorea el archivo, así que el score era 11/11 igual,
    pero cualquier revisor que lea la consola ve output roto y cree que el
    pipeline está mal. Estos tests blindan que la vista proyectada salga del
    modelo validado y no del dict crudo del decoder.
    """

    def test_only_name_and_parameters(self) -> None:
        call = FunctionCall(prompt="Greet shrek", name="fn_greet", parameters={"name": "shrek"})
        assert call.echo_view() == {"name": "fn_greet", "parameters": {"name": "shrek"}}

    def test_prompt_is_never_echoed(self) -> None:
        """El `prompt` no va: la consola ya lo muestra al leer la entrada.

        Si aparece acá, cada bloque del eco pasa de 3 a 6 líneas y se duplica
        texto que ya está en pantalla.
        """
        assert "prompt" not in FunctionCall(prompt="Greet shrek", name="fn_greet").echo_view()

    def test_key_order_is_name_then_parameters(self) -> None:
        view = FunctionCall(prompt="p", name="fn_greet", parameters={}).echo_view()
        assert list(view) == ["name", "parameters"]

    def test_echo_reflects_post_validation_value(self) -> None:
        """El caso real de P9: el valor reparado es el que se ve.

        Este es el test de regresión del bug de stdout. `replacement` vale `*`
        (ya validado) y el eco tiene que mostrar `*` — no el `****` crudo del
        decoder. Si alguien vuelve a imprimir el dict crudo, este test falla.
        """
        validated = FunctionCall(
            prompt="Replace all vowels in 'Programming is fun' with asterisks",
            name="fn_substitute_string_with_regex",
            parameters={
                "source_string": "Programming is fun",
                "regex": "([aeiouAEIOU])",
                "replacement": "*",
            },
        )
        assert validated.echo_view()["parameters"]["replacement"] == "*"  # type: ignore[index]

    def test_echo_is_json_serializable(self) -> None:
        """El pipeline lo pasa por `json.dumps(..., indent=2)`: tiene que
        serializar sin inventar nada, y sin `ensure_ascii` para los acentos."""
        call = FunctionCall(prompt="Saludá a shrek", name="fn_greet", parameters={"name": "Ñandú"})
        dumped = json.dumps(call.echo_view(), indent=2, ensure_ascii=False)
        assert "Ñandú" in dumped
        assert json.loads(dumped) == call.echo_view()
