"""Tests del boundary de salida: parseo, placeholders y detección de prompts
sin respaldo textual.

POR QUÉ ESTE ARCHIVO EXISTE: `build_results` garantiza la alineación posicional
que la moulinette necesita con su `zip()`. Si esa garantía se rompe, el score
se arruina entero (no se pierde un test: se pierden once). Es la función más
peligrosa del repo y estaba sin tests.

También cubre `find_unsupported_prompts`, el sensor que reporta los prompts
para los que el decoder tuvo que inventarse los argumentos.
"""

from __future__ import annotations

import json

from src.models.output import FunctionCall
from src.validator.output_validator import (
    _UNKNOWN_FN_SENTINEL,
    build_function_call,
    build_results,
    find_unsupported_prompts,
    parse_output,
    validate_output,
)


# --------------------------------------------------------------------------
# parse_output
# --------------------------------------------------------------------------
def test_parse_output_tolerates_surrounding_whitespace() -> None:
    """El decoder envuelve el JSON con '\\n\\n' — ver `generate`."""
    assert parse_output('\n\n{\n  "name": "fn_greet"\n}\n') == {"name": "fn_greet"}


def test_parse_output_raises_on_garbage() -> None:
    try:
        parse_output("not json at all")
    except json.JSONDecodeError:
        return
    raise AssertionError("debía lanzar JSONDecodeError")


# --------------------------------------------------------------------------
# build_function_call
# --------------------------------------------------------------------------
def test_build_function_call_defaults_empty_parameters() -> None:
    """Una función sin parámetros emite sólo `name` (subject V.4)."""
    call = build_function_call("Give me the time", {"name": "fn_get_time"})
    assert call.name == "fn_get_time"
    assert call.parameters == {}


def test_build_function_call_keeps_prompt_verbatim() -> None:
    """La moulinette compara el prompt con igualdad EXACTA de string."""
    prompt = "Reverse the string 'hello'"
    call = build_function_call(prompt, {"name": "fn_reverse_string", "parameters": {"s": "hello"}})
    assert call.prompt == prompt


# --------------------------------------------------------------------------
# validate_output
# --------------------------------------------------------------------------
def test_validate_output_reports_invalid_json_as_value() -> None:
    result = validate_output("{oops", [])
    assert isinstance(result, str)
    assert "invalid JSON" in result


def test_validate_output_reports_unknown_function() -> None:
    result = validate_output('{"name": "fn_nope"}', [])
    assert isinstance(result, str)
    assert "unknown function" in result


# --------------------------------------------------------------------------
# build_results — LA GARANTÍA DE ALINEACIÓN POSICIONAL
# --------------------------------------------------------------------------
def test_build_results_preserves_order_and_length() -> None:
    prompts = ["a", "b", "c"]
    generated = [
        '{"name": "fn_a"}',
        '{"name": "fn_b"}',
        '{"name": "fn_c"}',
    ]
    results = build_results(prompts, generated)
    assert [r.name for r in results] == ["fn_a", "fn_b", "fn_c"]
    assert [r.prompt for r in results] == prompts


def test_build_results_emits_placeholder_without_dropping_entries() -> None:
    """Un prompt roto NO puede eliminar su entry: `zip()` desalinearía todo."""
    prompts = ["ok1", "roto", "ok2"]
    generated = ['{"name": "fn_ok1"}', "}{ no json", '{"name": "fn_ok2"}']
    results = build_results(prompts, generated)
    assert len(results) == 3
    assert results[1].name == _UNKNOWN_FN_SENTINEL
    assert results[1].prompt == "roto"
    # Las de alrededor sobreviven intactas: se pierde UN test, no tres.
    assert results[0].name == "fn_ok1"
    assert results[2].name == "fn_ok2"


def test_build_results_handles_short_generated_list() -> None:
    """Si `generated` viene más corto, igual emite una entry por prompt."""
    results = build_results(["a", "b"], ['{"name": "fn_a"}'])
    assert len(results) == 2
    assert results[1].name == _UNKNOWN_FN_SENTINEL


# --------------------------------------------------------------------------
# find_unsupported_prompts — el sensor de "el prompt no matchea nada"
# --------------------------------------------------------------------------
def test_flags_call_whose_arguments_are_invented() -> None:
    """El caso real medido: '¿qué tiempo hace en París?' → square_root(100)."""
    prompts = ["What is the weather in Paris tomorrow?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 100.0})]
    assert find_unsupported_prompts(prompts, results) == [0]


def test_accepts_call_whose_value_is_verbatim_in_prompt() -> None:
    prompts = ["What is the square root of 16?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 16.0})]
    assert find_unsupported_prompts(prompts, results) == []


def test_float_matches_prompt_written_without_decimal() -> None:
    """`2.0` en el output vs "2" en el prompt: el humano no escribe 2.0."""
    prompts = ["Give me the sum of 2 and 3"]
    results = [FunctionCall(prompt=prompts[0], name="fn_sum", parameters={"a": 2.0, "b": 3.0})]
    assert find_unsupported_prompts(prompts, results) == []


def test_match_is_case_insensitive() -> None:
    prompts = ["Reverse the string 'Hello'"]
    results = [FunctionCall(prompt=prompts[0], name="fn_reverse_string", parameters={"s": "hello"})]
    assert find_unsupported_prompts(prompts, results) == []


def test_one_supported_value_is_enough() -> None:
    """P9: `replacement="****"` no está en el prompt, pero los otros dos sí.

    Es un fallo de accuracy del modelo, no una falta de match — el corretero
    ya lo mide. Por eso el criterio es 'ninguno', no 'todos'.
    """
    prompt = "Replace all vowels in 'Programming is fun' with a star using regex '([aeiouAEIOU])'"
    results = [
        FunctionCall(
            prompt=prompt,
            name="fn_regex_replace",
            parameters={
                "source_string": "Programming is fun",
                "regex": "([aeiouAEIOU])",
                "replacement": "****",
            },
        )
    ]
    assert find_unsupported_prompts([prompt], results) == []


def test_function_without_parameters_is_not_judged() -> None:
    """Sin argumentos no hay nada que comparar: no se reporta."""
    prompts = ["What time is it?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_time", parameters={})]
    assert find_unsupported_prompts(prompts, results) == []


def test_placeholder_is_not_double_reported() -> None:
    """El sentinel ya tiene su propio warning de 'unparseable'."""
    prompts = ["lo que sea"]
    results = [FunctionCall(prompt=prompts[0], name=_UNKNOWN_FN_SENTINEL, parameters={})]
    assert find_unsupported_prompts(prompts, results) == []


def test_booleans_are_not_judged_as_numbers() -> None:
    """`True` es subclase de `int`: si no se chequea antes, diría '1'."""
    prompts = ["Format the report as a table"]
    results = [
        FunctionCall(
            prompt=prompts[0],
            name="fn_format_report",
            parameters={"as_table": True},
        )
    ]
    assert find_unsupported_prompts(prompts, results) == [0]


def test_empty_prompt_text_is_skipped() -> None:
    prompts = [""]
    results = [FunctionCall(prompt="", name="fn_x", parameters={"a": 1.0})]
    assert find_unsupported_prompts(prompts, results) == []


def test_reports_all_offending_indices_at_once() -> None:
    """El pipeline puede tener varios; el sensor devuelve la lista entera."""
    prompts = ["¿qué tiempo hace en París?", "Greet shrek", "¿y en Madrid?"]
    results = [
        FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 100.0}),
        FunctionCall(prompt=prompts[1], name="fn_greet", parameters={"name": "shrek"}),
        FunctionCall(prompt=prompts[2], name="fn_get_square_root", parameters={"a": 5.0}),
    ]
    assert find_unsupported_prompts(prompts, results) == [0, 2]
