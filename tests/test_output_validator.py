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
    _collapse_repeated_run,
    _repair_string_value,
    _restore_internal_quotes,
    _snap_to_query_span,
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
# _snap_to_query_span — re-anclaje del value al tramo verbatim de la query
# --------------------------------------------------------------------------
def test_snap_restores_clipped_leading_punctuation() -> None:
    """Test privado 8: el modelo copia el path pero pierde el '/' inicial."""
    prompt = "Read the file at /home/user/data.json with utf-8 encoding"
    payload = {"name": "fn_read_file", "parameters": {"path": "home/user/data.json", "encoding": "utf-8"}}
    call = build_function_call(prompt, payload)
    assert call.parameters["path"] == "/home/user/data.json"
    assert call.parameters["encoding"] == "utf-8"  # ya arranca en borde: intacto


def test_snap_leaves_windows_drive_path_untouched() -> None:
    """Test privado 9: arranca con 'C', no con '/'. La regla NO fuerza '/'."""
    prompt = "Read C:\\Users\\john\\config.ini with latin-1 encoding"
    payload = {"name": "fn_read_file", "parameters": {"path": "C:\\Users\\john\\config.ini"}}
    call = build_function_call(prompt, payload)
    assert call.parameters["path"] == "C:\\Users\\john\\config.ini"


def test_snap_stops_at_quote_delimiter() -> None:
    """La comilla DELIMITA el valor: 'hello' no se estira a "'hello"."""
    call = build_function_call(
        "Reverse the string 'hello'",
        {"name": "fn_reverse_string", "parameters": {"s": "hello"}},
    )
    assert call.parameters["s"] == "hello"


def test_snap_stops_at_alphanumeric_suffix() -> None:
    """'llo' dentro de 'hello' es un sufijo, no un valor clipeado."""
    call = build_function_call(
        "Give me the last 3 letters of 'hello'",
        {"name": "fn_substring", "parameters": {"s": "llo"}},
    )
    assert call.parameters["s"] == "llo"


def test_snap_leaves_value_absent_from_prompt() -> None:
    """Sin ocurrencia verbatim la regla A no tiene a qué re-anclarse.

    Se prueba la regla A DIRECTAMENTE y no por `build_function_call` a propósito:
    el pipeline completo sí corrige este valor, pero por la regla C (§8.9). La
    cobertura que importa acá es "A sola no hace nada", y `build_function_call`
    ya no puede observarla.
    """
    prompt = "Replace all vowels with asterisks"
    assert _snap_to_query_span("****", prompt) == "****"


def test_snap_ignores_non_string_values() -> None:
    """Números/bools/null no pasan por el re-anclaje."""
    call = build_function_call(
        "What is the product of 3 and 5?",
        {"name": "fn_multiply_numbers", "parameters": {"a": 3.0, "b": 5.0}},
    )
    assert call.parameters == {"a": 3.0, "b": 5.0}


def test_snap_is_noop_with_empty_prompt() -> None:
    """`validate_output` construye con prompt="" → no-op por diseño."""
    call = build_function_call("", {"name": "fn_x", "parameters": {"s": "home/user"}})
    assert call.parameters["s"] == "home/user"


# --------------------------------------------------------------------------
# _restore_internal_quotes — restitución de comillas dobles INTERNAS
# --------------------------------------------------------------------------
def test_quotes_restores_internal_quotes_dropped_by_model() -> None:
    """Test privado 11: copia bien el contenido pero se come las comillas."""
    prompt = 'Format template: Say "hello" to {name}'
    assert _restore_internal_quotes("Say hello to {name}", prompt) == 'Say "hello" to {name}'


def test_quotes_restores_internal_quotes_mirror_case() -> None:
    """No es un caso único: 'hi' en vez de 'hello' se comporta igual."""
    prompt = 'Format template: Say "hi" to {user}'
    assert _restore_internal_quotes("Say hi to {user}", prompt) == 'Say "hi" to {user}'


def test_quotes_ignores_delimiting_quotes_around_whole_value() -> None:
    """El contraejemplo que define la regla: las comillas del MARCO son
    delimitadores, no contenido. Este valor NO se toca (test público)."""
    prompt = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    value = "Hello 34 I'm 233 years old"
    assert _restore_internal_quotes(value, prompt) == value


def test_quotes_skips_value_already_verbatim() -> None:
    """Si el valor ya aparece literal, no hay nada que restaurar."""
    assert _restore_internal_quotes("hello", 'Say "hello" and hello') == "hello"


def test_quotes_stays_silent_when_ambiguous() -> None:
    """Dos ocurrencias candidatas → la regla se calla en vez de adivinar."""
    assert _restore_internal_quotes("abc", 'Echo "abc" and abc to stdout') == "abc"


def test_quotes_stays_silent_when_only_delimited_spans_exist() -> None:
    """Sin comillas internas en juego no hay corrección posible."""
    assert _restore_internal_quotes("x", 'Format: "x" and "x"') == "x"


def test_quotes_is_noop_without_double_quotes_in_prompt() -> None:
    """Sin comillas dobles la query mutada es idéntica a la original."""
    assert _restore_internal_quotes("Say hello", "Say hello") == "Say hello"


def test_quotes_is_noop_with_empty_value() -> None:
    """Un valor vacío no se busca ni se devuelve nada."""
    assert _restore_internal_quotes("", 'Say "hello"') == ""


# --------------------------------------------------------------------------
# _collapse_repeated_run — deshacer el conteo de repeticiones
# --------------------------------------------------------------------------
def test_collapse_undoes_count_when_query_shows_no_run() -> None:
    """Test público 9: el modelo contó las vocales; la query no muestra el
    símbolo, dice la palabra 'asterisks'. Queda un solo carácter."""
    prompt = "Replace all vowels in 'Programming is fun' with asterisks"
    assert _collapse_repeated_run("****", prompt) == "*"


def test_collapse_respects_the_count_the_query_shows() -> None:
    """El agujero medido de la versión simple: si la query muestra '***' y el
    modelo contó cinco, el resultado es '***' y NO '*'."""
    assert _collapse_repeated_run("*****", "Replace vowels with ***") == "***"


def test_collapse_works_for_any_repeated_symbol() -> None:
    """La regla es agnóstica de vocabulario: sirve para cualquier símbolo."""
    assert _collapse_repeated_run("=====", "Set padding to ===") == "==="


def test_collapse_keeps_run_that_is_verbatim_in_prompt() -> None:
    """Si el modelo copió la corrida, la repetición es INTENCIONAL."""
    assert _collapse_repeated_run("**", "Replace vowels with **") == "**"


def test_collapse_skips_single_character() -> None:
    """Sin repetición no hay conteo que deshacer."""
    assert _collapse_repeated_run("*", "Replace vowels with asterisks") == "*"


def test_collapse_skips_values_that_are_not_a_pure_run() -> None:
    """Más de un carácter distinto → no es una corrida de repeticiones."""
    assert _collapse_repeated_run("NUMBERS", "Replace with NUMBERS") == "NUMBERS"
    assert _collapse_repeated_run("dog", "Substitute cat with dog") == "dog"


def test_collapse_skips_block_repetition() -> None:
    """La variante que matamos por medición: reconocer 'ababab' como bloque
    repetido daría '**' sobre el caso real (4 iguales = 2 bloques de 2)."""
    assert _collapse_repeated_run("abababab", "Repeat the unit ab") == "abababab"


def test_collapse_keeps_real_paths_and_encodings_intact() -> None:
    """Los valores legítimos con repetición tienen chars distintos: no tocan."""
    assert _collapse_repeated_run("/home/user/data.json", "Read the file at /home/user/data.json") == \
        "/home/user/data.json"
    assert _collapse_repeated_run("utf-8", "with utf-8 encoding") == "utf-8"
    assert _collapse_repeated_run("latin-1", "with latin-1 encoding") == "latin-1"


# --------------------------------------------------------------------------
# _repair_string_value — la cadena A -> B -> C
# --------------------------------------------------------------------------
def test_chain_applies_the_repair_that_has_evidence() -> None:
    """Cada regla sólo dispara si la anterior no tenía evidencia."""
    # A: el valor es verbatim y con puntuación líder → la estira.
    assert _repair_string_value("home/user/data.json", "Read the file at /home/user/data.json") == \
        "/home/user/data.json"
    # B: no es verbatim pero aparece al quitar comillas.
    assert _repair_string_value("Say hello to {name}", 'Format template: Say "hello" to {name}') == \
        'Say "hello" to {name}'
    # C: corrida de repetidos sin respaldo en la query.
    assert _repair_string_value("****", "Replace all vowels with asterisks") == "*"
    # Ninguna tiene evidencia → intacto.
    assert _repair_string_value("dog", "Substitute cat with dog") == "dog"


def test_chain_fixes_template_case_end_to_end() -> None:
    """Test privado 11 medido a través de la función real del pipeline."""
    prompt = 'Format template: Say "hello" to {name}'
    payload = {"name": "fn_format_template", "parameters": {"template": "Say hello to {name}"}}
    call = build_function_call(prompt, payload)
    assert call.parameters["template"] == 'Say "hello" to {name}'


def test_chain_fixes_replacement_case_end_to_end() -> None:
    """Test público 9 medido a través de la función real del pipeline."""
    prompt = "Replace all vowels in 'Programming is fun' with asterisks"
    payload = {"name": "fn_substitute_string_with_regex", "parameters": {"replacement": "****"}}
    call = build_function_call(prompt, payload)
    assert call.parameters["replacement"] == "*"


def test_chain_leaves_non_string_values_untouched() -> None:
    """Números/bools/null no entran a la cadena de reparaciones."""
    call = build_function_call(
        "What is the product of 3 and 5?",
        {"name": "fn_multiply_numbers", "parameters": {"a": 3, "b": 5.0}},
    )
    assert call.parameters == {"a": 3, "b": 5.0}


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
