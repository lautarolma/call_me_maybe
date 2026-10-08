"""Tests for the output boundary: parsing, placeholders and detection of
prompts without textual backing.

WHY THIS FILE EXISTS: `build_results` guarantees the positional alignment the
grader needs with its `zip()`. If that guarantee breaks, the whole score is
ruined (not one test is lost: eleven are lost). It is the most dangerous
function in the repo and it had no tests.

It also covers `find_unsupported_prompts`, the sensor that reports the prompts
for which the decoder had to invent the arguments.
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
    """The decoder wraps the JSON with '\\n\\n' — see `generate`."""
    assert parse_output('\n\n{\n  "name": "fn_greet"\n}\n') == {"name": "fn_greet"}


def test_parse_output_raises_on_garbage() -> None:
    try:
        parse_output("not json at all")
    except json.JSONDecodeError:
        return
    raise AssertionError("it should have raised JSONDecodeError")


# --------------------------------------------------------------------------
# build_function_call
# --------------------------------------------------------------------------
def test_build_function_call_defaults_empty_parameters() -> None:
    """A function without parameters emits only `name`."""
    call = build_function_call("Give me the time", {"name": "fn_get_time"})
    assert call.name == "fn_get_time"
    assert call.parameters == {}


def test_build_function_call_keeps_prompt_verbatim() -> None:
    """The grader compares the prompt with EXACT string equality."""
    prompt = "Reverse the string 'hello'"
    call = build_function_call(prompt, {"name": "fn_reverse_string", "parameters": {"s": "hello"}})
    assert call.prompt == prompt


# --------------------------------------------------------------------------
# _snap_to_query_span — re-anchoring the value to the query's verbatim span
# --------------------------------------------------------------------------
def test_snap_restores_clipped_leading_punctuation() -> None:
    """The model copies the path but loses the leading '/'."""
    prompt = "Read the file at /home/user/data.json with utf-8 encoding"
    payload: dict[str, object] = {
        "name": "fn_read_file",
        "parameters": {"path": "home/user/data.json", "encoding": "utf-8"},
    }
    call = build_function_call(prompt, payload)
    assert call.parameters["path"] == "/home/user/data.json"
    assert call.parameters["encoding"] == "utf-8"  # already starts at a boundary: untouched


def test_snap_leaves_windows_drive_path_untouched() -> None:
    """Starts with 'C', not '/'. The rule does NOT force '/'."""
    prompt = "Read C:\\Users\\john\\config.ini with latin-1 encoding"
    payload: dict[str, object] = {
        "name": "fn_read_file",
        "parameters": {"path": "C:\\Users\\john\\config.ini"},
    }
    call = build_function_call(prompt, payload)
    assert call.parameters["path"] == "C:\\Users\\john\\config.ini"


def test_snap_stops_at_quote_delimiter() -> None:
    """The quote DELIMITS the value: 'hello' is not stretched to "'hello"."""
    call = build_function_call(
        "Reverse the string 'hello'",
        {"name": "fn_reverse_string", "parameters": {"s": "hello"}},
    )
    assert call.parameters["s"] == "hello"


def test_snap_stops_at_alphanumeric_suffix() -> None:
    """'llo' inside 'hello' is a suffix, not a clipped value."""
    call = build_function_call(
        "Give me the last 3 letters of 'hello'",
        {"name": "fn_substring", "parameters": {"s": "llo"}},
    )
    assert call.parameters["s"] == "llo"


def test_snap_leaves_value_absent_from_prompt() -> None:
    """Without a verbatim occurrence rule A has nothing to re-anchor to.

    Rule A is tested DIRECTLY and not through `build_function_call` on purpose:
    the full pipeline does fix this value, but through rule C. The coverage
    that matters here is "A alone does nothing", and `build_function_call` can
    no longer observe it.
    """
    prompt = "Replace all vowels with asterisks"
    assert _snap_to_query_span("****", prompt) == "****"


def test_snap_ignores_non_string_values() -> None:
    """Numbers/bools/null do not go through the re-anchoring."""
    call = build_function_call(
        "What is the product of 3 and 5?",
        {"name": "fn_multiply_numbers", "parameters": {"a": 3.0, "b": 5.0}},
    )
    assert call.parameters == {"a": 3.0, "b": 5.0}


def test_snap_is_noop_with_empty_prompt() -> None:
    """`validate_output` builds with prompt="" → no-op by design."""
    call = build_function_call("", {"name": "fn_x", "parameters": {"s": "home/user"}})
    assert call.parameters["s"] == "home/user"


# --------------------------------------------------------------------------
# _restore_internal_quotes — restoring INTERNAL double quotes
# --------------------------------------------------------------------------
def test_quotes_restores_internal_quotes_dropped_by_model() -> None:
    """Copies the content correctly but swallows the quotes."""
    prompt = 'Format template: Say "hello" to {name}'
    assert _restore_internal_quotes("Say hello to {name}", prompt) == 'Say "hello" to {name}'


def test_quotes_restores_internal_quotes_mirror_case() -> None:
    """Not a one-off: 'hi' instead of 'hello' behaves the same."""
    prompt = 'Format template: Say "hi" to {user}'
    assert _restore_internal_quotes("Say hi to {user}", prompt) == 'Say "hi" to {user}'


def test_quotes_ignores_delimiting_quotes_around_whole_value() -> None:
    """The counterexample that defines the rule: the FRAME's quotes are
    delimiters, not content. This value is NOT touched."""
    prompt = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    value = "Hello 34 I'm 233 years old"
    assert _restore_internal_quotes(value, prompt) == value


def test_quotes_skips_value_already_verbatim() -> None:
    """If the value already appears literally, there is nothing to restore."""
    assert _restore_internal_quotes("hello", 'Say "hello" and hello') == "hello"


def test_quotes_stays_silent_when_ambiguous() -> None:
    """Two candidate occurrences → the rule stays silent instead of guessing."""
    assert _restore_internal_quotes("abc", 'Echo "abc" and abc to stdout') == "abc"


def test_quotes_stays_silent_when_only_delimited_spans_exist() -> None:
    """With no internal quotes in play there is no possible fix."""
    assert _restore_internal_quotes("x", 'Format: "x" and "x"') == "x"


def test_quotes_is_noop_without_double_quotes_in_prompt() -> None:
    """Without double quotes the mutated query is identical to the original."""
    assert _restore_internal_quotes("Say hello", "Say hello") == "Say hello"


def test_quotes_is_noop_with_empty_value() -> None:
    """An empty value is not searched and nothing is returned."""
    assert _restore_internal_quotes("", 'Say "hello"') == ""


# --------------------------------------------------------------------------
# _collapse_repeated_run — undoing the repetition count
# --------------------------------------------------------------------------
def test_collapse_undoes_count_when_query_shows_no_run() -> None:
    """The model counted the vowels; the query does not show the symbol, it
    says the word 'asterisks'. A single character remains."""
    prompt = "Replace all vowels in 'Programming is fun' with asterisks"
    assert _collapse_repeated_run("****", prompt) == "*"


def test_collapse_respects_the_count_the_query_shows() -> None:
    """The measured hole in the simple version: if the query shows '***' and
    the model counted five, the result is '***' and NOT '*'."""
    assert _collapse_repeated_run("*****", "Replace vowels with ***") == "***"


def test_collapse_works_for_any_repeated_symbol() -> None:
    """The rule is vocabulary-agnostic: it works for any symbol."""
    assert _collapse_repeated_run("=====", "Set padding to ===") == "==="


def test_collapse_keeps_run_that_is_verbatim_in_prompt() -> None:
    """If the model copied the run, the repetition is INTENTIONAL."""
    assert _collapse_repeated_run("**", "Replace vowels with **") == "**"


def test_collapse_skips_single_character() -> None:
    """With no repetition there is no count to undo."""
    assert _collapse_repeated_run("*", "Replace vowels with asterisks") == "*"


def test_collapse_skips_values_that_are_not_a_pure_run() -> None:
    """More than one distinct character → it is not a repetition run."""
    assert _collapse_repeated_run("NUMBERS", "Replace with NUMBERS") == "NUMBERS"
    assert _collapse_repeated_run("dog", "Substitute cat with dog") == "dog"


def test_collapse_skips_block_repetition() -> None:
    """The variant we killed by measurement: recognizing 'ababab' as a
    repeated block would give '**' on the real case (4 equal = 2 blocks of 2)."""
    assert _collapse_repeated_run("abababab", "Repeat the unit ab") == "abababab"


def test_collapse_keeps_real_paths_and_encodings_intact() -> None:
    """Legitimate values with repetition have distinct chars: not touched."""
    assert _collapse_repeated_run("/home/user/data.json", "Read the file at /home/user/data.json") == \
        "/home/user/data.json"
    assert _collapse_repeated_run("utf-8", "with utf-8 encoding") == "utf-8"
    assert _collapse_repeated_run("latin-1", "with latin-1 encoding") == "latin-1"


# --------------------------------------------------------------------------
# _repair_string_value — the A -> B -> C chain
# --------------------------------------------------------------------------
def test_chain_applies_the_repair_that_has_evidence() -> None:
    """Each rule only fires if the previous one had no evidence."""
    # A: the value is verbatim with leading punctuation -> it stretches it.
    assert _repair_string_value("home/user/data.json", "Read the file at /home/user/data.json") == \
        "/home/user/data.json"
    # B: it is not verbatim but appears when quotes are stripped.
    assert _repair_string_value("Say hello to {name}", 'Format template: Say "hello" to {name}') == \
        'Say "hello" to {name}'
    # C: a run of repeats without backing in the query.
    assert _repair_string_value("****", "Replace all vowels with asterisks") == "*"
    # None has evidence -> untouched.
    assert _repair_string_value("dog", "Substitute cat with dog") == "dog"


def test_chain_fixes_template_case_end_to_end() -> None:
    """Measured through the pipeline's real function."""
    prompt = 'Format template: Say "hello" to {name}'
    payload: dict[str, object] = {
        "name": "fn_format_template",
        "parameters": {"template": "Say hello to {name}"},
    }
    call = build_function_call(prompt, payload)
    assert call.parameters["template"] == 'Say "hello" to {name}'


def test_chain_fixes_replacement_case_end_to_end() -> None:
    """Measured through the pipeline's real function."""
    prompt = "Replace all vowels in 'Programming is fun' with asterisks"
    payload: dict[str, object] = {
        "name": "fn_substitute_string_with_regex",
        "parameters": {"replacement": "****"},
    }
    call = build_function_call(prompt, payload)
    assert call.parameters["replacement"] == "*"


def test_chain_leaves_non_string_values_untouched() -> None:
    """Numbers/bools/null do not enter the repair chain."""
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
# build_results — THE POSITIONAL ALIGNMENT GUARANTEE
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
    """A broken prompt must NOT drop its entry: `zip()` would misalign everything."""
    prompts = ["ok1", "broken", "ok2"]
    generated = ['{"name": "fn_ok1"}', "}{ no json", '{"name": "fn_ok2"}']
    results = build_results(prompts, generated)
    assert len(results) == 3
    assert results[1].name == _UNKNOWN_FN_SENTINEL
    assert results[1].prompt == "broken"
    # The surrounding ones survive intact: ONE test is lost, not three.
    assert results[0].name == "fn_ok1"
    assert results[2].name == "fn_ok2"


def test_build_results_handles_short_generated_list() -> None:
    """If `generated` comes shorter, it still emits one entry per prompt."""
    results = build_results(["a", "b"], ['{"name": "fn_a"}'])
    assert len(results) == 2
    assert results[1].name == _UNKNOWN_FN_SENTINEL


# --------------------------------------------------------------------------
# find_unsupported_prompts — the "prompt matches nothing" sensor
# --------------------------------------------------------------------------
def test_flags_call_whose_arguments_are_invented() -> None:
    """The real measured case: a weather question -> square_root(100)."""
    prompts = ["What is the weather in Paris tomorrow?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 100.0})]
    assert find_unsupported_prompts(prompts, results) == [0]


def test_accepts_call_whose_value_is_verbatim_in_prompt() -> None:
    prompts = ["What is the square root of 16?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 16.0})]
    assert find_unsupported_prompts(prompts, results) == []


def test_float_matches_prompt_written_without_decimal() -> None:
    """`2.0` in the output vs "2" in the prompt: humans do not write 2.0."""
    prompts = ["Give me the sum of 2 and 3"]
    results = [FunctionCall(prompt=prompts[0], name="fn_sum", parameters={"a": 2.0, "b": 3.0})]
    assert find_unsupported_prompts(prompts, results) == []


def test_match_is_case_insensitive() -> None:
    prompts = ["Reverse the string 'Hello'"]
    results = [FunctionCall(prompt=prompts[0], name="fn_reverse_string", parameters={"s": "hello"})]
    assert find_unsupported_prompts(prompts, results) == []


def test_one_supported_value_is_enough() -> None:
    """`replacement="****"` is not in the prompt, but the other two are.

    It is an accuracy failure of the model, not a lack of match — the grader
    already measures it. That is why the criterion is 'none', not 'all'.
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
    """With no arguments there is nothing to compare: not reported."""
    prompts = ["What time is it?"]
    results = [FunctionCall(prompt=prompts[0], name="fn_get_time", parameters={})]
    assert find_unsupported_prompts(prompts, results) == []


def test_placeholder_is_not_double_reported() -> None:
    """The sentinel already has its own 'unparseable' warning."""
    prompts = ["whatever"]
    results = [FunctionCall(prompt=prompts[0], name=_UNKNOWN_FN_SENTINEL, parameters={})]
    assert find_unsupported_prompts(prompts, results) == []


def test_booleans_are_not_judged_as_numbers() -> None:
    """`True` is a subclass of `int`: unchecked, it would say '1'."""
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
    """The pipeline may have several; the sensor returns the whole list."""
    prompts = ["What is the weather in Paris?", "Greet shrek", "And in Madrid?"]
    results = [
        FunctionCall(prompt=prompts[0], name="fn_get_square_root", parameters={"a": 100.0}),
        FunctionCall(prompt=prompts[1], name="fn_greet", parameters={"name": "shrek"}),
        FunctionCall(prompt=prompts[2], name="fn_get_square_root", parameters={"a": 5.0}),
    ]
    assert find_unsupported_prompts(prompts, results) == [0, 2]
