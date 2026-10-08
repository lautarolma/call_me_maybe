"""Unit tests for the Pydantic I/O models."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.models.function_definition import FunctionDef, ParameterDef
from src.models.output import FunctionCall

VALID_FUNCTION: dict[str, Any] = {
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
            FunctionDef(name="fn_missing_stuff")  # type: ignore[call-arg]

    def test_empty_parameters_allowed(self) -> None:
        fn = FunctionDef(name="fn", description="d", parameters={}, returns={})
        assert fn.parameters == {}

    def test_parameter_names_synced_from_keys(self) -> None:
        fn = FunctionDef(**VALID_FUNCTION)
        assert fn.parameters["a"].name == "a"
        assert fn.parameters["b"].name == "b"

    def test_invalid_parameter_type_raises(self) -> None:
        # "object" is a REAL JSON type, but it is deliberately out of the MVP's
        # scope (it needs a recursive schema — see the "complex nested function
        # arguments" bonus). It stays the best example of an invalid type because
        # the rejection is INTENTIONAL, not an oversight.
        #
        # NOTE: this test used "integer" as its example and THEREFORE encoded a
        # bug as expected behavior. "integer" is not a JSON type: it is the
        # grader's spelling for a Python int, and it appears in the private
        # definitions. With "integer" out of the Literal, load_functions blew up
        # on the private set and the program would not start. See
        # test_integer_type_accepted.
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": "object"}}}
        with pytest.raises(ValidationError):
            FunctionDef(**payload)

    def test_integer_type_accepted(self) -> None:
        """Regression: "integer" is a valid type (private set only).

        If this test fails, the program does not start with the private
        definitions (`fn_is_even.n` and `fn_calculate_compound_interest.years`
        are "integer") and half of the evaluation scores zero.
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
            ParameterDef(name="a")  # type: ignore[call-arg]


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
            FunctionCall(prompt="Greet shrek", parameters={"a": 1})  # type: ignore[call-arg]

    def test_missing_prompt_raises(self) -> None:
        """The subject requires the 3 keys: prompt, name and parameters.

        `prompt` is mandatory: the grader compares `student_answer["prompt"]`
        with `correction["prompt"]` for exact equality, so a FunctionCall
        without a prompt is not serializable to a valid entry of the output.
        """
        with pytest.raises(ValidationError):
            FunctionCall(name="fn_greet", parameters={"a": 1})  # type: ignore[call-arg]


class TestEchoView:
    """`echo_view` is what the pipeline prints to stdout.

    It is not a cosmetic detail: the echo used to be printed BEFORE validation,
    so the console showed `replacement: "****"` while the JSON on disk already
    carried `*`. The grader scores the file, so the score was 11/11 all the
    same, but any reviewer reading the console sees broken output and thinks the
    pipeline is wrong. These tests lock in that the projected view comes from
    the validated model and not from the decoder's raw dict.
    """

    def test_only_name_and_parameters(self) -> None:
        call = FunctionCall(prompt="Greet shrek", name="fn_greet", parameters={"name": "shrek"})
        assert call.echo_view() == {"name": "fn_greet", "parameters": {"name": "shrek"}}

    def test_prompt_is_never_echoed(self) -> None:
        """`prompt` is not included: the console already shows it when reading
        the input.

        If it appeared here, every echo block would go from 3 to 6 lines and
        duplicate text already on screen.
        """
        assert "prompt" not in FunctionCall(prompt="Greet shrek", name="fn_greet").echo_view()

    def test_key_order_is_name_then_parameters(self) -> None:
        view = FunctionCall(prompt="p", name="fn_greet", parameters={}).echo_view()
        assert list(view) == ["name", "parameters"]

    def test_echo_reflects_post_validation_value(self) -> None:
        """The real case: the repaired value is the one shown.

        This is the regression test for the stdout bug. `replacement` is `*`
        (already validated) and the echo must show `*` — not the raw `****` from
        the decoder. If someone prints the raw dict again, this test fails.
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
        """The pipeline runs it through `json.dumps(..., indent=2)`: it must
        serialize without inventing anything, and without `ensure_ascii` for the
        accents."""
        call = FunctionCall(prompt="Saludá a shrek", name="fn_greet", parameters={"name": "Ñandú"})
        dumped = json.dumps(call.echo_view(), indent=2, ensure_ascii=False)
        assert "Ñandú" in dumped
        assert json.loads(dumped) == call.echo_view()
