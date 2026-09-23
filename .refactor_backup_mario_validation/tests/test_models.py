"""Unit tests for the Pydantic I/O models."""

from __future__ import annotations

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
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": "integer"}}}
        with pytest.raises(ValidationError):
            FunctionDef(**payload)

    def test_non_string_parameter_type_raises(self) -> None:
        payload = {**VALID_FUNCTION, "parameters": {"a": {"type": 123}}}
        with pytest.raises(ValidationError):
            FunctionDef(**payload)

    def test_all_documented_types_accepted(self) -> None:
        for allowed in ("string", "number", "boolean", "null"):
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
        call = FunctionCall(name="fn_greet", parameters={"name": "shrek"})
        assert call.name == "fn_greet"
        assert call.parameters == {"name": "shrek"}

    def test_default_empty_parameters(self) -> None:
        call = FunctionCall(name="fn_greet")
        assert call.parameters == {}

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            FunctionCall(parameters={"a": 1})
