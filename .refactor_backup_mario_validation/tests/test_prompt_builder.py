"""Unit tests for the prompt builder (no model required)."""

from __future__ import annotations

from pathlib import Path

from src.loader.function_loader import load_functions
from src.models.function_definition import FunctionDef
from src.prompt.prompt_builder import SYSTEM_PROMPT, build_function_list, build_prompt

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "input"

REAL_FUNCTION_NAMES = {
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
}


def real_functions() -> list[FunctionDef]:
    """Load the 5 real functions from the project data (matches the subject)."""
    return load_functions(DATA_DIR / "functions_definition.json")


class TestSystemPrompt:
    def test_has_function_list_placeholder(self) -> None:
        assert "{function_list}" in SYSTEM_PROMPT

    def test_requests_plain_json_output(self) -> None:
        assert "Output ONLY a JSON object" in SYSTEM_PROMPT
        assert '"name"' in SYSTEM_PROMPT
        assert '"parameters"' in SYSTEM_PROMPT


class TestBuildFunctionList:
    def test_contains_all_five_function_names(self) -> None:
        listing = build_function_list(real_functions())
        for name in REAL_FUNCTION_NAMES:
            assert name in listing

    def test_contains_all_parameter_names_and_types(self) -> None:
        listing = build_function_list(real_functions())
        # fn_add_numbers: a y b number; fn_greet: name string;
        # fn_reverse_string: s string; fn_get_square_root: a number;
        # fn_substitute_string_with_regex: source_string/regex/replacement
        for param in ("a (number)", "b (number)", "name (string)", "s (string)",
                      "source_string (string)", "regex (string)", "replacement (string)"):
            assert param in listing, f"missing {param} from listing"

    def test_functions_keep_input_order(self) -> None:
        listing = build_function_list(real_functions())
        first = listing.index("1. fn_add_numbers")
        second = listing.index("2. fn_greet")
        assert first < second

    def test_numbered_entries(self) -> None:
        listing = build_function_list(real_functions())
        assert "1. fn_add_numbers" in listing
        assert "5. fn_substitute_string_with_regex" in listing

    def test_empty_functions_produce_empty_listing(self) -> None:
        assert build_function_list([]) == ""


class TestBuildPrompt:
    def test_includes_system_part(self) -> None:
        prompt = build_prompt(real_functions(), "What is 2+3?")
        assert "You are a function calling assistant" in prompt

    def test_includes_all_functions(self) -> None:
        prompt = build_prompt(real_functions(), "What is 2+3?")
        assert "fn_add_numbers" in prompt
        assert "fn_substitute_string_with_regex" in prompt

    def test_includes_user_query_at_end(self) -> None:
        prompt = build_prompt(real_functions(), "What is 2+3?")
        assert prompt.endswith("\nUser query: What is 2+3?")

    def test_never_empty_with_functions(self) -> None:
        prompt = build_prompt(real_functions(), "What is 2+3?")
        assert isinstance(prompt, str) and prompt

    def test_different_queries_yield_different_prompts(self) -> None:
        p1 = build_prompt(real_functions(), "What is 2+3?")
        p2 = build_prompt(real_functions(), "Reverse 'hola'")
        assert p1 != p2
        assert "What is 2+3?" in p1
        assert "Reverse 'hola'" in p2

    def test_template_has_no_leftover_placeholder(self) -> None:
        prompt = build_prompt(real_functions(), "What is 2+3?")
        assert "{function_list}" not in prompt
