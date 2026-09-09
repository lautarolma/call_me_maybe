"""Unit tests for the input, function and vocab loaders (no model required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import BYTE_CATEGORY, load_vocab

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "input"


class TestLoadPrompts:
    def test_loads_all_prompts_from_test_file(self) -> None:
        prompts = load_prompts(DATA_DIR / "function_calling_tests.json")
        assert len(prompts) == 11
        assert all(isinstance(p, str) and p for p in prompts)

    def test_loads_list_of_strings(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(["hello", "world"]), encoding="utf-8")
        assert load_prompts(f) == ["hello", "world"]

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_prompts(f)

    def test_invalid_entry_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text(json.dumps([{"not_a_prompt": 1}]), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid prompt format"):
            load_prompts(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_prompts(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "broken.json"
        f.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(f)


class TestLoadFunctions:
    def test_loads_five_functions_from_test_file(self) -> None:
        functions = load_functions(DATA_DIR / "functions_definition.json")
        assert len(functions) == 5
        names = {fn.name for fn in functions}
        assert names == {
            "fn_add_numbers",
            "fn_greet",
            "fn_reverse_string",
            "fn_get_square_root",
            "fn_substitute_string_with_regex",
        }

    def test_duplicate_names_raise(self, tmp_path: Path) -> None:
        f = tmp_path / "dupes.json"
        payload = [
            {
                "name": "fn_dup",
                "description": "first",
                "parameters": {"a": {"type": "number"}},
                "returns": {"type": "number"},
            },
            {
                "name": "fn_dup",
                "description": "second",
                "parameters": {"b": {"type": "number"}},
                "returns": {"type": "number"},
            },
        ]
        f.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate function name"):
            load_functions(f)

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty_functions.json"
        f.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty JSON array"):
            load_functions(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_functions(tmp_path / "nope.json")

    def test_not_a_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "notlist.json"
        f.write_text(json.dumps({"name": "fn_x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_functions(f)


class TestLoadVocab:
    def test_builds_preindexed_structures(self, tmp_path: Path) -> None:
        vocab_file = tmp_path / "vocab.json"
        raw = {"{": 0, '"name"': 1, "Ġworld": 2, "}": 3, "<|endoftext|>": 4}
        vocab_file.write_text(json.dumps(raw), encoding="utf-8")

        class FakeModel:
            def get_path_to_vocab_file(self) -> str:
                return str(vocab_file)

            def decode(self, ids: list[int]) -> str:
                decoded = {
                    0: "{",
                    1: '"name"',
                    2: " world",  # byte-level token: 'Ġ' decodes to a space
                    3: "}",
                    4: "",  # special token decodes to empty
                }
                return decoded[ids[0]]

        vocab = load_vocab(FakeModel())
        assert vocab.vocab_size == 5
        assert vocab.token2id["{"] == 0
        assert vocab.id2token[2] == "Ġworld"
        assert vocab.id2decoded[2] == " world"
        # Indexed by DECODED first char, not the raw byte-encoded key char.
        assert 0 in vocab.tokens_starting_with["{"]
        assert 2 in vocab.tokens_starting_with[" "]
        assert 3 in vocab.tokens_starting_with["}"]
        # Special/undecodable tokens are parked in the <byte> category.
        assert 4 in vocab.tokens_starting_with[BYTE_CATEGORY]
