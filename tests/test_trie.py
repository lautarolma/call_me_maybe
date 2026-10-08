"""Tests for the function-name trie.

Built from the project's 5 real functions, prefix matching, and full-name
validation. The project convention: tests replicate the real data from
functions_definition.json.
"""

import pytest

from src.decoder.trie import TrieNode, build_trie, find_node, is_complete_name, valid_next_chars

# Same 5 functions as data/input/functions_definition.json
FUNCTIONS = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
]


@pytest.fixture
def trie() -> TrieNode:
    return build_trie(FUNCTIONS)


class TestBuildTrie:
    def test_root_is_empty_node(self, trie: TrieNode) -> None:
        """The root is not a final name and has children for each first character."""
        assert not trie.is_end
        assert trie.function_name is None
        assert "f" in trie.children

    def test_common_prefix_shared(self, trie: TrieNode) -> None:
        """`fn_` is a common prefix: a single branch f→n→_."""
        node = find_node(trie, "fn_")
        assert node is not None
        assert set(node.children.keys()) == {"a", "g", "r", "s"}

    def test_build_with_empty_list(self) -> None:
        trie = build_trie([])
        assert trie.children == {}

    def test_duplicate_names_store_once(self) -> None:
        """Building with duplicates does not break; the terminal node stays the same."""
        trie = build_trie(["fn_a", "fn_a"])
        node = find_node(trie, "fn_a")
        assert node is not None
        assert node.is_end
        assert node.function_name == "fn_a"


class TestFindNode:
    def test_find_existing_prefix(self, trie: TrieNode) -> None:
        assert find_node(trie, "fn_add") is not None

    def test_find_full_name(self, trie: TrieNode) -> None:
        node = find_node(trie, "fn_add_numbers")
        assert node is not None
        assert node.is_end
        assert node.function_name == "fn_add_numbers"

    def test_find_inexistent_prefix_returns_none(self, trie: TrieNode) -> None:
        assert find_node(trie, "xyz") is None

    def test_find_empty_prefix_returns_root(self, trie: TrieNode) -> None:
        assert find_node(trie, "") is trie

    def test_find_partial_divergence_returns_none(self, trie: TrieNode) -> None:
        """fn_g is an existing prefix; the path is cut where it diverges."""
        assert find_node(trie, "fn_z") is None


class TestValidNextChars:
    def test_acceptance_criteria_fn_a(self, trie: TrieNode) -> None:
        """Plan acceptance criterion: valid_next_chars("fn_a") == {"d"}."""
        assert valid_next_chars(trie, "fn_a") == {"d"}

    def test_fn_g_shared_branch(self, trie: TrieNode) -> None:
        """fn_greet and fn_get_square_root share fn_g and diverge there:
        greet continues with 'r', get_square_root with 'e'."""
        assert valid_next_chars(trie, "fn_g") == {"e", "r"}

    def test_fn_reverse(self, trie: TrieNode) -> None:
        assert valid_next_chars(trie, "fn_reverse") == {"_"}

    def test_inexistent_prefix_empty_set(self, trie: TrieNode) -> None:
        assert valid_next_chars(trie, "xyz") == set()

    def test_empty_prefix_returns_first_chars(self, trie: TrieNode) -> None:
        """Empty prefix = root: every possible first character."""
        assert valid_next_chars(trie, "") == {"f"}

    def test_full_name_has_no_char_after(self, trie: TrieNode) -> None:
        """A complete name has no further children (it is not a prefix of another)."""
        assert valid_next_chars(trie, "fn_add_numbers") == set()


class TestIsCompleteName:
    def test_full_names_are_complete(self, trie: TrieNode) -> None:
        for name in FUNCTIONS:
            assert is_complete_name(trie, name), f"{name} should be complete"

    def test_prefix_is_not_complete(self, trie: TrieNode) -> None:
        assert not is_complete_name(trie, "fn_add")
        assert not is_complete_name(trie, "fn_")

    def test_inexistent_name_not_complete(self, trie: TrieNode) -> None:
        assert not is_complete_name(trie, "fn_other")

    def test_empty_prefix_not_complete(self, trie: TrieNode) -> None:
        """The root does not represent a complete name."""
        assert not is_complete_name(trie, "")

    def test_prefix_of_another_is_not_complete(self, trie: TrieNode) -> None:
        """fn_get is a prefix of fn_get_square_root but is not a complete name."""
        assert not is_complete_name(trie, "fn_get")
