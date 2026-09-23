"""Tests del trie de nombres de funciones (Task 3.2).

Construcción desde las 5 funciones reales del proyecto, prefix matching,
y validación de nombres completos.  La convención del proyecto: los tests
replican los datos reales de functions_definition.json.
"""

import pytest

from src.decoder.trie import TrieNode, build_trie, find_node, is_complete_name, valid_next_chars

# Mismas 5 funciones que data/input/functions_definition.json
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
        """La raíz no es nombre final y tiene hijos por cada primer carácter."""
        assert not trie.is_end
        assert trie.function_name is None
        assert "f" in trie.children

    def test_common_prefix_shared(self, trie: TrieNode) -> None:
        """`fn_` es un prefijo común: una sola rama f→n→_."""
        node = find_node(trie, "fn_")
        assert node is not None
        assert set(node.children.keys()) == {"a", "g", "r", "s"}

    def test_build_with_empty_list(self) -> None:
        trie = build_trie([])
        assert trie.children == {}

    def test_duplicate_names_store_once(self) -> None:
        """Construir con duplicados no rompe; el nodo terminal queda igual."""
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
        """fn_g es prefijo existente; el path se corta donde diverge."""
        assert find_node(trie, "fn_z") is None


class TestValidNextChars:
    def test_acceptance_criteria_fn_a(self, trie: TrieNode) -> None:
        """Criterio de aceptación del plan: valid_next_chars("fn_a") == {"d"}."""
        assert valid_next_chars(trie, "fn_a") == {"d"}

    def test_fn_g_shared_branch(self, trie: TrieNode) -> None:
        """fn_greet y fn_get_square_root comparten fn_g y divergen ahí:
        greet sigue con 'r', get_square_root con 'e'."""
        assert valid_next_chars(trie, "fn_g") == {"e", "r"}

    def test_fn_reverse(self, trie: TrieNode) -> None:
        assert valid_next_chars(trie, "fn_reverse") == {"_"}

    def test_inexistent_prefix_empty_set(self, trie: TrieNode) -> None:
        assert valid_next_chars(trie, "xyz") == set()

    def test_empty_prefix_returns_first_chars(self, trie: TrieNode) -> None:
        """Prefijo vacío = raíz: todos los primeros caracteres posibles."""
        assert valid_next_chars(trie, "") == {"f"}

    def test_full_name_has_no_char_after(self, trie: TrieNode) -> None:
        """Un nombre completo no tiene más hijos (no es prefijo de otro)."""
        assert valid_next_chars(trie, "fn_add_numbers") == set()


class TestIsCompleteName:
    def test_full_names_are_complete(self, trie: TrieNode) -> None:
        for name in FUNCTIONS:
            assert is_complete_name(trie, name), f"{name} debería ser completo"

    def test_prefix_is_not_complete(self, trie: TrieNode) -> None:
        assert not is_complete_name(trie, "fn_add")
        assert not is_complete_name(trie, "fn_")

    def test_inexistent_name_not_complete(self, trie: TrieNode) -> None:
        assert not is_complete_name(trie, "fn_other")

    def test_empty_prefix_not_complete(self, trie: TrieNode) -> None:
        """La raíz no representa un nombre completo."""
        assert not is_complete_name(trie, "")

    def test_prefix_of_another_is_not_complete(self, trie: TrieNode) -> None:
        """fn_get es prefijo de fn_get_square_root pero no es nombre completo."""
        assert not is_complete_name(trie, "fn_get")
