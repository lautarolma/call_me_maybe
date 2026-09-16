"""Trie (prefix tree) para nombres de funciones del constrained decoder.

Tarea 3.2 — Construido dinámicamente desde los nombres de las funciones en
functions_definition.json.  Cada nodo es un carácter del nombre; los nodos
terminales llevan el nombre completo y `is_end = True`.  Zero hardcoding:
si mañana se agregan funciones al JSON, el trie se adapta automáticamente.

API pública (lo que usan el token_filter y el facade):
  - build_trie(names)        → raíz del trie
  - find_node(prefix)        → nodo del prefijo o None
  - valid_next_chars(prefix) → chars que extienden al menos un nombre
  - is_complete_name(prefix) → True si el prefijo ES exactamente un nombre
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrieNode:
    """Nodo de un trie de caracteres.

    slots=True por la misma razón que state.py (Decisión 8 del plan): el
    filter (Task 3.4) recorre este trie por cada token candidato; slots
    elimina __dict__ (menos memoria, acceso más rápido).
    """

    children: dict[str, TrieNode] = field(default_factory=dict)
    function_name: str | None = None
    is_end: bool = False


def build_trie(function_names: list[str]) -> TrieNode:
    """Construye un trie desde una lista de nombres de función.

    Cada carácter del nombre genera un nivel de nodo.  Al terminar cada
    nombre se marca el nodo como terminal con `is_end = True` y se almacena
    el nombre completo en `function_name`.
    """
    root = TrieNode()
    for name in function_names:
        node = root
        for char in name:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.function_name = name
        node.is_end = True
    return root


def find_node(root: TrieNode, prefix: str) -> TrieNode | None:
    """Recorre el trie con *prefix* y devuelve el nodo resultante.

    Retorna ``None`` si algún carácter del prefijo no existe en el trie
    (path inexistente).
    """
    node = root
    for char in prefix:
        if char not in node.children:
            return None
        node = node.children[char]
    return node


def valid_next_chars(root: TrieNode, prefix: str) -> set[str]:
    """Devuelve los caracteres que, añadidos a *prefix*, mantienen al menos
    un nombre como candidato válido.

    Si *prefix* no existe en el trie (path inexistente), retorna un set
    vacío — significa que el modelo está generando un nombre inválido.
    """
    node = find_node(root, prefix)
    if node is None:
        return set()
    return set(node.children.keys())


def is_complete_name(root: TrieNode, prefix: str) -> bool:
    """True si *prefix* coincide exactamente con un nombre de función."""
    node = find_node(root, prefix)
    return node is not None and node.is_end
