"""Token filter del constrained JSON decoder (Task 3.4).

POR QUÉ EXISTE ESTE MÓDULO (por dentro):
- La generación sale de TODOS los ids del vocabulario (~151K en Qwen). Por
  cada step, compute_allowed_ids() reduce ese espacio a los tokens que
  mantienen el output como JSON válido (sintaxis + semántica). Es la capa
  MÁS caliente del pipeline: el generator (Task 4.1) la llama una vez por
  token generado y hace argmax sobre el resultado (por eso logits viaja en
  la firma, aunque acá NO se consume: el masking lógico YA ocurrió cuando
  un id queda fuera del set).
- La validación está repartida por diseño:
    Fase 1 (pre-filtro)  → state.expected_first_chars() + pre-índice
                           Vocab.tokens_starting_with. O(1) sobre el vocab.
    Fase 2 (sintaxis)    → DecoderState.simulate(): la state machine.
    Fase 3 (semántica)   → SchemaContext.allows_token(): trie + schema.

TRES DESVÍOS DOCUMENTADOS del pseudocódigo del plan (PLAN_DIDACTICO L1615):
1. Fase 2 usa vocab.id2decoded, NO vocab.id2token. La state machine trabaja
   con el texto DECODIFICADO ('Ġthe' → ' the'): es lo que el validador
   realmente compara. Vocab mantiene las dos vistas a propósito (ver
   vocab_loader.py); id2token aporta tokens byte-mapheados que romperían
   la validación char por char.
2. El wildcard '*' de expected_first_chars se interpreta acá como "saltarse
   el pre-filtro": se toman TODOS los buckets del pre-índice menos el de
   tokens no decodificables (BYTE_CATEGORY = '<byte>'). Las keys y strings
   libres pueden empezar con cualquier carácter real.
3. _is_clean_utf8() se aplica al texto DECODIFICADO (no al crudo del plan):
   los bytes UTF-8 incompletos de un token aparecen como U+FFFD o
   surrogates JUSTO en la decodificación (model.decode no siempre lanza).

QUÉ NO HACE (separación de concerns):
- No conoce el schema ni los nombres de funciones: Fase 1 y 2 son pura
  sintaxis; Fase 3 delega en SchemaContext. El argmax vive en Task 4.1.
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
from src.decoder.trie import TrieNode
from src.loader.vocab_loader import BYTE_CATEGORY, Vocab


def _is_clean_utf8(text: str) -> bool:
    """True si el texto decodificado no tiene marcadores de bytes inválidos.

    CÓMO FUNCIONA (por dentro):
    - U+FFFD (replacement character): lo que produce el decode de una
      secuencia de bytes UTF-8 inválida. La state machine lo aceptaría como
      un char más de un string, pero NO es texto real del usuario: el token
      está contaminado y hay que descartarlo.
    - Surrogates (U+D800..U+DFFF): no son scalar values; str.encode('utf-8')
      los rechaza. Si entraran al output, romperían el encode final.
    - La comparación "\ud800" <= ch <= "\udfff" es correcta para chars
      individuales: Python compara strings por code point.
    """
    return "\ufffd" not in text and not any(
        "\ud800" <= ch <= "\udfff" for ch in text
    )


def compute_allowed_ids(
    state: DecoderState,
    schema: SchemaContext,
    vocab: Vocab,
    trie: TrieNode,
    logits: list[float],
) -> set[int]:
    """Computa el set de ids permitidos para el próximo step de generación.

    Args:
        state: Estado COMMITEADO del decoder (el del token ganador anterior).
        schema: SchemaContext ya sincronizado con state (update(state)).
        vocab: Vocabulario pre-indexado (id2decoded + tokens_starting_with).
        trie: Trie de nombres de función (build_trie).
        logits: Preferencias del modelo (NO se consumen acá; el argmax vive
            en Task 4.1 sobre el set retornado — por eso la firma del plan).

    Returns:
        set de ids cuyo texto decodificado mantiene el output válido.

    CÓMO FUNCIONA (por dentro):
    - Fase 1: junta los buckets del pre-índice para los chars esperados. Si
      expected_first_chars() trae '*' (key o string abiertos), se toman
      TODOS los buckets reales (nunca BYTE_CATEGORY): cualquier carácter es
      legal para empezar/continuar una key o un string.
    - Fase 2: por cada candidato, el texto DECODIFICADO (desvío 1) debe
      pasar _is_clean_utf8 (desvío 3) y simulate(): si cualquier carácter
      rompe la gramática JSON, el token completo es inválido (el ídem del
      plan: "1 token = 1 secuencia de chars atómica").
    - Fase 3: SchemaContext.allows_token() valida semántica (trie, keys,
      tipos, cierres). PURA: no muta ni state ni schema.
    """
    expected_chars = state.expected_first_chars()

    # Fase 1: pre-filtro por primer carácter (decodificado).
    if "*" in expected_chars:
        candidate_ids: set[int] = set()
        for first_char, ids in vocab.tokens_starting_with.items():
            if first_char != BYTE_CATEGORY:
                candidate_ids.update(ids)
    else:
        candidate_ids = set()
        for char in expected_chars:
            candidate_ids.update(vocab.tokens_starting_with.get(char, set()))

    # Fase 2: validación char-by-char (state machine) sobre texto decodificado.
    allowed_ids: set[int] = set()
    for token_id in candidate_ids:
        decoded = vocab.id2decoded.get(token_id)
        if decoded is None or not _is_clean_utf8(decoded):
            continue
        valid, new_state = state.simulate(decoded)
        if not valid:
            continue

        # Fase 3: schema constraints (trie, keys, tipos, cierres).
        if schema.allows_token(decoded, new_state, trie):
            allowed_ids.add(token_id)

    return allowed_ids
