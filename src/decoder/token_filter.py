"""Token filter for the constrained JSON decoder.

Reduces the vocabulary (~151K tokens) to those that keep the output
syntactically and semantically valid at each step. Called once per
generated token; when logits are provided, only a top-k subset is
evaluated (M1/M2). The filter is split into three phases:
1. Pre-filter by first character (bucket index).
2. Syntax via DecoderState.simulate() on decoded text.
3. Semantics via SchemaContext.allows_token().
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.trie import TrieNode
from src.loader.vocab_loader import BYTE_CATEGORY, Vocab

# Tiers for top-k masking with M2. Validate candidates in increasing
# batches instead of all at once. M1 returns immediately on success.
TIER_SIZES: list[int] = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def _is_clean_utf8(text: str) -> bool:
    """Return True if decoded text contains no invalid UTF-8 markers.

    Rejects U+FFFD (replacement character) and surrogate code points
    (U+D800–U+DFFF), which would be invalid for UTF-8 re-encoding.

    Args:
        text: Decoded token text.

    Returns:
        True if the text is clean UTF-8.
    """
    return "\ufffd" not in text and not any(
        "\ud800" <= ch <= "\udfff" for ch in text
    )


def compute_allowed_ids(
    state: DecoderState,
    schema: SchemaContext,
    vocab: Vocab,
    trie: TrieNode,
    logits: list[float] | None = None,
    top_k: int = 2000,
) -> set[int]:
    """Compute allowed token IDs for the next generation step.

    Two return contracts exist:
    1. With logits (hot path): validate only the model's top-k candidates
       and return those among them that are valid. This is a subset of all
       valid tokens (not the full set). Advantage: ~K validations instead
       of ~151K.
    2. Without logits (skip-if-single): validate the entire vocabulary and
       return the complete set of valid tokens (used when the size must be
       exactly 1).

    M1 (top-1 opportunistic): if the argmax passes simulate + allows_token,
    return {best_id} immediately (zero extra validations). M2 (top-k with
    tiering): validate ranked candidates in increasing tiers [1,5,10,...,2000]
    and return the ENTIRE TIER that first produces any valid token (not just
    the first valid). Returning the full tier preserves alternatives for the
    fine pass, so a veto of the single best does not cut generation when
    other valid tokens exist in that tier.

    Phases:
    1. Pre-filter by first characters (expected_first_chars). If '*' appears,
       include all real buckets (exclude BYTE_CATEGORY).
    2. Syntax: for each candidate, require clean UTF-8 decoded text and
       state.simulate(decoded) to be valid (one token = one char sequence).
    3. Semantics: schema.allows_token(decoded, new_state, trie).

    Args:
        state: Committed decoder state (after previous token).
        schema: SchemaContext synchronized with state (update called).
        vocab: Indexed vocabulary (id2decoded, tokens_starting_with).
        trie: Trie of allowed function names.
        logits: Model logits; if provided, use M1 and M2. If None, scan full
            vocabulary.
        top_k: Maximum candidates to rank when logits are provided.

    Returns:
        Set of token IDs whose decoded text keeps the output valid.
    """
    expected_chars = state.expected_first_chars()

    # Fase 1: pre-filtro por primer carácter (decodificado).
    _STATIC_PREINDEX_PHASES = {
        DecoderPhase.ROOT,
        DecoderPhase.OBJECT_OPEN,
        DecoderPhase.IN_OBJECT,
        DecoderPhase.KEY_END,
        DecoderPhase.COLON,
        DecoderPhase.ESCAPE_IN_STRING,
        DecoderPhase.VALUE_END,
        DecoderPhase.PARAMS_OBJECT,
    }

    if state.phase in _STATIC_PREINDEX_PHASES:
        candidate_ids: set[int] = vocab.valid_by_phase[state.phase.value]

    elif "*" in expected_chars:
        candidate_ids = set()
        for first_char, ids in vocab.tokens_starting_with.items():
            if first_char != BYTE_CATEGORY:
                candidate_ids.update(ids)
    else:
        candidate_ids = set()
        for char in expected_chars:
            candidate_ids.update(vocab.tokens_starting_with.get(char, set()))

    # ─── OPTIMIZACIÓN (Anexo de Latencia — M1/M2) ───
    # Si logits se proveyeron, usar Top-1 opportunistic (O(1)) y luego
    # Top-K masking (O(K)) para reducir el universo de candidatos.
    if logits is not None:
        # M1: top-1 opportunistic: if the highest-logit token passes
        # simulate + allows_token, return immediately.
        best_id = max(range(len(logits)), key=lambda i: logits[i])
        best_decoded = vocab.id2decoded.get(best_id)
        if (
            best_decoded is not None
            and _is_clean_utf8(best_decoded)
        ):
            valid, new_state = state.simulate(best_decoded)
            if valid and schema.allows_token(best_decoded, new_state, trie):
                return {best_id}

        # M2: top-k masking with tiering. Return the FIRST tier that
        # produces any valid token, and return that ENTIRE tier (not just
        # the first valid). Returning the full tier preserves alternatives
        # for the fine pass.
        import heapq
        ranked_ids = heapq.nlargest(
            top_k, range(len(logits)), key=logits.__getitem__
        )
        ranked_ids = [tid for tid in ranked_ids if tid in candidate_ids]

        checked = 0
        for tier_size in TIER_SIZES:
            end = min(checked + tier_size, len(ranked_ids))
            passing: set[int] = set()
            for idx in range(checked, end):
                token_id = ranked_ids[idx]
                decoded = vocab.id2decoded.get(token_id)
                if decoded is None or not _is_clean_utf8(decoded):
                    continue
                valid, new_state = state.simulate(decoded)
                if not valid:
                    continue
                if schema.allows_token(decoded, new_state, trie):
                    passing.add(token_id)
            checked = end
            if passing:
                return passing
            if checked >= len(ranked_ids):
                break

        # Fallback: si ningún tier encontró uno válido, retornar vacío
        # (el caller manejará el empty set).
        return set()

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
