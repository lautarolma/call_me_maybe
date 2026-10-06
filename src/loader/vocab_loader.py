"""Load the model vocabulary and build pre-indexed token structures.

Byte-level BPE (GPT-2 / tiktoken style) routes raw UTF-8 bytes through a
byte-to-unicode table, so the vocab key ``'Ġthe'`` really represents
``' the'`` (``Ġ`` is byte 0x20, the space). Indexing by the first character
of the raw key would therefore never match the text the constrained decoder
compares: every token is decoded once through the SDK's public ``decode``
API at startup and indexed by its first *decoded* character.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model

#: Bucket key for tokens whose text cannot be decoded cleanly (invalid byte
#: sequences, special tokens that decode to empty). These tokens are never
#: reachable while generating JSON text.
BYTE_CATEGORY = "<byte>"


# slots=True: fixed attribute layout without __dict__ — faster attribute
# reads on the decoder's hot path (vocab.valid_by_phase, etc.).
@dataclass(slots=True)
class Vocab:
    """Pre-indexed vocabulary for constrained decoding.

    Two views are kept deliberately: raw vocab text (``token2id``/``id2token``)
    for the tokenizer, decoded text (``id2decoded``) for the JSON validator.
    ``Vocab`` is the boundary between both worlds.

    Attributes:
        token2id: Token text (vocab.json key) -> token id.
        id2token: Token id -> token text (vocab.json key).
        id2decoded: Token id -> decoded text (via the SDK tokenizer).
        tokens_starting_with: First decoded character -> set of token ids
            whose decoded text starts with it. A set, not a list: the filter
            asks "is this id allowed?" every generation step and set
            membership is O(1).
        valid_by_phase: Phase name (a plain string, not the ``DecoderPhase``
            enum, so this module never imports the decoder) -> set of token
            ids valid for that phase, precomputed at startup so the hot
            filter skips both the character scan and the state import.
        vocab_size: Total number of tokens in the vocabulary.
    """

    token2id: dict[str, int]
    id2token: dict[int, str]
    id2decoded: dict[int, str]
    tokens_starting_with: dict[str, set[int]]
    vocab_size: int
    valid_by_phase: dict[str, set[int]]


# Simplified mapping: phase name -> set of first-decoded-char that are valid
# for that phase in the COMMON case (no dynamic state dependency). Used to
# build valid_by_phase at startup. Dynamic phases (IN_NUMBER_VALUE) fall back
# to the full filter.
_WS_SIMPLIFIED = frozenset(" \t\n\r")
_DIGITS_SIMPLIFIED = frozenset("0123456789")
_SIMPLE_ESCAPES_SIMPLIFIED = frozenset('"\\/nrtbf')

_STATIC_PHASE_FIRST_CHARS = {
    "ROOT": frozenset({"{"} | _WS_SIMPLIFIED),
    "OBJECT_OPEN": frozenset({'"'} | _WS_SIMPLIFIED),
    "IN_OBJECT": frozenset({'"', "}"} | _WS_SIMPLIFIED),
    "KEY_END": frozenset({":"} | _WS_SIMPLIFIED),
    "COLON": frozenset(
        {'"', "-", "{", "t", "f", "n"}
        | _DIGITS_SIMPLIFIED
        | _WS_SIMPLIFIED
    ),
    "ESCAPE_IN_STRING": (
        _SIMPLE_ESCAPES_SIMPLIFIED | frozenset({"u"})
    ),
    "VALUE_END": frozenset({",", "}"} | _WS_SIMPLIFIED),
    "PARAMS_OBJECT": frozenset({'"', "}"} | _WS_SIMPLIFIED),
}


def load_vocab(model: Small_LLM_Model) -> Vocab:
    """Load the model vocabulary and build pre-indexed structures.

    Costs ``O(V * D)`` once at startup — V vocabulary entries, D the cost
    of decoding one token — so each generation step stays O(1) lookups.

    Args:
        model: Initialized ``Small_LLM_Model``; its vocab file is read and
            its tokenizer decodes every token text.

    Returns:
        A ``Vocab`` with token mappings, decoded text per token, the
        first-character index and the per-phase token sets.
    """
    # SDK helper: resolves vocab.json from the HuggingFace cache, downloading
    # it on first run — that is why the second startup is instant.
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    token2id: dict[str, int] = raw_vocab
    # Inversion is safe only because vocab.json guarantees unique ids;
    # a collision would silently overwrite (standard dict behavior).
    id2token: dict[int, str] = {token_id: token_text for token_text, token_id in token2id.items()}

    id2decoded: dict[int, str] = {}
    tokens_starting_with: dict[str, set[int]] = {}
    for token_text, token_id in token2id.items():
        try:
            # SDK decode: id -> byte-mapped string -> real UTF-8 text, with
            # special tokens skipped (skip_special_tokens=True).
            decoded = model.decode([token_id])
        except Exception:
            # One malformed token must not kill startup; it falls into the
            # <byte> bucket, which is never valid inside JSON text.
            decoded = ""
        if not decoded:
            # Special tokens (</s>) decode to '' — same <byte> bucket.
            tokens_starting_with.setdefault(BYTE_CATEGORY, set()).add(token_id)
            continue
        # Strings iterate by code point, so decoded[0] is correct even when
        # the token starts with a multi-byte character such as 'ñ'.
        first_char = decoded[0]
        id2decoded[token_id] = decoded
        tokens_starting_with.setdefault(first_char, set()).add(token_id)

    # Precompute valid_by_phase from the static first-char table (see the
    # Vocab docstring for why the keys are plain phase-name strings).
    valid_by_phase: dict[str, set[int]] = {}
    for _phase_name, _chars in _STATIC_PHASE_FIRST_CHARS.items():
        _ids: set[int] = set()
        for _ch in _chars:
            _ids.update(tokens_starting_with.get(_ch, set()))
        valid_by_phase[_phase_name] = _ids

    return Vocab(
        token2id=token2id,
        id2token=id2token,
        id2decoded=id2decoded,
        tokens_starting_with=tokens_starting_with,
        vocab_size=len(token2id),
        valid_by_phase=valid_by_phase,
    )
