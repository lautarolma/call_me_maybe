"""State machine of the constrained JSON decoder.

The model does not generate free JSON: on every step this machine
decides, from the current state, whether a character — and therefore a
complete token — is legal. It is the pipeline's syntactic arbiter and
its hottest layer: ``compute_allowed_ids`` calls ``simulate()`` once
per candidate token (up to ~151K ids), hence ``@dataclass(slots=True)``
and never pydantic — model instantiation would dominate the loop.

It knows nothing about the schema (which keys exist, which types are
expected: that is ``schema_validator``'s job). Only JSON syntax over
the subject's subset: an object with keys, string/number/bool/null
values and one level of nesting (``parameters``). The one concession:
``name_buffer`` accumulates the text of the output's "name" value —
bookkeeping the schema reads, not syntax validation.

Two use paths:
- ``simulate(token_text) -> (bool, DecoderState)`` explores without
  touching the real state; the token filter calls it per candidate.
- ``update_from_text(token_text) -> bool`` advances the real state with
  the winning token, atomically: any failing character leaves the
  state exactly as it was.
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass, field
from enum import Enum

# JSON whitespace: space, tab, newline, carriage return. Nothing else.
_WS = " \t\n\r"
_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"
# Simple JSON escapes: \" \\ \/ \n \t \r \b \f (\uXXXX is separate: it
# consumes 4 hex digits and can be split across tokens).
_SIMPLE_ESCAPES = frozenset('"\\/nrtbf')

# JSON number grammar:  -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
#   _NUMBER_PREFIX_RE: the "half-finished" version (accepts "2." and "2e+")
#   for incremental char-by-char validation. The leading "-" joins in
#   _step_colon() and must be followed by a digit. The * allows zero or
#   more characters in the parts that may still be pending. Leading zeros
#   ("01") are rejected: the first alternative tolerates only "0" alone,
#   and the second cannot start with zero. Watch the decimal alternation:
#   "2.e" MUST fail (a dot with no digits leaves the fraction pending; the
#   exponent is only legal AFTER at least one digit) — hence: fraction
#   with digits + optional exponent | dot with pending digits (no
#   exponent) | exponent.
_NUMBER_PREFIX_RE = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]*)?|\.[0-9]*|[eE][+-]?[0-9]*)?"
)
#   _NUMBER_RE: strict version deciding whether the buffer HOLDS a
#   complete number when the value closes ("," / "}" / whitespace).
_NUMBER_RE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")


class DecoderPhase(str, Enum):
    """Phases of the output-JSON state machine.

    ``str, Enum`` so each member IS its name (``DecoderPhase.ROOT ==
    "ROOT"``): readable reprs in logs and direct comparison against
    strings without casting.
    """

    ROOT = "ROOT"                    # initial state: waiting for '{'
    OBJECT_OPEN = "OBJECT_OPEN"      # '{' read: waiting for the first '"'
    IN_OBJECT = "IN_OBJECT"          # inside the output object: key or '}'
    KEY_START = "KEY_START"          # opening '"' of a key read
    IN_KEY = "IN_KEY"                # accumulating key characters
    KEY_END = "KEY_END"              # closing '"' read: waiting for ':'
    COLON = "COLON"                  # ':' read: waiting for the value
    VALUE_START = "VALUE_START"      # reserved; unreachable (no transition sets it)
    IN_STRING_VALUE = "IN_STRING_VALUE"
    IN_NUMBER_VALUE = "IN_NUMBER_VALUE"
    IN_BOOL_VALUE = "IN_BOOL_VALUE"  # true / false
    IN_NULL_VALUE = "IN_NULL_VALUE"  # null
    ESCAPE_IN_STRING = "ESCAPE_IN_STRING"
    VALUE_END = "VALUE_END"          # value closed: ',' or '}'
    PARAMS_OBJECT = "PARAMS_OBJECT"  # inside the parameters object (depth 1)
    COMPLETE = "COMPLETE"            # final '}': generation stops


@dataclass(slots=True)
class DecoderState:
    """Mutable decoder state. ONE per generation (not per token).

    ``slots=True`` and never pydantic: the inner loop copies this object
    for every candidate token, so the layout must be trivial to copy —
    ``__slots__`` drops ``__dict__``, and ``keys_enclosed`` is never
    mutated in place (always replaced by a fresh set, see
    ``_register_params_key``), which keeps ``copy.copy()`` safe.
    """

    phase: DecoderPhase = DecoderPhase.ROOT
    current_key: str = ""              # key whose value is being read
    keys_enclosed: set[str] = field(default_factory=set)  # parameters keys only
    depth: int = 0                     # 0 = output object, 1 = parameters
    number_buffer: str = ""            # number being accumulated
    # ⚠ DOCUMENTED DEVIATION: the machine accumulates the TEXT of the
    # output object's "name" value (depth 0) so SchemaContext can resolve
    # the selected function. Same spirit as keys_enclosed: bookkeeping the
    # schema READS, not syntax validation. Escapes are skipped: the buffer
    # holds the "decoded" name (\u0066n_... -> fn_...). Why here and not
    # in the schema: one BPE token can mix structure and content
    # ('fn_add_numbers", "parameters": {'); rebuilding the name's span
    # from the post-token state would force re-simulating the token. The
    # only place that sees the characters in context is this machine.
    name_buffer: str = ""              # text of the output's "name" value (depth 0)
    bool_buffer: str = ""              # true/false/null being accumulated
    unicode_remaining: int = 0         # hex digits left in an in-flight \uXXXX

    # ------------------------------------------------------------------ API

    def simulate(self, token_text: str) -> tuple[bool, DecoderState]:
        """Simulate one complete token WITHOUT mutating this state.

        Runs on a shallow copy (cheap with slots; sets are replaced,
        never mutated). If any character fails, the token is invalid:
        it returns ``(False, self)`` — the same original object. On
        success it returns the resulting state so the generator can
        adopt it directly instead of re-simulating the winning token.
        """
        new_state = copy(self)
        for char in token_text:
            if not new_state._advance_char(char):
                return False, self
        return True, new_state

    def update_from_text(self, text: str) -> bool:
        """Advance THIS state with a complete text (the winning token). Atomic.

        Same copy-and-advance as ``simulate``, but at the end the copy's
        fields are committed to ``self``: either the whole text is
        consumed and the state moves on, or ``self`` stays exactly as it
        was — the generator never keeps a half-applied token. Fields are
        copied explicitly (not ``__dict__.update``) because ``slots=True``
        has no ``__dict__``; mypy prefers it too.
        """
        new_state = copy(self)
        for char in text:
            if not new_state._advance_char(char):
                return False
        self.phase = new_state.phase
        self.current_key = new_state.current_key
        self.keys_enclosed = new_state.keys_enclosed
        self.depth = new_state.depth
        self.number_buffer = new_state.number_buffer
        self.name_buffer = new_state.name_buffer
        self.bool_buffer = new_state.bool_buffer
        self.unicode_remaining = new_state.unicode_remaining
        return True

    def expected_first_chars(self) -> set[str]:
        """Characters the NEXT token may start with (Phase 1 of the filter).

        Keys into ``Vocab.tokens_starting_with`` (decoded first character
        -> ids): Phase 1 of ``compute_allowed_ids`` unions the buckets of
        all these characters. ``'*'`` is a wildcard — "any real character
        is possible" (free keys and strings) — which the filter reads as
        "skip the pre-filter"; ``<byte>`` tokens never match, so they
        stay out of generation. For numbers it returns EXACTLY the
        characters the grammar allows: after ``"2."`` only digits; after
        ``"2e"`` digits, ``+/-``, or a terminal.
        """
        phase = self.phase
        if phase is DecoderPhase.ROOT:
            # the output ALWAYS starts with '{' (ws may follow).
            return {"{", *(_WS)}
        if phase is DecoderPhase.OBJECT_OPEN:
            return {'"', *(_WS)}
        if phase is DecoderPhase.IN_OBJECT:
            return {'"', "}", *(_WS)}
        if phase is DecoderPhase.KEY_START or phase is DecoderPhase.IN_KEY:
            return {"*"}  # any character can start/continue a key
        if phase is DecoderPhase.KEY_END:
            return {":", *(_WS)}
        if phase is DecoderPhase.COLON:
            return {'"', "-", "{", "t", "f", "n", *(_DIGITS), *(_WS)}
        if phase is DecoderPhase.IN_STRING_VALUE:
            if self.unicode_remaining > 0:
                return set(_HEX_DIGITS)  # next char(s) of \uXXXX
            return {"*"}
        if phase is DecoderPhase.IN_NUMBER_VALUE:
            # terminals only once the buffer is a COMPLETE number: with
            # "2." (fraction pending) a ',' cannot close the value.
            if self._is_valid_json_number():
                return self._number_next_chars() | {",", "}", *(_WS)}
            return self._number_next_chars()
        if phase is DecoderPhase.IN_BOOL_VALUE or phase is DecoderPhase.IN_NULL_VALUE:
            target = self._literal_target()
            if self.bool_buffer == target:
                return {",", "}", *(_WS)}  # literal complete: terminals
            return {target[len(self.bool_buffer)]}  # the exact next char
        if phase is DecoderPhase.ESCAPE_IN_STRING:
            return {*_SIMPLE_ESCAPES, "u"}
        if phase is DecoderPhase.VALUE_END:
            return {",", "}", *(_WS)}
        if phase is DecoderPhase.PARAMS_OBJECT:
            return {'"', "}", *(_WS)}
        # COMPLETE (or unreachable state): nothing is valid.
        return set()

    # ------------------------------------------------------- char transition

    def _advance_char(self, char: str) -> bool:
        """Process ONE character and move the state machine. False = invalid.

        The native ``match/case`` dispatch (Python 3.10) compiles to a
        CPython jump table — no per-branch object instantiation or dict
        lookup — and groups the phases by domain role (object structure,
        keys, strings, numbers, literals), delegating to concise
        per-role handlers.
        """
        match self.phase:
            case DecoderPhase.ROOT:
                return self._step_root(char)
            case DecoderPhase.OBJECT_OPEN | DecoderPhase.IN_OBJECT | DecoderPhase.PARAMS_OBJECT:
                return self._step_object_structural(char)
            case DecoderPhase.KEY_START | DecoderPhase.IN_KEY | DecoderPhase.KEY_END:
                return self._step_key(char)
            case DecoderPhase.COLON:
                return self._step_colon(char)
            case DecoderPhase.IN_STRING_VALUE | DecoderPhase.ESCAPE_IN_STRING:
                return self._step_string(char)
            case DecoderPhase.IN_NUMBER_VALUE:
                return self._step_number(char)
            case DecoderPhase.IN_BOOL_VALUE | DecoderPhase.IN_NULL_VALUE:
                return self._step_literal(char)
            case DecoderPhase.VALUE_END:
                return self._step_value_end(char)
            case DecoderPhase.COMPLETE:
                return char in _WS
            case _:
                return False

    # -------------------------------------------------- step handlers por rol

    def _step_root(self, char: str) -> bool:
        if char in _WS:
            return True
        if char == "{":
            self.phase = DecoderPhase.OBJECT_OPEN
            return True
        return False

    def _step_object_structural(self, char: str) -> bool:
        if char in _WS:
            return True
        if char == '"':
            self._start_new_key()
            self.phase = DecoderPhase.KEY_START
            return True
        if char == "}":
            if self.phase is DecoderPhase.OBJECT_OPEN:
                return False  # empty root object: the schema demands name/parameters
            if self.phase is DecoderPhase.IN_OBJECT:
                self.phase = DecoderPhase.COMPLETE
                return True
            # PARAMS_OBJECT
            self.depth = 0
            self.phase = DecoderPhase.VALUE_END
            return True
        return False

    def _step_key(self, char: str) -> bool:
        if self.phase is DecoderPhase.KEY_START:
            if char == '"':
                self.phase = DecoderPhase.KEY_END
                return True
            self.current_key += char
            self.phase = DecoderPhase.IN_KEY
            return True

        if self.phase is DecoderPhase.IN_KEY:
            if char == '"':
                self.phase = DecoderPhase.KEY_END
                return True
            self.current_key += char
            return True

        # KEY_END
        if char in _WS:
            return True
        if char == ":":
            self.phase = DecoderPhase.COLON
            return True
        return False

    def _step_colon(self, char: str) -> bool:
        if char in _WS:
            return True
        if char == '"':
            # ⚠ DOCUMENTED DEVIATION: a NEW "name" value starts -> reset the
            # accumulated buffer. Watch the depth: fn_greet's "name" PARAM
            # lives at depth 1 and must NOT reset the output's buffer.
            if self.current_key == "name" and self.depth == 0:
                self.name_buffer = ""
            self.phase = DecoderPhase.IN_STRING_VALUE
            return True
        if char == "{":
            self.depth += 1
            self.phase = DecoderPhase.PARAMS_OBJECT
            return True
        if char == "-" or char in _DIGITS:
            self.number_buffer = char
            self.phase = DecoderPhase.IN_NUMBER_VALUE
            return True
        if char in "tf":
            self.bool_buffer = char
            self.phase = DecoderPhase.IN_BOOL_VALUE
            return True
        if char == "n":
            self.bool_buffer = char
            self.phase = DecoderPhase.IN_NULL_VALUE
            return True
        return False

    def _step_string(self, char: str) -> bool:
        if self.phase is DecoderPhase.ESCAPE_IN_STRING:
            if char == "u":
                self.unicode_remaining = 4
                self.phase = DecoderPhase.IN_STRING_VALUE
                return True
            if char in _SIMPLE_ESCAPES:
                self.phase = DecoderPhase.IN_STRING_VALUE
                return True
            return False

        # IN_STRING_VALUE
        if self.unicode_remaining > 0:
            if char not in _HEX_DIGITS:
                return False
            self.unicode_remaining -= 1
            return True
        if char == '"':
            self._register_params_key()
            self.phase = DecoderPhase.VALUE_END
            return True
        if char == "\\":
            # ⚠ An escape inside the output's "name" value (depth 0) never
            # touches name_buffer (it is skipped, see below) — the trie in
            # schema_validator sees an intact buffer and would let it pass
            # forever. Rejecting HERE, at the backslash itself, is the only
            # place that covers EVERY case no matter how BPE fuses the
            # escape into a token (a complete 2-char '\n' token resolves
            # ESCAPE_IN_STRING -> IN_STRING_VALUE INSIDE simulate(), so the
            # final state schema_validator sees never sits in
            # ESCAPE_IN_STRING and a guard there would miss it). No real
            # function name uses '\\'; parameter VALUES can (the '\\d+'
            # regex of fn_substitute_string_with_regex) — hence this guard
            # is specific to current_key == "name" and depth == 0.
            if self.current_key == "name" and self.depth == 0:
                return False
            self.phase = DecoderPhase.ESCAPE_IN_STRING
            return True
        # ⚠ DOCUMENTED DEVIATION: accumulate the text of the "name" value
        # ONLY for the output object's name (depth 0). Escape-phase chars
        # and \uXXXX hex digits were already skipped above: the buffer ends
        # up with the "decoded" name.
        if self.current_key == "name" and self.depth == 0:
            self.name_buffer += char
        return True

    def _step_number(self, char: str) -> bool:
        if char in _WS or char == "," or char == "}":
            if not self._is_valid_json_number():
                return False
            return self._close_value(char)
        if _NUMBER_PREFIX_RE.fullmatch(self.number_buffer + char) is not None:
            self.number_buffer += char
            return True
        return False

    def _step_literal(self, char: str) -> bool:
        target = self._literal_target()
        if self.bool_buffer == target:
            if char in _WS or char == "," or char == "}":
                return self._close_value(char)
            return False
        if char == target[len(self.bool_buffer)]:
            self.bool_buffer += char
            return True
        return False

    def _step_value_end(self, char: str) -> bool:
        if char in _WS:
            return True
        if char == ",":
            self.phase = (
                DecoderPhase.PARAMS_OBJECT
                if self.depth == 1
                else DecoderPhase.IN_OBJECT
            )
            return True
        if char == "}":
            if self.depth == 1:
                self.depth = 0
                self.phase = DecoderPhase.VALUE_END
            else:
                self.phase = DecoderPhase.COMPLETE
            return True
        return False

    # ------------------------------------------------------------- helpers

    def _start_new_key(self) -> None:
        """Reset the key accumulator when a new key opens (the '"' arrives)."""
        self.current_key = ""

    def _register_params_key(self) -> None:
        """Add current_key to keys_enclosed if the key lives in parameters.

        Why ``depth == 1``: keys_enclosed feeds ``schema_validator`` so it
        knows which required parameters keys were already emitted. The
        output object's keys ("name", "parameters") never go in. Memory
        note: the set is ALWAYS replaced (``set | {...}``), never mutated
        in place, so ``simulate()``'s ``copy.copy()`` stays safe: a
        shallow copy shares the set reference, and in-place mutation
        would contaminate original and copy alike.
        """
        if self.depth == 1 and self.current_key:
            self.keys_enclosed = set(self.keys_enclosed) | {self.current_key}

    def _close_value(self, terminal: str) -> bool:
        """Close the current value with a terminal (ws / ',' / '}').

        Pre: TYPE validation (number grammar or complete literal) already
        ran in the phase branch; only structure happens here — register
        the closed key and move to the waiting phase. Whitespace does not
        consume: the phase stays VALUE_END for the next real ',' or '}'.
        Buffers are cleared so one value's leftovers never leak into the
        next.
        """
        self._register_params_key()
        self.number_buffer = ""
        self.bool_buffer = ""
        if terminal in _WS:
            self.phase = DecoderPhase.VALUE_END
            return True
        if terminal == ",":
            self.phase = (
                DecoderPhase.PARAMS_OBJECT
                if self.depth == 1
                else DecoderPhase.IN_OBJECT
            )
            return True
        # terminal == "}"
        if self.depth == 1:
            self.depth = 0
            self.phase = DecoderPhase.VALUE_END
        else:
            self.phase = DecoderPhase.COMPLETE
        return True

    def _literal_target(self) -> str:
        """Literal the in-flight bool/null buffer is validated against."""
        if self.phase is DecoderPhase.IN_NULL_VALUE:
            return "null"
        return "true" if self.bool_buffer[:1] == "t" else "false"

    def _is_valid_json_number(self) -> bool:
        """True if the buffer is a COMPLETE JSON number (strict regex)."""
        return _NUMBER_RE.fullmatch(self.number_buffer) is not None

    def _number_next_chars(self) -> set[str]:
        """Characters that keep the current number a valid PREFIX.

        Probes each grammar character against the prefix regex: with
        ``number_buffer == "2."`` only digits pass (the fraction is
        mandatory); with ``"2e"`` digits and ``+/-`` pass. This cannot be
        derived from a pair of has_digit/has_dot booleans: the
        accumulated string itself is what decides — two booleans cannot
        express every grammar position (e.g. leading zeros: "01" must
        fail, and "0." must not).
        """
        if not self.number_buffer:
            return {"-", *(_DIGITS)}
        return {
            ch
            for ch in "0123456789+-.eE"
            if _NUMBER_PREFIX_RE.fullmatch(self.number_buffer + ch) is not None
        }
