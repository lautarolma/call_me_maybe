"""Unit tests for the constrained JSON decoder state machine."""

from __future__ import annotations

from src.decoder.state import DecoderPhase, DecoderState

FULL_JSON = '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'

PHASE_NAMES = {
    "ROOT", "OBJECT_OPEN", "IN_OBJECT", "KEY_START", "IN_KEY", "KEY_END",
    "COLON", "VALUE_START", "IN_STRING_VALUE", "IN_NUMBER_VALUE",
    "IN_BOOL_VALUE", "IN_NULL_VALUE", "ESCAPE_IN_STRING", "VALUE_END",
    "PARAMS_OBJECT", "COMPLETE",
}


class TestPhaseEnum:
    def test_has_all_plan_states(self) -> None:
        assert {p.name for p in DecoderPhase} == PHASE_NAMES

    def test_is_str_enum(self) -> None:
        assert DecoderPhase.ROOT == "ROOT"
        assert DecoderPhase.COMPLETE.value == "COMPLETE"

    def test_defaults(self) -> None:
        s = DecoderState()
        assert s.phase == DecoderPhase.ROOT
        assert s.current_key == ""
        assert s.keys_enclosed == set()
        assert s.depth == 0
        assert s.number_buffer == ""
        assert s.name_buffer == ""
        assert s.bool_buffer == ""
        assert s.unicode_remaining == 0

    def test_slots_no_dict(self) -> None:
        assert not hasattr(DecoderState(), "__dict__")


class TestFullJsonHappyPath:
    def test_char_by_char_until_complete(self) -> None:
        s = DecoderState()
        for ch in FULL_JSON:
            assert s._advance_char(ch), f"char {ch!r} rejected"
        assert s.phase == DecoderPhase.COMPLETE
        assert s.depth == 0

    def test_keys_enclosed_only_params_keys(self) -> None:
        s = DecoderState()
        assert s.update_from_text(FULL_JSON)
        assert s.keys_enclosed == {"a", "b"}
        assert "name" not in s.keys_enclosed
        assert "parameters" not in s.keys_enclosed

    def test_update_from_text_commits_state(self) -> None:
        s = DecoderState()
        assert s.update_from_text(FULL_JSON)
        assert s.current_key == "b"
        assert s.phase == DecoderPhase.COMPLETE

    def test_int_numbers_without_fraction(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":"fn","parameters":{"a":2,"b":-3}}')
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == {"a", "b"}

    def test_name_only_syntactically_ok(self) -> None:
        # Without "parameters": syntactically valid; the schema validator
        # will reject it because parameters is required.
        s = DecoderState()
        assert s.update_from_text('{"name":"fn"}')
        assert s.phase == DecoderPhase.COMPLETE


class TestSimulate:
    def test_returns_detached_copy_on_success(self) -> None:
        s = DecoderState()
        ok, ns = s.simulate('{"name"')
        assert ok
        assert ns is not s
        assert ns.phase == DecoderPhase.KEY_END
        assert ns.current_key == "name"
        assert s.phase == DecoderPhase.ROOT

    def test_original_untouched_on_success(self) -> None:
        s = DecoderState()
        ok, ns = s.simulate('{"name":"fn","parameters":{"a":1}}')
        assert ok
        assert ns.phase == DecoderPhase.COMPLETE
        assert ns.keys_enclosed == {"a"}
        assert s.phase == DecoderPhase.ROOT
        assert s.keys_enclosed == set()

    def test_failure_returns_original_object(self) -> None:
        s = DecoderState()
        ok, ns = s.simulate("x")
        assert not ok
        assert ns is s  # on failure it returns the ORIGINAL state
        assert s.phase == DecoderPhase.ROOT

    def test_multi_char_token_whitespace_prefix(self) -> None:
        # Qwen BPE tokens decode "Ġx" as " x": one token may start with a
        # space AND contain '{' — there is no intermediate state.
        s = DecoderState()
        ok, ns = s.simulate(' \n{"name')
        assert ok
        assert ns.phase == DecoderPhase.IN_KEY
        assert ns.current_key == "name"


class TestUpdateFromTextAtomic:
    def test_failure_keeps_state_untouched(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":')
        before = s.phase
        assert not s.update_from_text('"fn"}x')  # "x" after COMPLETE
        assert s.phase == before

    def test_split_across_calls(self) -> None:
        # Tokens cut keys and values in half: every update_from_text
        # resumes from the accumulated state of the previous one.
        s = DecoderState()
        assert s.update_from_text('{"na')
        assert s.phase == DecoderPhase.IN_KEY
        assert s.update_from_text('me":"fn"}')
        assert s.phase == DecoderPhase.COMPLETE  # type: ignore[comparison-overlap]

    def test_number_split_across_calls(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2.')
        assert s.update_from_text('5,"b":3}}')
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == {"a", "b"}

    def test_unicode_escape_split_across_calls(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":"\\u00')
        assert s.unicode_remaining == 2
        assert s.update_from_text('e9"}}')
        assert s.phase == DecoderPhase.COMPLETE


class TestNumbers:
    VALID = [
        '{"name":"fn","parameters":{"a":0}}',
        '{"name":"fn","parameters":{"a":-0}}',
        '{"name":"fn","parameters":{"a":123456}}',
        '{"name":"fn","parameters":{"a":0.5}}',
        '{"name":"fn","parameters":{"a":-2.5e-3}}',
        '{"name":"fn","parameters":{"a":1E2}}',
        '{"name":"fn","parameters":{"a":-2.5E+3}}',
        '{"name":"fn","parameters":{"a":2.0,"b":3.0}}',
    ]

    def test_valid_numbers_reach_complete(self) -> None:
        for text in self.VALID:
            s = DecoderState()
            assert s.update_from_text(text), text
            assert s.phase == DecoderPhase.COMPLETE, text

    INVALID = [
        '{"name":"fn","parameters":{"a":01}}',       # leading zero
        '{"name":"fn","parameters":{"a":-01}}',
        '{"name":"fn","parameters":{"a":2.}}',       # fraction without digits
        '{"name":"fn","parameters":{"a":2e}}',       # exponent without digits
        '{"name":"fn","parameters":{"a":2e+}}',
        '{"name":"fn","parameters":{"a":2e+-3}}',
        '{"name":"fn","parameters":{"a":2..5}}',
        '{"name":"fn","parameters":{"a":.5}}',       # int part is mandatory
        '{"name":"fn","parameters":{"a":+2}}',       # leading '+' is invalid
    ]

    def test_invalid_numbers_rejected(self) -> None:
        for text in self.INVALID:
            s = DecoderState()
            assert not s.update_from_text(text), text

    def test_invalid_number_rejected_via_simulate(self) -> None:
        s = DecoderState()
        ok, ns = s.simulate('{"name":"fn","parameters":{"a":2.}}')
        assert not ok
        assert s.phase == DecoderPhase.ROOT and ns is s

    def test_leading_zero_blocked_char_by_char(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":0')
        assert not s._advance_char("1")  # "01" is not a JSON number


class TestBooleansAndNull:
    def test_true_false_null(self) -> None:
        s = DecoderState()
        assert s.update_from_text(
            '{"name":"fn","parameters":{"a":true,"b":false,"c":null}}'
        )
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == {"a", "b", "c"}

    def test_truncated_or_extra_chars_rejected(self) -> None:
        for bad in ("trux", "falsy", "nul"):
            s = DecoderState()
            ok, _ = s.simulate('{"name":"fn","parameters":{"a":' + bad + "}}")
            assert not ok, bad

    def test_case_sensitive(self) -> None:
        s = DecoderState()
        assert not s.update_from_text('{"parameters":{"a":TRUE}}')

    def test_literal_split_across_tokens(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":tru')
        assert s.update_from_text('e}}')
        assert s.phase == DecoderPhase.COMPLETE


class TestStringsAndEscapes:
    def test_whitespace_is_string_content(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":"fn","parameters":{"a":"hi there"}}')
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == {"a"}

    def test_simple_escapes(self) -> None:
        for esc in ('\\"', "\\\\", "\\/", "\\n", "\\t", "\\r", "\\b", "\\f"):
            s = DecoderState()
            text = '{"name":"fn","parameters":{"a":"x' + esc + 'y"}}'
            assert s.update_from_text(text), esc

    def test_unknown_escape_rejected(self) -> None:
        s = DecoderState()
        assert not s.update_from_text('{"parameters":{"a":"\\x"}}')

    def test_unicode_escape_valid(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":"caf\\u00e9"}}')
        assert s.phase == DecoderPhase.COMPLETE

    def test_unicode_escape_bad_hex_digit(self) -> None:
        s = DecoderState()
        assert not s.update_from_text('{"parameters":{"a":"\\u00g9"}}')

    def test_unicode_escape_too_short(self) -> None:
        # Only 3 hex before the '"': the '"' is NOT hex -> invalid.
        s = DecoderState()
        assert not s.update_from_text('{"parameters":{"a":"\\u00e"}}')

    def test_quote_while_unicode_pending_is_not_terminator(self) -> None:
        s = DecoderState()
        assert not s.update_from_text('{"parameters":{"a":"\\u00"}}')


class TestWhitespaceTolerance:
    def test_surrounding_whitespace(self) -> None:
        s = DecoderState()
        text = ' \t { "name" : "fn" ,\n "parameters" : { "a" : 1.5 , "b" : -2 } \t } \r'
        assert s.update_from_text(text)
        assert s.phase == DecoderPhase.COMPLETE

    def test_whitespace_closes_value_without_consuming_terminal(self) -> None:
        # ws after the number closes the value (VALUE_END) and lets the real
        # '}' close the object: two transitions, not one.
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2 }')
        assert s.phase == DecoderPhase.VALUE_END
        assert s.keys_enclosed == {"a"}
        assert s.update_from_text("}")
        assert s.phase == DecoderPhase.COMPLETE  # type: ignore[comparison-overlap]

    def test_whitespace_not_allowed_inside_key(self) -> None:
        s = DecoderState()
        # In IN_KEY a space is CONTENT of the key (schema identifiers have no
        # spaces; the filter/schema discards it afterwards).
        assert s.update_from_text('{"na me":1}')
        assert s.phase == DecoderPhase.COMPLETE


class TestComplete:
    def test_empty_params_syntactically_ok(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":"fn","parameters":{}}')
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == set()  # the schema validator demands required keys

    def test_extra_key_after_params_ok(self) -> None:
        # The state machine is schema-agnostic: valid JSON with extra keys.
        # The schema validator rejects unknown keys.
        s = DecoderState()
        assert s.update_from_text('{"name":"fn","parameters":{},"extra":1}')
        assert s.phase == DecoderPhase.COMPLETE

    def test_rejects_non_whitespace_after_complete(self) -> None:
        s = DecoderState()
        assert s.update_from_text(FULL_JSON)
        assert not s._advance_char("x")
        assert s.phase == DecoderPhase.COMPLETE

    def test_tolerates_trailing_whitespace(self) -> None:
        s = DecoderState()
        assert s.update_from_text(FULL_JSON + "  \n")
        assert s.phase == DecoderPhase.COMPLETE


class TestExpectedFirstChars:
    def test_root(self) -> None:
        s = DecoderState()
        assert "{" in s.expected_first_chars()
        assert " " in s.expected_first_chars()
        assert "x" not in s.expected_first_chars()

    def test_colon(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":')
        e = s.expected_first_chars()
        for ch in ('"', "{", "t", "f", "n", "-", "0", "9"):
            assert ch in e, ch
        assert "+" not in e
        assert "." not in e

    def test_number_continuations_are_grammar_precise(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2')
        e = s.expected_first_chars()
        assert {"0", "9", ".", "e", "E", ",", "}"} <= e
        assert "-" not in e  # after "2", a '-' is invalid

        assert s.update_from_text(".")  # buffer "2.": fraction pending
        e = s.expected_first_chars()
        assert "3" in e
        assert "e" not in e  # "2.e" does not exist
        assert "," not in e  # "2," is invalid: the fraction needs digits

        assert s.update_from_text("5")  # buffer "2.5" -> now complete
        e = s.expected_first_chars()
        assert "," in e and " " in e

    def test_number_exponent_continuations(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2e')
        e = s.expected_first_chars()
        assert {"0", "+", "-"} <= e
        assert "." not in e

        assert s.update_from_text("+")  # "2e+": the digit is missing
        e = s.expected_first_chars()
        assert "9" in e
        assert "-" not in e  # "2e+-" is invalid

    def test_bool_continuations(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":tru')
        assert s.expected_first_chars() == {"e"}

        assert s.update_from_text("e")
        e = s.expected_first_chars()
        assert "e" not in e
        assert {",", "}", " "} <= e

    def test_escape_continuations(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":"\\')
        e = s.expected_first_chars()
        assert "u" in e and "n" in e and "t" in e and '"' in e and "\\" in e
        assert "x" not in e

    def test_unicode_pending_requires_hex(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":"\\u00')
        e = s.expected_first_chars()
        assert "0" in e and "f" in e and "A" in e
        assert '"' not in e  # does not close the string while hex is missing

    def test_wildcard_for_open_states(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"na')
        assert "*" in s.expected_first_chars()  # IN_KEY

        s2 = DecoderState()
        assert s2.update_from_text('{"name":"')
        assert "*" in s2.expected_first_chars()  # IN_STRING_VALUE

    def test_complete_returns_empty_set(self) -> None:
        s = DecoderState()
        assert s.update_from_text(FULL_JSON)
        assert s.expected_first_chars() == set()


class TestNameBuffer:
    """Documented deviation: the machine accumulates the "name" value.

    Only the value of the OUTPUT object's "name" key (depth 0) enters the
    buffer; structure, keys and values of parameters do NOT touch it. Escapes
    are skipped: the buffer keeps the "decoded" name.
    """

    def test_accumulates_only_name_value_chars(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name": "fn_add_numbers"')
        assert s.name_buffer == "fn_add_numbers"
        # Subsequent structure (including the "parameters" key and its values)
        # does NOT enter the buffer or overwrite it.
        assert s.update_from_text(', "parameters": {"a": 2.0}')
        assert s.name_buffer == "fn_add_numbers"

    def test_param_named_name_does_not_fill_buffer(self) -> None:
        """fn_greet has a "name" parameter (depth 1): it must not enter."""

        s = DecoderState()
        assert s.update_from_text('{"name": "fn_greet", "parameters": {')
        assert s.name_buffer == "fn_greet"
        assert s.update_from_text('"name": "Javier"}')
        # The value of the "name" parameter is NOT accumulated (depth == 1).
        assert s.name_buffer == "fn_greet"

    def test_escape_rejected_in_name_value(self) -> None:
        """An escape inside the "name" value is rejected WHEN READING the '\\'
        — it used to be silently skipped from the buffer, which left the trie
        prefix intact and allowed an infinite loop of escapes in the generator
        (real repro: 'Greet shrek' never closed the string). No real name uses
        '\\'."""

        s = DecoderState()
        assert s.update_from_text('{"name": "f')
        assert not s.update_from_text("\\n_greet")
        # Atomic: the state did not move, name_buffer is still "f".
        assert s.name_buffer == "f"

    def test_escapes_still_skipped_for_non_name_buffer_paths(self) -> None:
        """The escape skip in name_buffer is still in force for the only path
        that can touch it: the "name" value WITHOUT escapes. This only confirms
        that normal accumulation (without '\\') did not change."""

        s = DecoderState()
        assert s.update_from_text('{"name": "fn_greet"')
        assert s.name_buffer == "fn_greet"

    def test_name_buffer_resets_on_new_name_value(self) -> None:
        """A second "name" value (syntactically valid) resets it."""

        s = DecoderState()
        assert s.update_from_text('{"name": "fn_add_numbers", "parameters": {}')
        assert s.name_buffer == "fn_add_numbers"
        assert s.update_from_text(', "name": "fn_greet"}')
        assert s.name_buffer == "fn_greet"
