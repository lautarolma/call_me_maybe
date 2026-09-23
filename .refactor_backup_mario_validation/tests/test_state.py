"""Unit tests for the constrained JSON decoder state machine (Task 3.1)."""

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
        # Sin "parameters": sintácticamente válido; el schema validator
        # (Task 3.3) lo rechazará porque parameters es required.
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
        assert ns is s  # spec A6.4: se retorna el estado ORIGINAL al fallar
        assert s.phase == DecoderPhase.ROOT

    def test_multi_char_token_whitespace_prefix(self) -> None:
        # Los tokens BPE de Qwen decodifican "Ġx" como " x": un token puede
        # arrancar con espacio Y contener '{' — no hay estado intermedio.
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
        assert not s.update_from_text('"fn"}x')  # "x" tras COMPLETE
        assert s.phase == before

    def test_split_across_calls(self) -> None:
        # Los tokens cortan keys y values a la mitad: cada update_from_text
        # retoma desde el estado acumulado del anterior.
        s = DecoderState()
        assert s.update_from_text('{"na')
        assert s.phase == DecoderPhase.IN_KEY
        assert s.update_from_text('me":"fn"}')
        assert s.phase == DecoderPhase.COMPLETE

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
        '{"name":"fn","parameters":{"a":2.}}',       # fracción sin dígitos
        '{"name":"fn","parameters":{"a":2e}}',       # exponente sin dígitos
        '{"name":"fn","parameters":{"a":2e+}}',
        '{"name":"fn","parameters":{"a":2e+-3}}',
        '{"name":"fn","parameters":{"a":2..5}}',
        '{"name":"fn","parameters":{"a":.5}}',       # int part obligatorio
        '{"name":"fn","parameters":{"a":+2}}',       # '+' inicial inválido
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
        assert not s._advance_char("1")  # "01" no es número JSON


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
        # Solo 3 hex antes del '"': el '"' NO es hex -> inválido.
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
        # ws tras el number cierra el value (VALUE_END) y deja que el '}'
        # real cierre el objeto: dos transiciones, no una.
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2 }')
        assert s.phase == DecoderPhase.VALUE_END
        assert s.keys_enclosed == {"a"}
        assert s.update_from_text("}")
        assert s.phase == DecoderPhase.COMPLETE

    def test_whitespace_not_allowed_inside_key(self) -> None:
        s = DecoderState()
        # En IN_KEY el espacio es CONTENIDO de la key (identificadores del
        # schema no tienen espacios; el filter/schema lo descarta después).
        assert s.update_from_text('{"na me":1}')
        assert s.phase == DecoderPhase.COMPLETE


class TestComplete:
    def test_empty_params_syntactically_ok(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name":"fn","parameters":{}}')
        assert s.phase == DecoderPhase.COMPLETE
        assert s.keys_enclosed == set()  # schema validator exige required keys

    def test_extra_key_after_params_ok(self) -> None:
        # La state machine es schema-agnóstica: JSON válido con más keys.
        # El schema validator (Task 3.3) rechaza keys desconocidas.
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
        assert "-" not in e  # tras "2", un '-' es inválido

        assert s.update_from_text(".")  # buffer "2.": fracción pendiente
        e = s.expected_first_chars()
        assert "3" in e
        assert "e" not in e  # "2.e" no existe
        assert "," not in e  # "2," es inválido: la fracción exige dígitos

        assert s.update_from_text("5")  # buffer "2.5" -> now complete
        e = s.expected_first_chars()
        assert "," in e and " " in e

    def test_number_exponent_continuations(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"parameters":{"a":2e')
        e = s.expected_first_chars()
        assert {"0", "+", "-"} <= e
        assert "." not in e

        assert s.update_from_text("+")  # "2e+": falta el dígito
        e = s.expected_first_chars()
        assert "9" in e
        assert "-" not in e  # "2e+-" es inválido

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
        assert '"' not in e  # no cierra el string mientras falten hex

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
    """⚠ Desvío documentado (Task 3.3): la máquina acumula el value de "name".

    Solamente el value de la key "name" del OUTPUT object (depth 0) entra al
    buffer; estructura, keys y values de parameters NO lo tocan. Los escapes
    se skippean: el buffer queda con el nombre "decodificado".
    """

    def test_accumulates_only_name_value_chars(self) -> None:
        s = DecoderState()
        assert s.update_from_text('{"name": "fn_add_numbers"')
        assert s.name_buffer == "fn_add_numbers"
        # Estructura posterior (incluida la key "parameters" y sus values)
        # NO entra al buffer ni lo pisa.
        assert s.update_from_text(', "parameters": {"a": 2.0}')
        assert s.name_buffer == "fn_add_numbers"

    def test_param_named_name_does_not_fill_buffer(self) -> None:
        """fn_greet tiene un parámetro "name" (depth 1): no debe entrar."""

        s = DecoderState()
        assert s.update_from_text('{"name": "fn_greet", "parameters": {')
        assert s.name_buffer == "fn_greet"
        assert s.update_from_text('"name": "Javier"}')
        # El value del parámetro "name" NO se acumula (depth == 1).
        assert s.name_buffer == "fn_greet"

    def test_escapes_skipped_in_buffer(self) -> None:
        """Escapes simples (\\n) y \\uXXXX se skippean: el buffer queda con
        el nombre "decodificado" sin los caracteres de escape."""

        s = DecoderState()
        # \n se consume como escape: f + (skip \n) + _greet = "f_greet"
        assert s.update_from_text('{"name": "f\\n_greet"')
        assert s.name_buffer == "f_greet"

    def test_name_buffer_resets_on_new_name_value(self) -> None:
        """Un segundo value de "name" (sintácticamente válido) resetea."""

        s = DecoderState()
        assert s.update_from_text('{"name": "fn_add_numbers", "parameters": {}')
        assert s.name_buffer == "fn_add_numbers"
        assert s.update_from_text(', "name": "fn_greet"}')
        assert s.name_buffer == "fn_greet"
