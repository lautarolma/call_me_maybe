"""Tests del token filter (Task 3.4): compute_allowed_ids.

VOLUNTAD DE ESTOS TESTS (por dentro):
- No testean el filter aislado: caminan el pipeline COMPLETO como lo hará el
  generator en Task 4.1. Por cada token: compute_allowed_ids(state, schema,
  vocab, trie) → verificar membership del id objetivo → commit del token
  (state.update_from_text → schema.update). El orden state → schema es el
  del plan.
- El vocab mock replica la forma de un vocabulario BPE real: hay tokens que
  mezclan estructura y contenido ('fn_add_numbers", "parameters": {',
  ', "b": 3', '"a": 2.0'). Esos cruces de fase son SITUACIÓN NORMAL en BPE y
  son justo los que ejercitan los triggers internos del schema (cambio de
  current_key, entrada a value a mitad de token).
- El bucket <byte> (id 60) NO tiene entrada en id2decoded, igual que en
  vocab_loader.py (BYTE_CATEGORY). Los ids 62/63 decodifican a U+FFFD y a un
  surrogate: _is_clean_utf8 debe descartarlos en Fase 2.
- Aserciones de MEMBERSHIP, no de sets exactos (salvo ROOT): el vocab deja
  varios caminos válidos en cada estado; over-specify rompería en cuanto
  alguien agregue un token.
"""

from __future__ import annotations

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
from src.decoder.token_filter import compute_allowed_ids, iter_ranked_allowed_ids
from src.decoder.trie import build_trie
from src.loader.vocab_loader import BYTE_CATEGORY, Vocab
from src.models.function_definition import FunctionDef, ParameterDef

# Replica de data/input/functions_definition.json (subset usado en tests).
FUNCTIONS = [
    FunctionDef(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": ParameterDef(type="number"),
            "b": ParameterDef(type="number"),
        },
        returns={"type": "number"},
    ),
    FunctionDef(
        name="fn_greet",
        description="Greet someone by name.",
        parameters={
            "name": ParameterDef(type="string"),
        },
        returns={"type": "string"},
    ),
    FunctionDef(
        name="fn_empty",
        description="Function without parameters.",
        parameters={},
        returns={"type": "null"},
    ),
]

# Vocab mock: id -> texto DECODIFICADO (lo que ve la state machine).
VOCAB: dict[int, str] = {
    # Estructura / literales
    1: "{",
    2: " ",
    3: "}",
    4: ",",
    5: ":",
    6: '"',
    # Keys
    10: "name",
    11: "parameters",
    12: "name2",       # key de output NO definida en el schema (gap depth 0)
    13: "na",          # clave partida a la mitad (subword real)
    14: '"name"',      # key completa con comillas
    # Nombres de función (subwords y completos)
    20: "fn_",
    21: "add",
    22: "_numbers",
    23: "greet",
    24: "empty",
    25: "fn_g",        # prefijo de fn_greet
    26: "x",           # char que no arranca ningún nombre
    27: "fn_add_numbers",
    28: "fn_greet",
    # Tokens mixtos estructura+contenido (el pan de cada día del BPE)
    30: '": "',
    31: '": 2',
    32: ', "b": 3',    # key "b" leída A MITAD de token (arranca en el value de a)
    33: '"a": 2.0',    # key "a" + value en un solo token
    34: '"b": 4',
    35: ', "name": "Javier"',
    36: '"name": "Javier"',
    # Keys de parameters
    40: '"a"',
    41: '"b"',
    42: '"zzz"',       # key inexistente en el schema
    43: '"a": 4',      # duplicado EXACTO de "a" (mismo texto que el commiteado)
    # Values
    50: '"x"',
    51: "2.0",
    52: '"Javier"',
    53: "true",
    54: "null",
    55: '"a": "x"',    # value string para un parámetro number
    56: "n",
    # Tokens sucios / especiales
    60: "<byte>",      # SOLO en el bucket BYTE_CATEGORY (no entra a id2decoded)
    61: " the",        # 'Ġthe' decodificado: empieza con espacio
    62: "\ufffd",      # decode de bytes UTF-8 inválidos
    63: "\udc80",      # surrogate: no es scalar value
    # Tokens key+estructura
    70: '"name": ',          # key "name" + colon + ws (termina en COLON)
    71: '"parameters": {',   # abre parameters desde OBJECT_OPEN
    72: ', "parameters": {',  # cierra el name, abre parameters (un token)
}

IDS: dict[str, int] = {text: tid for tid, text in VOCAB.items()}


BYTE_IDS = frozenset({60})  # tokens que SOLO van en BYTE_CATEGORY


def build_vocab() -> Vocab:
    """Construye el Vocab mock imitando el indexado de vocab_loader.py.

    Los tokens byte (BYTE_IDS) NO entran a id2decoded ni a buckets de
    caracteres: en el vocab real son bytes crudos que fallan al decodificar.
    Solo existen en el bucket BYTE_CATEGORY, que el wildcard de
    compute_allowed_ids excluye.
    """
    id2decoded: dict[int, str] = {}
    starting: dict[str, set[int]] = {}
    for tid, text in VOCAB.items():
        if tid not in BYTE_IDS:
            id2decoded[tid] = text
            starting.setdefault(text[0], set()).add(tid)
    starting.setdefault(BYTE_CATEGORY, set()).add(60)
    return Vocab(
        token2id={text: tid for tid, text in VOCAB.items()},
        id2token={tid: text for tid, text in VOCAB.items()},
        id2decoded=id2decoded,
        tokens_starting_with=starting,
        vocab_size=len(VOCAB) + 1,
    )


def make_pipeline() -> tuple[DecoderState, SchemaContext, Vocab, object]:
    vocab = build_vocab()
    trie = build_trie([fn.name for fn in FUNCTIONS])
    return DecoderState(), SchemaContext(FUNCTIONS), vocab, trie


def step(state: DecoderState, schema: SchemaContext, text: str) -> None:
    """Avanza el estado con un token y sincroniza el schema (contrato Task 4.1)."""
    assert state.update_from_text(text), f"state machine rejected {text!r}"
    schema.update(state)


def allowed(
    state: DecoderState, schema: SchemaContext, vocab: Vocab, trie: object
) -> set[int]:
    return compute_allowed_ids(state, schema, vocab, trie)


class TestRootAcceptance:
    """Acceptance criteria 1: en ROOT solo tokens que arrancan con '{'."""

    def test_root_allowed_set_is_exact(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        # Candidatos: bucket '{' (id 1) + buckets de ws (2 y 61 ' the').
        # 61 falla en simulate ('t' no es válido en ROOT); los sucios y el
        # bucket <byte> ni son candidatos. Nada más permitido.
        assert allowed(state, schema, vocab, trie) == {1, 2}

    def test_output_tokens_rejected_at_root(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        a = allowed(state, schema, vocab, trie)
        assert IDS["name"] not in a      # empieza con 'n' (no esperado)
        assert IDS['"'] not in a         # '"' solo tras el '{'

    def test_dirty_and_special_tokens_never_allowed(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        a = allowed(state, schema, vocab, trie)
        assert 60 not in a
        assert 62 not in a
        assert 63 not in a
        assert IDS[" the"] not in a      # simulate lo rechaza en ROOT


class TestRankedIterator:
    def test_top_k_is_ranked_and_restricted_to_phase_candidates(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        logits = [0.0] * (max(VOCAB) + 1)
        logits[2] = 10.0
        logits[1] = 5.0

        ranked = list(
            iter_ranked_allowed_ids(
                state, schema, vocab, trie, logits, top_k=2
            )
        )

        assert [token_id for token_id, _ in ranked[:2]] == [2, 1]

    def test_fallback_searches_candidates_outside_top_k(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')
        logits = [0.0] * (max(VOCAB) + 1)
        logits[IDS["x"]] = 10.0
        logits[IDS["fn_"]] = 1.0

        ranked = list(
            iter_ranked_allowed_ids(
                state, schema, vocab, trie, logits, top_k=1
            )
        )

        assert ranked[0] == (IDS["fn_"], "fn_")


class TestWildcardAndDirtyTokens:
    """expected_first_chars '*' (key/string abiertos) toma TODOS los buckets."""

    def test_dirty_tokens_skipped_while_key_open(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"')
        step(state, schema, "na")  # IN_KEY, current_key "na"
        a = allowed(state, schema, vocab, trie)
        # Wildcard activo: U+FFFD y surrogate son CANDIDATOS pero mueren en
        # _is_clean_utf8; el <byte> nunca estuvo en id2decoded.
        assert 62 not in a
        assert 63 not in a
        assert 60 not in a
        assert IDS['"'] in a  # cierra la key "na" -> KEY_END (depth 0, ok)


class TestNameTrie:
    """Acceptance criteria 2: el value de "name" se restringe al trie."""

    def test_only_trie_prefixes_while_name_open(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')  # IN_STRING_VALUE, name_buffer ""
        a = allowed(state, schema, vocab, trie)
        assert IDS["fn_"] in a          # prefijo real
        assert IDS["fn_g"] in a         # prefijo real (fn_greet)
        assert IDS["add"] not in a      # no es prefijo (falta fn_)
        assert IDS["greet"] not in a
        assert IDS["x"] not in a
        assert IDS["name"] not in a
        assert IDS['"'] not in a        # no se puede cerrar un name vacío

    def test_partial_name_cannot_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')
        step(state, schema, "fn_")  # buffer "fn_": prefijo, no nombre
        assert IDS['"'] not in allowed(state, schema, vocab, trie)

    def test_complete_name_can_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"name": ')
        step(state, schema, '"')
        step(state, schema, "fn_add_numbers")
        assert IDS['"'] in allowed(state, schema, vocab, trie)

    def test_name_built_across_subwords_closes_valid(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_", "add", "_numbers"):
            step(state, schema, t)
        assert IDS['"'] in allowed(state, schema, vocab, trie)

    def test_output_key_not_schema_validated(self) -> None:
        """Gap documentado: keys del OUTPUT object (depth 0) no se validan."""
        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"')  # KEY_START
        # "name2" no existe en el schema, pero es una key del output object:
        # depth 0 queda fuera del scope (el trie solo entra con key "name").
        assert IDS["name2"] in allowed(state, schema, vocab, trie)


class TestParamKeys:
    def _at_params(self) -> tuple[DecoderState, SchemaContext, Vocab, object]:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        return state, schema, vocab, trie

    def test_known_and_unknown_keys(self) -> None:
        state, schema, vocab, trie = self._at_params()
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a"'] in a
        assert IDS['"b"'] in a
        assert IDS['"'] in a             # abre una key: prefijo vacío, ok
        assert IDS['"zzz"'] not in a     # key inexistente en el schema
        assert IDS['"a": 2.0'] in a      # key + value en un solo token
        assert IDS["}"] not in a         # faltan "a" y "b" (cláusula 4)

    def test_mid_token_key_read_is_validated(self) -> None:
        """EL caso del trigger por cambio: ', "b": 3' lee la key a mitad de
        token (arranca DENTRO del value de "a") y termina en IN_NUMBER_VALUE,
        fuera de _KEY_PHASES. Un trigger por fase no lo vería jamás."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')  # IN_NUMBER_VALUE, key "a" (abierta)
        a = allowed(state, schema, vocab, trie)
        # La key "b" se lee dentro de este token: válida (change detection) y
        # su value type también se chequea (post en IN_NUMBER_VALUE -> number).
        assert IDS[', "b": 3'] in a

    def test_identical_rebuild_slip_is_documented(self) -> None:
        """Limitación conocida (documentada en _allows_param_key): un
        duplicado EXACTO reconstruye current_key con el MISMO texto que el
        commiteado ("a" -> reset "" -> "a"): el trigger por cambio no puede
        distinguirlo y la key pasa. Se bloquean los divergentes (crecer de su
        texto NO empieza como prefix), pero el idéntico escapa."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")    # cierra "a" -> PARAMS_OBJECT
        # current_key commiteado == "a" (stale), keys_enclosed == {"a"}
        assert IDS['"a": 4'] in allowed(state, schema, vocab, trie)  # slip

    def test_reopened_key_diverging_from_closed_is_blocked(self) -> None:
        """Contraste del slip: una key que DIVERGE del texto commiteado
        (multi-token) sí queda bloqueada por el prefix check."""

        state, schema, vocab, trie = self._at_params()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")    # PARAMS_OBJECT, key commiteada "a"
        step(state, schema, '"')    # abre una key nueva (KEY_START, "")
        a = allowed(state, schema, vocab, trie)
        # "na" no puede completarse hacia ninguna key disponible ({b}):
        assert IDS["na"] not in a


class TestValueTypes:
    def test_number_param_constrains_value_entry(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
            '"a"',
            ":",
        ):
            step(state, schema, t)  # COLON con current_key "a" (depth 1)
        a = allowed(state, schema, vocab, trie)
        assert IDS["2.0"] in a           # number == number
        assert IDS['"'] not in a         # abre string para un number
        assert IDS["true"] not in a      # boolean != number
        assert IDS["null"] not in a      # null != number

    def test_string_param_accepts_string_and_rejects_number(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_greet",
            '"',
            ', "parameters": {',
            '"name": ',  # COLON key "name" (depth 1: el PARÁMETRO)
        ):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS['"'] in a             # abre string para un string
        assert IDS['"x"'] in a           # contenido string: ok (sin trie acá)
        assert IDS["2.0"] not in a       # number != string

    def test_full_key_string_value_in_one_token_type_checked(self) -> None:
        """Casos que la cláusula 3 SÍ detecta y el gap documentado (B8).

        - '33' ('"a": 2.0') desde PARAMS_OBJECT: termina en IN_NUMBER_VALUE
          (∈ _VALUE_PHASES) → el tipo number coincide con a:number → pasa.
        - '55' ('"a": "x"') desde PARAMS_OBJECT: abre key, value string y lo
          cierra TODO en el mismo token → post en VALUE_END (∉ _VALUE_PHASES)
          y el arranque no fue COLON: la cláusula 3 NO gatilla. Es el gap
          documentado en _allows_value_type (key+value completos por token =
          parser paramétrico, B8). Acá se pinea como comportamiento esperado
          del MVP, no como bug."""

        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a": 2.0'] in a    # type check SÍ dispara (post en fase)
        assert IDS['"a": "x"'] in a    # gap documentado (B8) — no bloqueado


class TestParamsClose:
    def test_close_blocked_until_all_required(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
        ):
            step(state, schema, t)
        assert IDS["}"] not in allowed(state, schema, vocab, trie)
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        assert IDS["}"] in allowed(state, schema, vocab, trie)

    def test_close_with_value_in_same_token(self) -> None:
        """El cierre puede venir junto al value final ('"b": 3}') y el value
        recién cerrado cuenta para el ⊆ gracias a keys_enclosed SIMULADO."""

        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
            '"a": 2.0',
            ", ",
            '"b": 3',
        ):
            step(state, schema, t)
        step(state, schema, "}")  # cierra params: {a,b} ⊂ new_state.keys_enclosed
        assert state.phase.name == "VALUE_END"
        assert state.depth == 0
        step(state, schema, "}")  # cierre del output object
        assert state.phase.name == "COMPLETE"


class TestFnEmpty:
    def test_empty_params_close_immediately(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_empty", '"', ', "parameters": {'):
            step(state, schema, t)
        a = allowed(state, schema, vocab, trie)
        assert IDS["}"] in a            # required vacío: cierra ya
        assert IDS['"a"'] not in a      # ninguna key es válida (available ∅)


class TestNameFirstEnforcement:
    def test_parameters_before_name_are_blocked(self) -> None:
        """"parameters" ANTES del name: sin función seleccionada, ni las keys
        ni el '}' pasan. El generador queda atascado en depth 1: la única
        salida es generar el name primero (refuerzo deliberado)."""

        state, schema, vocab, trie = make_pipeline()
        step(state, schema, "{")
        step(state, schema, '"parameters": {')
        a = allowed(state, schema, vocab, trie)
        assert IDS['"a"'] not in a
        assert IDS["}"] not in a


class TestFnGreetWalk:
    def test_full_walk_with_close(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in (
            "{",
            '"name": ',
            '"',
            "fn_greet",
            '"',
            ', "parameters": {',
            '"name": ',
            '"Javier"',   # abre+cierra el string en un solo token BPE
        ):
            step(state, schema, t)
        # value del parámetro cerrado -> keys_enclosed {name} -> cierre ok
        assert IDS["}"] in allowed(state, schema, vocab, trie)


class TestFullGenerationWalk:
    def test_fn_add_numbers_end_to_end(self) -> None:
        """El acceptance criteria del plan en vivo: cada token propuesto debe
        estar en allowed_ids ANTES de commitearse (orden del generator)."""

        state, schema, vocab, trie = make_pipeline()
        sequence = [
            "{",
            '"name": ',
            '"',
            "fn_add_numbers",
            '"',
            ', "parameters": {',
            '"a": 2.0',
            ', "b": 3',
            "}",
            "}",
        ]
        for i, text in enumerate(sequence):
            assert IDS[text] in allowed(state, schema, vocab, trie), (
                f"step {i}: {text!r} no permitido"
            )
            step(state, schema, text)
        assert state.phase.name == "COMPLETE"
        assert state.keys_enclosed == {"a", "b"}
