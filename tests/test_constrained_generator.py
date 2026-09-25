"""Tests del constrained generator (Task 4.1) + pase fino (Inciso 4.1.1).

VOLUNTAD DE ESTOS TESTS (por dentro):
- Testean el generator con un FakeModel que avanza una secuencia fija de
  tokens (logits altos para el siguiente token de la secuencia, bajos para
  el resto): el argmax elige el target cuando está en allowed.
- La MITAD de los tests es del PASO FINO (Inciso 4.1.1): los gaps que el
  filter (Task 3.4) deja pasar por diseño (abstención por estados límite)
  DEBEN ser rechazados por la re-simulación char-por-char del ganador:
    * gap 2 residual 2: token que entra a parameters + primera key en un
      token ('", "parameters": {"a') — key válida pasa, key inválida se
      bloquea (antes la inválida entraba de contrabando).
    * gap 3: key+value+cierre completos en un token (', "b": "x",' con
      "b": number) — el pase fino lo bloquea; el filter lo permitía (B8).
    * gap 4: entrar Y salir de parameters en un token — sin todos los
      required se bloquea; con todos, pasa.
    * slip del duplicado exacto ('"a": 4' con "a" ya emitida) — bloqueado.
    * gap del plan: output COMPLETE sin haber pasado por PARAMS_OBJECT
      ('}' tras el name sólamente) — bloqueado.
- Cada token del vocab mock existe SOLO para el caso que ejercita; el
  vocabulario real de Qwen tiene miles de tokens así (los de ~20-30 chars
  son raros en BPE, por eso el filter los tolera y el pase fino los cubre
  con costo de 1 re-simulación por step).
"""

from __future__ import annotations

from src.decoder.constrained_generator import (
    _inject_static_header,
    _next_static_text,
    _passes_fine_validation,
    generate,
)
from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.trie import build_trie
from src.loader.vocab_loader import (
    BYTE_CATEGORY,
    Vocab,
    _STATIC_PHASE_FIRST_CHARS,
)
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
    7: "\n  ",  # newline + indent 2 (formato natural del tramo Opt2)
    # Keys del output object
    10: "name",
    11: "parameters",
    # Nombres de función
    20: "fn_",
    27: "fn_add_numbers",
    28: "fn_greet",
    24: "empty",
    29: "fn_empty",
    # Tokens mixtos estructura+contenido
    30: '": "',
    32: ', "b": 3',
    33: '"a": 2.0',
    # Keys de parameters
    40: '"a"',
    41: '"b"',
    43: '"a": 4',              # duplicado EXACTO de "a" (slip documentado)
    # Values
    50: '"x"',
    51: "2.0",
    52: '"Javier"',
    53: "true",
    # Keys + estructura
    70: '"name": ',
    71: '"parameters": {',
    72: ', "parameters": {',
    # Casos del PASO FINO (Inciso 4.1.1) — tokens multi-fase de ~19-28 chars
    73: ', "parameters": {"a',       # entra a params + 1ra key "a" (válida)
    74: ', "parameters": {"zz',      # ídem con key INEXISTENTE (gap 2)
    75: ', "b": "x",',               # key+value string+cierre para b:number (gap 3)
    76: ', "b": 4,',                 # ídem con type correcto → pasa el pase fino
    77: ', "parameters": {"a": 1}',   # entra Y sale de params, falta "b" (gap 4)
    78: ', "parameters": {"a": 1, "b": 2}',  # ídem con todos los required → pasa
    79: ", ",
    # Tramos del ORÁCULO (Fase 2): canónicos + value standalone del E2E
    80: "\n  }\n}",          # T5: cierre de parameters + cierre del ROOT
    81: "3",                 # value de "b" en el E2E (T4 ya emitió '"b": ')
    82: "\n}",               # T6: cierre del ROOT (fn_empty, formato INLINE {})
}

IDS: dict[str, int] = {text: tid for tid, text in VOCAB.items()}

BYTE_IDS = frozenset()  # este mock no modela tokens <byte>


def build_vocab() -> Vocab:
    """Construye el Vocab mock imitando el indexado de vocab_loader.py."""
    id2decoded: dict[int, str] = {}
    starting: dict[str, set[int]] = {}
    for tid, text in VOCAB.items():
        if tid not in BYTE_IDS:
            id2decoded[tid] = text
            starting.setdefault(text[0], set()).add(tid)
    starting.setdefault(BYTE_CATEGORY, set()).add(60)
    valid_by_phase = {
        phase: {
            tid
            for ch in phase_chars
            for tid in starting.get(ch, set())
        }
        for phase, phase_chars in _STATIC_PHASE_FIRST_CHARS.items()
    }
    return Vocab(
        token2id={text: tid for text, tid in VOCAB.items()},
        id2token={tid: text for tid, text in VOCAB.items()},
        id2decoded=id2decoded,
        tokens_starting_with=starting,
        vocab_size=max(VOCAB) + 1,  # ids arbitrarios en el mock: cubrir el máximo
        valid_by_phase=valid_by_phase,
    )


def make_pipeline() -> tuple[DecoderState, SchemaContext, Vocab, object]:
    vocab = build_vocab()
    trie = build_trie([fn.name for fn in FUNCTIONS])
    return DecoderState(), SchemaContext(FUNCTIONS), vocab, trie


def step(state: DecoderState, schema: SchemaContext, text: str) -> None:
    """Avanza el estado con un token y sincroniza el schema (contrato Task 4.1)."""
    assert state.update_from_text(text), f"state machine rejected {text!r}"
    schema.update(state)


def at_params(
    fn_name: str = "fn_add_numbers",
) -> tuple[DecoderState, SchemaContext, Vocab, object]:
    """Estado en VALUE_END tras el name (depth 0, función seleccionada)."""
    state, schema, vocab, trie = make_pipeline()
    for t in ("{", '"name": ', '"', fn_name, '"'):
        step(state, schema, t)
    return state, schema, vocab, trie


def set_params_ctx() -> tuple[DecoderState, SchemaContext, Vocab, object]:
    """Estado PARAMS_OBJECT con selected_function (fn_add_numbers) y keys ∅."""
    state, schema, vocab, trie = at_params("fn_add_numbers")
    step(state, schema, ', "parameters": {')
    return state, schema, vocab, trie


class TestFineValidationGap2EnterParamsWithFirstKey:
    """Inciso 4.1.1: token que entra a parameters + lee la 1ra key (gap 2)."""

    def test_valid_first_key_passes(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # "a" entra a parameters en el MISMO token que abre el objeto: el
        # pase fino la valida (antes el filter la dejaba de contrabando).
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "parameters": {"a')

    def test_invalid_first_key_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # "zz" no es prefijo de ninguna key disponible: el prefix check del
        # paso por carácter lo bloquea (gap 2 residual 2 CERRADO).
        assert not _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"zz'
        )


class TestFineValidationGap3CompleteKeyValue:
    """Inciso 4.1.1: key+value+cierre completos en un token (gap 3)."""

    def test_wrong_type_blocked(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')  # IN_NUMBER_VALUE, "a" abierta
        # ', "b": "x",' abre un string para b:number: el COLON intermedio
        # expone el tipo y el '"' que abre el string se bloquea.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "b": "x",')

    def test_right_type_passes(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, ', "b": 4,')


class TestFineValidationGap4EnterAndExitParams:
    """Inciso 4.1.1: entrar Y salir de parameters en un token (gap 4)."""

    def test_missing_required_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        # El objeto parameters completo en un token (arrancando desde VALUE_END
        # exige la coma: desde acá solo ',' o '}' son válidos), sin "b": el '}'
        # de cierre (depth 1→0 intermedio) dispara la cláusula 4 → falta "b".
        assert not _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"a": 1}'
        )

    def test_all_required_passes(self) -> None:
        state, schema, vocab, trie = at_params("fn_add_numbers")
        assert _passes_fine_validation(
            FUNCTIONS, schema, state, trie, ', "parameters": {"a": 1, "b": 2}'
        )


class TestFineValidationDuplicateSlip:
    """Inciso 4.1.1: el slip del duplicado exacto se cierra en el pase fino."""

    def test_identical_duplicate_key_blocked(self) -> None:
        state, schema, vocab, trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")  # cierra "a" → PARAMS_OBJECT
        # El filter lo permitía (slip documentado); el paso por carácter ve
        # el reset ""→"a" → available ya no contiene "a" → bloqueado.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, '"a": 4')


class TestFineValidationMissingParamsObject:
    """Inciso 4.1.1: COMPLETE sin haber pasado por PARAMS_OBJECT."""

    def test_close_without_params_object_blocked(self) -> None:
        state, schema, vocab, trie = at_params("fn_empty")
        # '}' cierra el output object directo: nunca vimos '{' de parameters.
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, "}")

    def test_full_empty_params_close_passes(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "fn_empty", '"', ', "parameters": {'):
            step(state, schema, t)
        step(state, schema, "}")  # cierra parameters vacío (depth 1→0)
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, "}")  # cierra output


class TestFineValidationNameEscapeRejected:
    """BUG-011 (2026-09-23): un escape dentro del value de "name" no debe
    poder colarse infinitamente. name_buffer no lo toca (se skippea en
    state.py), así que sin el guard explícito en _allows_name_value el
    trie sigue viendo un prefijo válido y el generador nunca cierra el
    string (repro real: 'Greet shrek' → 200 forwards en '\\n\\t\\t\\t...'
    sin completar)."""

    def test_escape_inside_name_value_blocked(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "f"):
            step(state, schema, t)
        assert not _passes_fine_validation(FUNCTIONS, schema, state, trie, "\\n")

    def test_plain_continuation_of_name_still_passes(self) -> None:
        state, schema, vocab, trie = make_pipeline()
        for t in ("{", '"name": ', '"', "f"):
            step(state, schema, t)
        assert _passes_fine_validation(FUNCTIONS, schema, state, trie, "n_greet")


class _FakeTensor:
    """Dummy que REPLICA la forma 2D [1, N] del tensor real de encode().

    BUG-005: Small_LLM_Model.encode() arma torch.tensor([ids]) → 2D; su
    .tolist() devuelve list[list[int]]. Antes este dummy devolvía una lista
    PLANA (1D) y el bug de dimensiones pasaba desapercibido en la suite.
    """

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __getitem__(self, idx: int) -> _FakeRow:
        # t[0] de un tensor 2D [1, N] → vista 1D [N]
        return _FakeRow(self._ids)

    def tolist(self) -> list[list[int]]:
        return [list(self._ids)]


class _FakeRow:
    """Vista 1D de una fila de tensor (t[0].tolist() → list[int] plano)."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return list(self._ids)


class FakeModel:
    """Mock de Small_LLM_Model: empuja una secuencia fija de tokens.

    encode(prompt) devuelve un único id dummy (prompt_length = 1); cada
    get_logits_from_input_ids asigna logit alto al siguiente token de la
    secuencia esperada y -100 al resto. Así el argmax elige el target SOLO
    si el target está en allowed (de lo contrario elige otro token y la
    generación se desvía — el test lo detecta).
    """

    def __init__(self, vocab: Vocab, sequence: list[str]) -> None:
        self._vocab = vocab
        self._sequence = sequence

    def encode(self, text: str) -> _FakeTensor:
        return _FakeTensor([999])  # prompt dummy de 1 id

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab.id2decoded[tid] for tid in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        # BUG-005: contrato de FORMAS con el SDK real — get_logits espera
        # list[int] PLANO. Si generate() dejara de aplanar ([0].tolist()), acá
        # entraría list[list[int]] y este assert tiñe la suite de rojo.
        assert all(isinstance(x, int) for x in input_ids), (
            "get_logits_from_input_ids debe recibir list[int] plano, "
            f"no {type(input_ids[0]).__name__}"
        )
        step = len(input_ids) - 1  # prompt_length == N del encode
        logits = [-100.0] * self._vocab.vocab_size
        if step < len(self._sequence):
            logits[IDS[self._sequence[step]]] = 100.0
        return logits


class TestGenerator:
    def test_fn_add_numbers_end_to_end(self) -> None:
        """Acceptance criteria: prompt → JSON con "fn_add_numbers", éxito."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = FakeModel(
            vocab,
            [
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
            ],
        )
        generated, ok = generate(model, "What is 2+3?", vocab, FUNCTIONS, trie)
        assert ok
        assert "fn_add_numbers" in generated

    def test_zero_max_tokens_fails_cleanly(self) -> None:
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = FakeModel(vocab, ["{"])
        generated, ok = generate(model, "hi", vocab, FUNCTIONS, trie, max_tokens=0)
        assert not ok
        assert generated == ""

    def test_garbage_target_is_replaced_by_second_best(self) -> None:
        """Si el target del modelo NO está permitido, se genera igual un
        token válido (argmax sobre allowed) o se corta — nunca JSON roto."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        # Target "fn_" en ROOT: no permitido (solo '{' y ws) → se elige '{'.
        model = FakeModel(vocab, ["fn_"])
        generated, ok = generate(model, "boo", vocab, FUNCTIONS, trie, max_tokens=3)
        assert not ok  # sin COMPLETE en 3 tokens
        assert generated.startswith("{")  # el token ganador fue el '{'

    def test_n_token_prompt_does_not_leak_into_generated(self) -> None:
        """BUG-005 (regresión): el prompt de N tokens NO se cuela en el output.

        encode() devuelve tensor 2D con ids que NO existen en el vocab del
        test (999/777/555). Si generated_ids incluyera tokens del prompt (por
        prompt_length mal calculado), FakeModel.decode haría KeyError — fallo
        ruidoso. Con el fix, prompt_length == 3 y generated solo tiene '{'.
        """
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])

        class _PromptfulModel(FakeModel):
            """encode() de 3 tokens — ninguno existe en VOCAB (KeyError si
            alguno llegara a generated_ids)."""

            def encode(self, text: str) -> _FakeTensor:  # noqa: D102
                return _FakeTensor([999, 777, 555])

        generated, ok = generate(
            _PromptfulModel(vocab, ["{"]),
            "prompt largo", vocab, FUNCTIONS, trie, max_tokens=1,
        )
        assert not ok  # solo 1 token generado: sin COMPLETE
        # El único token generado fue '{' (id 1): nada de prompt en el output.
        assert generated == "{"


# ─── Fase 2: oráculo por estado (tramos estáticos Nivel 1, Anexo 25/09) ──
# _next_static_text(state, schema, emitted) recorre _TRAMPS (T1-T6, funciones
# PURAS) y devuelve el canónico del primer dominio que matchea, alineado al
# ws final de emitted (E). Estos tests verifican:
#   * el canónico de CADA tramo (tabla Nivel 1 del Anexo);
#   * los dominios DISJUNTOS (N gate verificado contra state.py) — BUG-013;
#   * el alineamiento E (clase BUG-012: nunca duplicar ws);
#   * el estado post-inyección con state.simulate(C) como fuente de verdad
#     (dec. 25/09: NO tablas escritas a mano — desync con state.py es bug).

T1_CANON_ADD = ',\n  "parameters": {\n    "a": '
T1_CANON_GREET = ',\n  "parameters": {\n    "name": "'
T2_CANON_ADD = '\n  "parameters": {\n    "a": '
T3_CANON_FIRST = '\n    "a": '
T3_CANON_NEXT = '\n    "b": '
T4_CANON_NEXT = ',\n    "b": '
T5_CANON = "\n  }\n}"
T6_CANON = "\n}"

# Canónicos viejos (tramo lineal de Opt2) — SOLO pueden venir del oráculo.
_OLD_TAIL = ',\n  "parameters": {'

# id2decoded del vocab mock (split elegido para el test; el encode real de
# Qwen produce su propio split y el best-effort lo maneja igual).
_ORACLE_IDS: dict[str, list[int]] = {
    T1_CANON_ADD: [4, 7, 71, 7, 2, 2, 40, 5, 2],
    T3_CANON_NEXT: [7, 2, 2, 41, 5, 2],
    T4_CANON_NEXT: [4, 7, 2, 2, 41, 5, 2],
    T5_CANON: [80],
    T6_CANON: [82],
}


class TestLevel1Oracle:
    """Tabla Nivel 1: cada tramo responde SU canónico desde su dominio."""

    def test_t1_post_name_value_end(self) -> None:
        # Nominal: value de "name" cerrado → VALUE_END d0, N=1 → la coma +
        # apertura de parameters + la PRIMERA key (fusión T1+ENTRY(K1)).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _next_static_text(state, schema, "") == T1_CANON_ADD

    def test_t1_string_param_opens_with_quote(self) -> None:
        # fn_greet: su ÚNICO param es "name" de tipo string → OP(k)='"'.
        # (El param se llama igual que la key del output object: N=1 sigue
        # valiendo porque es el VALUE_END d0 — el param vive a depth 1.)
        state, schema, _vocab, _trie = at_params("fn_greet")
        assert _next_static_text(state, schema, "") == T1_CANON_GREET

    def test_t2_fused_comma_token(self) -> None:
        # Token BPE fusionado ('fn_add_numbers",') → IN_OBJECT directo (sin
        # pasar por VALUE_END): T2 cubre el salto (N=1 se mantiene).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        step(state, schema, ",")
        assert state.phase is DecoderPhase.IN_OBJECT
        assert _next_static_text(state, schema, "") == T2_CANON_ADD

    def test_empty_params_return_none(self) -> None:
        # DECISIÓN 25/09 (tramo gratis): ord=∅ → T1/T2 devuelven None — el
        # modelo genera "parameters": {} con su token FUSIONADO (tid 6257);
        # inyectar '{' suelto sería una costura tipo BUG-012. fn_empty es
        # el único caso del subject; el probe real confirmó el formato
        # INLINE {}.
        state, schema, _vocab, _trie = at_params("fn_empty")
        assert _next_static_text(state, schema, "") is None

    def test_t3_first_required_key(self) -> None:
        # PARAMS_OBJECT d1 con keys pendientes → ENTRY(Knext).
        state, schema, _vocab, _trie = set_params_ctx()
        assert _next_static_text(state, schema, "") == T3_CANON_FIRST

    def test_t3_after_comma(self) -> None:
        # El modelo emitió la coma tras "a" → PARAMS_OBJECT otra vez (en este
        # decoder VALUE_END + ',' → PARAMS_OBJECT, NO IN_OBJECT) → T3 da la
        # SIGUIENTE key requerida. Este es el flujo de los values NUMBER:
        # IN_NUMBER_VALUE (número abierto) → la coma del modelo → T3.
        state, schema, _vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ",")
        assert state.phase is DecoderPhase.PARAMS_OBJECT
        assert _next_static_text(state, schema, "") == T3_CANON_NEXT

    def test_t4_next_key_after_string_value(self) -> None:
        # VALUE_END d1 con pendientes → coma + próxima key. Solo un value
        # que CIERRA en su token llega a VALUE_END d1 (string); un number
        # deja IN_NUMBER_VALUE y el flujo retoma en T3 tras la coma.
        fn = FunctionDef(
            name="fn_text",
            description="",
            parameters={
                "x": ParameterDef(type="string"),
                "y": ParameterDef(type="string"),
            },
            returns={"type": "string"},
        )
        state, _schema, _vocab, _trie = make_pipeline()
        schema = SchemaContext([fn])
        for t in ("{", '"name": ', '"', "fn_text", '"', ', "parameters": {'):
            step(state, schema, t)
        step(state, schema, '"x": "Javier"')  # cierra string → VALUE_END d1
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _next_static_text(state, schema, "") == ',\n    "y": "'

    def test_t5_close_after_last_string_value(self) -> None:
        # VALUE_END d1 sin pendientes → cierre de parameters + del ROOT.
        # fn_greet: su ÚNICO param es string → el cierre lo da T5.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _next_static_text(state, schema, "") == T5_CANON

    def test_t6_close_root_after_empty_params(self) -> None:
        # fn_empty: tras el '}' del parameters vacío (formato INLINE del
        # probe real) el ROOT queda por cerrar. Sin T6 el modelo debe
        # emitir ese '}' por forward (probe: SUCCESS=False). T6 lo da.
        state, schema, _vocab, _trie = at_params("fn_empty")
        step(state, schema, ', "parameters": {')
        step(state, schema, "}")
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _next_static_text(state, schema, "") == T6_CANON

    def test_t6_close_root_after_full_params(self) -> None:
        # Post-parameters con todas las keys: VALUE_END d0, N=0 (current_key
        # es la última key de params, no "name"), ρ=0, P=1 → cierre del ROOT.
        state, schema, _vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        step(state, schema, "}")  # cierra parameters → d0
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert _next_static_text(state, schema, "") == T6_CANON

    def test_t6_blocked_while_name_open(self) -> None:
        # N=1 (post-name, parameters aún sin abrir) → T6 None: el ROOT NO
        # se cierra sin parameters (el pase fino lo bloquea si el modelo lo
        # intentara por forward). fn_empty en VALUE_END post-name: T1 ya
        # dio None (ord=∅) y T6 exige N=0 → None total → sigue por forward.
        state, schema, _vocab, _trie = at_params("fn_empty")
        assert _next_static_text(state, schema, "") is None

    def test_t6_fused_empty_params_token(self) -> None:
        # HALLazgo del probe real (25/09): el token FUSIONADO ', "parameters":
        # {}' trae '{'+'}' en UN tocho → has_seen_params_object() queda False
        # (el schema.update corre POST-token y nunca ve el PARAMS_OBJECT
        # intermedio). P NO puede gatear T6 — con N=0 ∧ ρ=0 el ROOT solo
        # puede cerrarse (el pase fino del camino normal cubre el flanco).
        state, schema, _vocab, _trie = at_params("fn_empty")
        step(state, schema, ', "parameters": {}')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 0
        assert not schema.has_seen_params_object()  # el flag NO vio el '{'
        assert _next_static_text(state, schema, "") == T6_CANON

    def test_bug013_inner_param_name_not_an_opener(self) -> None:
        # BUG-013 (regresión): el param interno "name" de fn_greet (depth 1)
        # NO dispara T1/T2 (los tramos de APERTURA exigen N=1: depth==0 ∧
        # keys_enclosed==∅ — al cerrar el param, keys_enclosed={'name'}). Y
        # como es el ÚNICO param, ρ=0 → T5 sí responde (el objeto queda
        # COMPLETO y hay que cerrarlo): el oráculo NUNCA duplica el tramo
        # de apertura — esa era la clase de bug del trigger viejo.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')
        assert state.phase is DecoderPhase.VALUE_END and state.depth == 1
        assert _next_static_text(state, schema, "") == T5_CANON

    def test_initial_root_does_not_match(self) -> None:
        state, schema, _vocab, _trie = make_pipeline()
        assert _next_static_text(state, schema, "") is None

    def test_insertion_order_of_parameters(self) -> None:
        # ord = tuple(F.parameters): el orden JSON es el del dict (insertion
        # order), NO alphabetical — "b" declarada primero sale primero.
        fn = FunctionDef(
            name="fn_ordered",
            description="",
            parameters={
                "b": ParameterDef(type="number"),
                "a": ParameterDef(type="number"),
            },
            returns={"type": "number"},
        )
        state, _schema, _vocab, _trie = make_pipeline()
        schema = SchemaContext([fn])
        for t in ("{", '"name": ', '"', "fn_ordered", '"'):
            step(state, schema, t)
        assert _next_static_text(state, schema, "") == ',\n  "parameters": {\n    "b": '

    def test_align_trims_emitted_ws(self) -> None:
        # E: si emitted ya termina en el ws del canónico (el modelo lo puso),
        # el tramo NO lo duplica (clase BUG-012) → inyecta solo el resto.
        # T5 con emitted='\n  ' → e='\n  ' → resta '}\n}'. El 1er char de
        # emitted es parte del canon T5 → el modelo YA está donde el tramo.
        state, schema, _vocab, _trie = at_params("fn_greet")
        step(state, schema, ', "parameters": {')
        step(state, schema, '"name": ')
        step(state, schema, '"x"')  # único param (string): ρ=0 → VALUE_END d1
        assert _next_static_text(state, schema, "\n  ") == "}\n}"

    def test_t2_aligns_fused_comma_indent(self) -> None:
        # El token fusionado '",\n  ' (coma + indent del modelo en UN token)
        # deja IN_OBJECT d0 N=1 con emitted=',\n  ' → T2 alinea y corta el
        # '\n  ' del canónico: inyecta SOLO '"parameters": {\n    "a": '.
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        step(state, schema, ",")  # IN_OBJECT d0 (el ',' no resetea la key)
        assert state.phase is DecoderPhase.IN_OBJECT and state.depth == 0
        assert _next_static_text(state, schema, ",\n  ") == (
            '"parameters": {\n    "a": '
        )

    def test_align_unmatchable_ws_falls_to_forward(self) -> None:
        # emitted termina en ws que NO es prefijo del canónico (p.ej. el
        # modelo ya emitió la indent de IN_OBJECT y viene la coma del T1):
        # alineamiento imposible → None → forward normal (seguro, sin
        # forzar tramos — Riesgo 2 cubierto por diseño).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert _next_static_text(state, schema, "\n  ") is None


class TestOracleDomainDisjunction:
    """Los dominios de los tramos son disjuntos por construcción (phase ×
    depth × gates): para un estado dado, UN solo tramo responde. Estos pares
    son los que compartían fase y solo se distinguen por los gates."""

    def test_t1_vs_t6_same_phase_different_gate(self) -> None:
        # VALUE_END d0: T1 responde SOLO con N=1 (post-name, params por
        # abrir); T6 SOLO con N=0 ∧ ρ=0 ∧ P=1 (post-parameters).
        state, schema, _vocab, _trie = at_params("fn_add_numbers")
        assert _next_static_text(state, schema, "") == T1_CANON_ADD
        state2, schema2, _v2, _t2 = set_params_ctx()
        step(state2, schema2, '"a": 2.0')
        step(state2, schema2, ', "b": 3')
        step(state2, schema2, "}")
        assert _next_static_text(state2, schema2, "") == T6_CANON

    def test_t3_vs_t6_pending_keys_gate(self) -> None:
        # PARAMS_OBJECT d1 es T3 (ρ=1); con ρ=0 no matchea T3 → si el cierre
        # viene por el modelo, T6 lo complementa solo en d0.
        state, schema, _vocab, _trie = set_params_ctx()
        assert _next_static_text(state, schema, "") == T3_CANON_FIRST


class TailFakeModel(FakeModel):
    """FakeModel cuyo encode() tokeniza SOLO los canónicos del oráculo.

    El prompt y el STATIC_HEADER (header 1) devuelven el id dummy 999 como el
    FakeModel base — 999 no existe en el id2decoded del vocab mock, así que
    el header 1 nunca se inyecta (mismo comportamiento que los tests
    existentes). Los canónicos _ORACLE_IDS se devuelven con ids del vocab
    mock para poder inyectarlos por el mismo camino que en producción.
    """

    def __init__(self, vocab: Vocab, sequence: list[str]) -> None:
        super().__init__(vocab, sequence)
        self.calls = 0

    def encode(self, text: str) -> _FakeTensor:
        if text in _ORACLE_IDS:
            return _FakeTensor(list(_ORACLE_IDS[text]))
        return _FakeTensor([999])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.calls += 1
        return super().get_logits_from_input_ids(input_ids)


class TestInjectOracleText:
    """_inject_static_header con un canónico del oráculo: el estado post
    debe coincidir EXACTO con state.simulate(C) (fuente de verdad — dec.
    25/09; una tabla escrita a mano desincronizada con state.py sería un
    bug silencioso)."""

    def _state_tuple(self, s: DecoderState) -> tuple[object, object, object, object]:
        return (s.phase, s.depth, s.current_key, s.keys_enclosed)

    def test_t1_injection_advances_to_colon(self) -> None:
        state, schema, vocab, _trie = at_params("fn_add_numbers")
        model = TailFakeModel(vocab, [])
        ids: list[int] = []
        emitted: list[str] = []
        assert _inject_static_header(
            T1_CANON_ADD, model, vocab, ids, state, schema, emitted
        )
        assert ids == _ORACLE_IDS[T1_CANON_ADD]
        _ok, sim = state.simulate(T1_CANON_ADD)  # fuente de verdad
        assert self._state_tuple(state) == self._state_tuple(sim)
        # COLON d1 esperando el value numérico de "a" — el canónico terminó
        # en ' ' (la apertura del value de un number NO lleva comilla).
        assert state.phase is DecoderPhase.COLON and state.depth == 1
        assert "".join(emitted) == T1_CANON_ADD

    def test_t5_injection_closes_both_objects(self) -> None:
        state, schema, vocab, _trie = set_params_ctx()
        step(state, schema, '"a": 2.0')
        step(state, schema, ', "b": 3')
        model = TailFakeModel(vocab, [])
        ids: list[int] = []
        assert _inject_static_header(T5_CANON, model, vocab, ids, state, schema)
        _ok, sim = state.simulate(T5_CANON)
        assert self._state_tuple(state) == self._state_tuple(sim)
        assert state.phase is DecoderPhase.COMPLETE

    def test_inject_returns_false_without_advance(self) -> None:
        # encode() solo entiende canónicos del oráculo: un texto fuera del
        # registro → id 999 no existe → la inyección no avanza → False (el
        # caller cae al camino normal; sin este contrato, un continue ciego
        # tras la falla re-matchearía el mismo tramo → loop infinito).
        state, schema, vocab, _trie = at_params("fn_add_numbers")
        model = FakeModel(vocab, [])
        ids: list[int] = []
        before = (state.phase, state.depth, state.current_key)
        assert not _inject_static_header(
            _OLD_TAIL, model, vocab, ids, state, schema
        )
        assert not ids
        assert (state.phase, state.depth, state.current_key) == before


class TestOracleEndToEnd:
    """Oráculo end-to-end: los canónicos se inyectan sin forwards.

    La secuencia del modelo NO contiene los chars de los canónicos ('\n
    "parameters": {\n    "a": ' no existe como texto en ninguna entrada del
    vocab): si aparecen en el output, SOLO pudieron venir de la inyección.
    Los placeholders (" ") ocupan los índices que cada canónico consume sin
    forward (el FakeModel indexa el step por len(input_ids)). Cada test fija
    el conteo EXACTO de forwards: una regresión (tramo que no matchea, que
    duplica ws, o que se cae al forward) desvía la secuencia o cambia calls.
    """

    def test_fn_add_numbers_full_oracle(self) -> None:
        """T1 + T3 + T6 consumidos; el modelo paga 9 forwards (5 name +
        '2.0', ',', '3', '}'). El flujo number real: '2.0' deja
        IN_NUMBER_VALUE (número abierto) → el modelo paga la coma → T3 da
        el ws + '"b": ' → '3' abierto → el modelo paga el '}' de cierre de
        parameters → T6 da el '\n}' del ROOT."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        # 5 name + 9 placeholders (T1: 9 ids) + '2.0' + ',' + 6
        # placeholders (T3: 6 ids) + '3' + '}' (T6 cierra sin forward).
        model = TailFakeModel(
            vocab,
            [
                "{", '"name": ', '"', "fn_add_numbers", '"',
                *([" "] * 9),
                "2.0", ",",
                *([" "] * 6),
                "3", "}",
            ],
        )
        generated, ok = generate(model, "What is 2+3?", vocab, FUNCTIONS, trie)
        assert ok
        assert T1_CANON_ADD in generated
        assert T3_CANON_NEXT in generated  # ',\n    "b": ' en la salida real
        assert generated.endswith(T6_CANON)  # el '\n}' de cierre del ROOT
        assert model.calls == 9

    def test_fn_empty_gets_root_close(self) -> None:
        """fn_empty: ord=∅ → T1 None → el modelo emite ',
        "parameters": {}' por forward; T6 da el '\n}' que faltaba (probe
        real: SUCCESS=False por ese cierre)."""
        vocab = build_vocab()
        trie = build_trie([fn.name for fn in FUNCTIONS])
        model = TailFakeModel(
            vocab,
            [
                "{", '"name": ', '"', "fn_empty", '"',
                ', "parameters": {', "}",
            ],
        )
        generated, ok = generate(model, "hi", vocab, FUNCTIONS, trie)
        assert ok
        # Formato INLINE real de Qwen para parameters vacío ('{"' directo,
        # probe_fn_empty 24/09): T6 aporta el '\n}' que cerró el ROOT.
        assert generated.endswith('"fn_empty", "parameters": {}\n}')
        assert model.calls == 7  # 5 name + '{'+'}' de parameters (T6 gratis)
