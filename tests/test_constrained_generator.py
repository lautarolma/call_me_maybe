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

from src.decoder.constrained_generator import _passes_fine_validation, generate
from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderState
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
    return Vocab(
        token2id={text: tid for tid, text in VOCAB.items()},
        id2token={tid: text for tid, text in VOCAB.items()},
        id2decoded=id2decoded,
        tokens_starting_with=starting,
        vocab_size=max(VOCAB) + 1,  # ids arbitrarios en el mock: cubrir el máximo
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


class _FakeTensor:
    """Dummy con .tolist() — evita depender de torch en los tests."""

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
        step = len(input_ids) - 1  # prompt_length == 1
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
