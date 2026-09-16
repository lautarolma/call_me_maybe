"""Schema-aware validation del constrained JSON decoder (Task 3.3).

POR QUÉ EXISTE ESTE MÓDULO (por dentro):
- La state machine (state.py) valida SINTAXIS: "cualquier string bien formado".
  El schema validator valida SEMÁNTICA: qué keys son válidas, qué tipo de
  value espera cada una, y qué required keys faltan antes de cerrar
  parameters. Es la otra mitad de la distinción sintaxis ↔ semántica
  documentada en TeoricNotes.
- Es la contracara del trie: el trie restringe el value de "name" a nombres
  de función existentes (Task 3.4, Fase 3). SchemaContext TOMA ese nombre y
  resuelve la función seleccionada; a partir de ahí valida los parámetros
  (keys, tipos, required).

DE DÓNDE SALE EL NOMBRE SELECCIONADO (decisión de diseño):
- update(state) recibe SOLO el estado (firma EXACTA del plan, sin
  token_text). El texto del value de "name" lo acumula la STATE MACHINE en
  state.name_buffer (⚠ desvío documentado en state.py, mismo espíritu que
  keys_enclosed: bookkeeping que el schema lee).
- ¿Por qué en la state machine y no acá? Un token BPE puede mezclar
  estructura y contenido ('fn_add_numbers", "parameters": {'): cuando la
  resolución debería dispararse, el estado post-token ya tiene otra key en
  current_key. Reconstruir el span del name desde el estado post-token
  obligaría a re-simular el token. La state machine ve los chars en
  contexto: es la única fuente confiable y encima simplifica Task 3.4
  (allows_token puede usar new_state.name_buffer con el trie directo).

QUÉ NO HACE (separación de concerns):
- No toca el trie (trie.py) ni el pre-filtro (Task 3.4). Solo conoce la
  función seleccionada, sus parámetros (keys + tipos) y las keys ya emitidas.

NOTA sobre el acceptance criteria del plan ("state en VALUE_START"):
- VALUE_START es el estado INALCANZABLE (el plan A6.3 lo reservaba; la
  implementación saltea directo al estado de value concreto). Los tests usan
  las fases reales: COLON / IN_STRING_VALUE / IN_NUMBER_VALUE / etc.
"""

from __future__ import annotations

from src.decoder.state import DecoderPhase, DecoderState
from src.models.function_definition import FunctionDef

# Fases en las que se está LEYENDO un value (o a punto de arrancarlo en COLON).
# En estas fases current_key ya se definió y el tipo esperado aplica.
_VALUE_READ_PHASES = (
    DecoderPhase.COLON,
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.IN_NUMBER_VALUE,
    DecoderPhase.IN_BOOL_VALUE,
    DecoderPhase.IN_NULL_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)


class SchemaContext:
    """Estado semántico del decoder: función seleccionada + contexto de params.

    NO es un @dataclass a propósito: este objeto se actualiza UNA vez por
    step de generación (no se copia por candidato como DecoderState), así
    que no necesita __eq__/__repr__ generados ni default factories. Un
    __slots__ manual alcanza y deja el contrato explícito.
    """

    __slots__ = (
        "_index",
        "_current_key",
        "_depth",
        "_keys_enclosed",
        "_phase",
        "selected_function",
    )

    def __init__(self, functions: list[FunctionDef]) -> None:
        # Índice name -> FunctionDef: el loader ya rechaza duplicados
        # (BUG-002), acá la construcción es directa.
        self._index = {fn.name: fn for fn in functions}
        self._current_key = ""
        self._depth = 0
        self._keys_enclosed: set[str] = set()
        self._phase = DecoderPhase.ROOT
        #: Función elegida por el value de "name" (None hasta resolverse).
        self.selected_function: FunctionDef | None = None

    # ------------------------------------------------------------------ API

    def update(self, state: DecoderState) -> None:
        """Refresca el contexto desde el estado commiteado del decoder.

        SE LLAMA UNA vez por step (no por candidato): el filter NO muta el
        schema en Fase 3, solo lee. La mutación vive únicamente acá.
        """
        self._phase = state.phase
        self._current_key = state.current_key
        # keys_enclosed del estado ya viene como set nuevo reemplazado
        # (nunca mutado in-place) — copiar es gratis y evita aliasing raro.
        self._keys_enclosed = set(state.keys_enclosed)
        self._depth = state.depth
        self._resolve_function(state)

    def current_expected_type(self) -> str | None:
        """Tipo esperado para el value en curso, o None si no aplica.

        - Key "name" del output object (depth 0) → siempre "string".
        - Parámetro de la función seleccionada (depth 1) → su tipo en el
          schema.
        - Cualquier otra cosa ("parameters" object, key desconocida,
          función aún no seleccionada, o posición donde no se lee un
          value) → None (sin constraint).
        """
        if self._phase not in _VALUE_READ_PHASES:
            return None
        if self._depth == 0:
            # El value de "name" es string; "parameters" es un objeto sin
            # tipo escalar (lo validan las cláusulas de estructura, no el tipo).
            return "string" if self._current_key == "name" else None
        # depth == 1: value de un parámetro de la función seleccionada.
        if self.selected_function is None:
            return None
        param = self.selected_function.parameters.get(self._current_key)
        return param.type if param is not None else None

    def required_keys_remaining(self) -> set[str]:
        """Required keys de parameters que todavía no se emitieron.

        En este MVP TODOS los parámetros son required (el JSON de entrada no
        distingue required/opcional). Como las keys ya emitidas NO pueden
        repetirse, este mismo set es el de "keys válidas para el próximo
        key": el filter (Task 3.4) lo usa para bloquear keys inexistentes o
        duplicadas.
        """
        if self.selected_function is None:
            return set()
        return set(self.selected_function.parameters) - self._keys_enclosed

    def all_required_present(self) -> bool:
        """True si TODOS los required keys ya fueron emitidos.

        Si no hay función seleccionada devuelve False (conservador): no se
        puede afirmar que un parameters object puede cerrarse sin saber qué
        keys exige.
        """
        return self.selected_function is not None and not self.required_keys_remaining()

    def can_close_params(self) -> bool:
        """True si el '}' de cierre de parameters está permitido ahora."""
        return self.all_required_present()

    # ------------------------------------------------------ name resolution

    def _resolve_function(self, state: DecoderState) -> None:
        """Resuelve selected_function cuando el name del output object está.

        Regla por BUFFER, no por fase: cualquier name_buffer no vacío es el
        value de "name" del output object (la state machine solo acumula en
        depth 0 con key "name"; el parámetro "name" de fn_greet vive en
        depth 1 y no entra). El buffer sobrevive al cierre del name aunque
        el MISMO token siga con estructura ("parameters"...): por eso la
        resolución no depende de dónde quedó el estado post-token.
        - Si el buffer es un PREFIJO parcial (fn_get_s), _index.get da None
          y se espera al próximo step (el trie ya garantiza prefix-validity).
        - Si es un nombre COMPLETO aunque el string no se haya cerrado aún,
          resolver temprano es correcto: el trie no permite extender un
          nombre completo (valid_next_chars == set()).
        """
        if self.selected_function is not None:
            return
        if not state.name_buffer:
            return
        self.selected_function = self._index.get(state.name_buffer)
