"""Máquina de estados del constrained JSON decoder (Task 3.1).

POR QUÉ EXISTE ESTE MÓDULO (por dentro):
- El modelo NO genera JSON libre: en cada step hay que validar que el token
  propuesto mantenga el output como JSON sintácticamente válido. Esta state
  machine es el "árbitro sintáctico": dado el estado actual, decide si cada
  carácter (y por ende el token completo) es legal.
- Es la capa MÁS caliente del pipeline: compute_allowed_ids (Task 3.4) llama
  a simulate() por cada token candidato (~151K ids en el peor caso). Por eso
  es un @dataclass(slots=True) y NO Pydantic: Pydantic cuesta ~200μs por
  instanciación; un dataclass con slots no tiene __dict__ y se copia en ~50ns.
- NO conoce el schema (qué keys existen, qué tipos se esperan): eso es
  trabajo de schema_validator.py (Task 3.3). Acá solo se garantiza SINTAXIS
  JSON sobre el subconjunto del subject: objeto con keys, values string /
  number / bool / null, y UN nivel de anidamiento (parameters).
  Única concesión: name_buffer acumula el TEXT del value de "name"
  (desvío Task 3.3) — bookkeeping que el schema lee, no validación.

CONTRATO DE USO (dos caminos):
- simulate(token_text) -> (bool, DecoderState): EXPLORA sin tocar el estado
  real. El token filter la llama para cada candidato.
- update_from_text(token_text) -> bool: AVANZA el estado real (muta) con el
  token GANADOR. Es atómico: si algún carácter falla, el estado queda tal
  cual estaba (se simula sobre una copia y se commitea solo si todo pasó).
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass, field
from enum import Enum

# Whitespace JSON: space, tab, newline, carriage return. Nada más.
_WS = " \t\n\r"
_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"
# Escapess simples de JSON: \" \\ \/ \n \t \r \b \f (el \uXXXX va aparte
# porque consume 4 dígitos hex y puede partirse entre tokens).
_SIMPLE_ESCAPES = frozenset('"\\/nrtbf')

# Grammar de number JSON:  -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
#   _NUMBER_PREFIX_RE: versión "a medio terminar" (acepta "2." y "2e+")
#   para la validación incremental char por char. El signo inicial "-" se
#   incorpora en _step_colon() y debe ser seguido por un dígito. Los *
#   permiten cero o más caracteres en las partes que pueden quedar pendientes.
#   Rechaza leading zeros ("01") porque la primera alternativa
#   solo tolera "0" SOLO, y la segunda no puede arrancar con cero.
#   OJO la alternancia de la parte decimal: "2.e" DEBE fallar (un punto sin
#   dígitos deja la fracción pendiente: el exponente solo es legal DESPUÉS
#   de al menos un dígito). Por eso: fracción con dígitos + exponente
#   opcional | punto con dígitos pendientes (sin exponente) | exponente.
_NUMBER_PREFIX_RE = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]+([eE][+-]?[0-9]*)?|\.[0-9]*|[eE][+-]?[0-9]*)?"
)
#   _NUMBER_RE: versión estricta para decidir si el buffer TIENE un número
#   completo al momento de cerrar el value ("," / "}" / whitespace).
_NUMBER_RE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")


class DecoderPhase(str, Enum):
    """Fases de la state machine del output JSON.

    Elegido como `str, Enum`: cada miembro ES su nombre ("ROOT" ==
    DecoderPhase.ROOT), lo que da repr legibles en logs y comparaciones
    directas contra strings sin castear.
    """

    ROOT = "ROOT"                    # Estado inicial: esperando '{'
    OBJECT_OPEN = "OBJECT_OPEN"      # '{' leído: esperando el primer '"'
    IN_OBJECT = "IN_OBJECT"          # En el output object: key o '}'
    KEY_START = "KEY_START"          # '"' de apertura de key leído
    IN_KEY = "IN_KEY"                # Acumulando caracteres de la key
    KEY_END = "KEY_END"              # '"' de cierre de key: esperando ':'
    COLON = "COLON"                  # ':' leído: esperando el value
    VALUE_START = "VALUE_START"      # (reservado por el plan A6.3; no se usa)
    IN_STRING_VALUE = "IN_STRING_VALUE"
    IN_NUMBER_VALUE = "IN_NUMBER_VALUE"
    IN_BOOL_VALUE = "IN_BOOL_VALUE"  # true / false
    IN_NULL_VALUE = "IN_NULL_VALUE"  # null
    ESCAPE_IN_STRING = "ESCAPE_IN_STRING"
    VALUE_END = "VALUE_END"          # Value cerrado: ',' o '}'
    PARAMS_OBJECT = "PARAMS_OBJECT"  # Dentro del objeto parameters (depth 1)
    COMPLETE = "COMPLETE"            # '}' final: generación detenida


@dataclass(slots=True)
class DecoderState:
    """Estado mutable del decoder. UNO por generación (no por token).

    POR QUÉ slots=True y no Pydantic (Decisión 8 del plan): el inner loop
    instancia/copia este objeto por cada token candidato; __slots__ elimina
    __dict__ (menos memoria, acceso más rápido) y copy.copy() ronda los ~50ns
    contra los ~5μs de un deepcopy — viable porque keys_enclosed NUNCA se
    muta in-place: siempre se reemplaza por un set nuevo (ver
    _register_params_key).
    """

    phase: DecoderPhase = DecoderPhase.ROOT
    current_key: str = ""              # Key cuyo value se está leyendo
    keys_enclosed: set[str] = field(default_factory=set)  # Solo keys de parameters
    depth: int = 0                     # 0 = output object, 1 = parameters
    number_buffer: str = ""            # Acumula el number en curso
    # ⚠ DESVÍO DOCUMENTADO (Task 3.3): la máquina acumula el TEXT del value
    # de la key "name" (depth 0) para que SchemaContext resuelva la función
    # seleccionada. Mismo espíritu que keys_enclosed: bookkeeping que el
    # schema LEE, no validación sintáctica. Los escapes se SKIPPEAN: el
    # buffer queda con el nombre "decodificado" (\u0066n_... → fn_...).
    # ¿Por qué acá y no en el schema? Un token BPE puede mezclar estructura
    # y contenido ('fn_add_numbers", "parameters": {'); reconstruir el span
    # del name desde el estado post-token obligaría a re-simular el token.
    # El único lugar que ve los chars en contexto es la state machine.
    name_buffer: str = ""              # Text del value de "name" (depth 0)
    bool_buffer: str = ""              # Acumula true/false/null en curso
    unicode_remaining: int = 0         # Hex pendientes de un \uXXXX en curso

    # ------------------------------------------------------------------ API

    def simulate(self, token_text: str) -> tuple[bool, DecoderState]:
        """Simula procesar UN token completo SIN mutar este estado.

        CÓMO FUNCIONA (por dentro):
        - Se trabaja sobre copy(self): shallow copy barata (slots, sets
          reemplazados nunca mutados) y segura.
        - Si CUALQUIER carácter falla, la generación del token es inválida:
          se retorna (False, self) — el MISMO objeto original (spec A6.4).
        - Devuelve el estado resultante para que el generator (Task 4.1)
          pueda usarlo directamente sin re-simular el token ganador.
        """
        new_state = copy(self)
        for char in token_text:
            if not new_state._advance_char(char):
                return False, self
        return True, new_state

    def update_from_text(self, text: str) -> bool:
        """Avanza ESTE estado con un texto completo (token ganador). Atómico.

        CÓMO FUNCIONA (por dentro):
        - Igual que simulate (copiar + avanzar char por char), pero al final
          COMMITEA los campos de la copia en self. Si algo falla a mitad de
          camino, self queda exactamente como antes — el generator nunca se
          queda con un estado a medio token.
        - Los campos se copian explícitamente (no __dict__.update) porque
          slots=True no tiene __dict__; además es mypy-friendly.
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
        """Chars con los que PUEDE arrancar el próximo token (Fase 1 del filter).

        CÓMO SE CONSUME (por dentro):
        - Es la llave de entrada al pre-índice Vocab.tokens_starting_with
          (primer carácter DECODIFICADO -> ids). Fase 1 de compute_allowed_ids
          junta los buckets de todos estos chars.
        - '*' es un comodín: significa "cualquier carácter real es posible"
          (keys y strings libres). El filter lo interpreta como "saltarse el
          pre-filtro" (Task 3.4). Los tokens <byte> nunca matchean estos
          chars, quedan fuera de la generación.
        - Para numbers devuelve EXACTAMENTE los chars que mantienen la
          grammar: "2." solo admite dígitos, "2e" admite dígitos/+-/terminal.
        """
        phase = self.phase
        if phase is DecoderPhase.ROOT:
            # El output SIEMPRE arranca con '{' (más ws posible adelante).
            return {"{", *(_WS)}
        if phase is DecoderPhase.OBJECT_OPEN:
            return {'"', *(_WS)}
        if phase is DecoderPhase.IN_OBJECT:
            return {'"', "}", *(_WS)}
        if phase is DecoderPhase.KEY_START or phase is DecoderPhase.IN_KEY:
            return {"*"}  # cualquier carácter puede iniciar/continuar una key
        if phase is DecoderPhase.KEY_END:
            return {":", *(_WS)}
        if phase is DecoderPhase.COLON:
            return {'"', "-", "{", "t", "f", "n", *(_DIGITS), *(_WS)}
        if phase is DecoderPhase.IN_STRING_VALUE:
            if self.unicode_remaining > 0:
                return set(_HEX_DIGITS)  # siguiente(s) char(s) de \uXXXX
            return {"*"}
        if phase is DecoderPhase.IN_NUMBER_VALUE:
            # Terminales solo si el buffer ya es un number COMPLETO: con
            # "2." (fracción pendiente) un ',' NO puede cerrar el value.
            if self._is_valid_json_number():
                return self._number_next_chars() | {",", "}", *(_WS)}
            return self._number_next_chars()
        if phase is DecoderPhase.IN_BOOL_VALUE or phase is DecoderPhase.IN_NULL_VALUE:
            target = self._literal_target()
            if self.bool_buffer == target:
                return {",", "}", *(_WS)}  # literal completo: terminales
            return {target[len(self.bool_buffer)]}  # el próximo char exacto
        if phase is DecoderPhase.ESCAPE_IN_STRING:
            return {*_SIMPLE_ESCAPES, "u"}
        if phase is DecoderPhase.VALUE_END:
            return {",", "}", *(_WS)}
        if phase is DecoderPhase.PARAMS_OBJECT:
            return {'"', "}", *(_WS)}
        # COMPLETE (o estado inalcanzable): nada es válido.
        return set()

    # ------------------------------------------------------- char transition

    def _advance_char(self, char: str) -> bool:
        """Procesa UN carácter y mueve la state machine. False = inválido.

        CÓMO FUNCIONA (por dentro):
        - Usa `match/case` nativo de Python 3.10: compilado en bytecode CPython
          como jump table O(1), sin el overhead de instanciación de objetos ni
          dict lookups.
        - Agrupa las fases por rol de dominio (estructura de objeto, lectura de
          keys, strings, números, literales) delegando en handlers concisos.
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
                return False  # Objeto raíz vacío: el schema exige name/parameters
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
            # ⚠ DESVÍO DOCUMENTADO (Task 3.3): arranca un value de "name"
            # NUEVO → reset del buffer acumulado. OJO el depth: el parámetro
            # "name" de fn_greet vive en depth 1 y NO resetea el del output.
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
            self.phase = DecoderPhase.ESCAPE_IN_STRING
            return True
        # ⚠ DESVÍO DOCUMENTADO (Task 3.3): acumula el text del value de
        # "name" SOLO en el name del output object (depth 0). Los chars en
        # ESCAPE_IN_STRING y los hex de \uXXXX ya pasaron por arriba
        # (skippeados): el buffer queda con el nombre "decodificado".
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
        """Reinicia el acumulador de key al abrir una nueva (viene el '"')."""
        self.current_key = ""

    def _register_params_key(self) -> None:
        """Suma current_key a keys_enclosed si la key vive en parameters.

        POR QUÉ depth == 1: keys_enclosed alimenta a schema_validator
        (Task 3.3) para saber qué required keys de parameters ya fueron
        emitidas. Las keys del output object ("name", "parameters") NO van.
        NOTA de memoria: el set SIEMPRE se reemplaza (set | {...}), nunca se
        muta in-place, para que copy.copy() de simulate sea seguro: un shallow
        copy comparte la referencia del set; si la mutáramos, la copia
        contaminaría al original (y viceversa).
        """
        if self.depth == 1 and self.current_key:
            self.keys_enclosed = set(self.keys_enclosed) | {self.current_key}

    def _close_value(self, terminal: str) -> bool:
        """Cierra el value actual con un terminal (ws / ',' / '}').

        Pre: la VALIDACIÓN del tipo (number grammar o literal completo) ya
        corrió en la rama de la phase; acá solo la estructura: registrar la
        key cerrada y mover a la fase de espera.
        El whitespace NO consume: deja la phase en VALUE_END (el siguiente
        ',' o '}' real es quien transiciona). Los buffers se limpian para
        no dejar basura de un value en el siguiente.
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
        """Literal contra el que se valida el buffer bool/null en curso."""
        if self.phase is DecoderPhase.IN_NULL_VALUE:
            return "null"
        return "true" if self.bool_buffer[:1] == "t" else "false"

    def _is_valid_json_number(self) -> bool:
        """True si el buffer es un number JSON COMPLETO (regex estricta)."""
        return _NUMBER_RE.fullmatch(self.number_buffer) is not None

    def _number_next_chars(self) -> set[str]:
        """Chars que mantienen el number actual como PREFIXO válido.

        CÓMO FUNCIONA (por dentro): prueba cada char de la grammar contra el
        regex de prefijo. Con number_buffer == "2.", solo pasan los dígitos
        (la fracción es obligatoria); con "2e" pasan dígitos y +/-. Esto NO
        se puede derivar de los dos booleanos number_has_digit/number_has_dot
        del plan: necesita la cadena acumulada (ver Q&A de diseño).
        """
        if not self.number_buffer:
            return {"-", *(_DIGITS)}
        return {
            ch
            for ch in "0123456789+-.eE"
            if _NUMBER_PREFIX_RE.fullmatch(self.number_buffer + ch) is not None
        }
