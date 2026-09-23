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
from src.decoder.trie import TrieNode, find_node, is_complete_name
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

# Fases de value "puro" (string/number/bool/null): sirven para detectar si un
# token ENTRÓ a un value en este step (pre no estaba acá, post sí).
_VALUE_PHASES = (
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.IN_NUMBER_VALUE,
    DecoderPhase.IN_BOOL_VALUE,
    DecoderPhase.IN_NULL_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)

# Fases en las que el decoder está DENTRO del value de la key "name" del
# output object (depth 0), o a punto de arrancarlo (COLON ya definió la key).
# El filtro (Task 3.4) las usa para aplicar el trie (Fase 3, cláusula 1).
_NAME_READ_PHASES = (
    DecoderPhase.COLON,
    DecoderPhase.IN_STRING_VALUE,
    DecoderPhase.ESCAPE_IN_STRING,
)

# Tipo JSON declarado por cada fase de value en curso (cláusula 3: value type).
_PHASE_KIND: dict[DecoderPhase, str] = {
    DecoderPhase.IN_STRING_VALUE: "string",
    DecoderPhase.ESCAPE_IN_STRING: "string",
    DecoderPhase.IN_NUMBER_VALUE: "number",
    DecoderPhase.IN_BOOL_VALUE: "boolean",
    DecoderPhase.IN_NULL_VALUE: "null",
}

# Tipo JSON declarado por el PRIMER carácter de un value que arranca en COLON
# (mismo set de chars que expected_first_chars en COLON, menos '{' = objeto).
# Cubre el value que se abre Y se cierra dentro del MISMO token ('2,', 'true}',
# '"x",'): en ese caso el post-state no queda en una fase de value y el tipo
# hay que leerlo del texto.
_VALUE_START_KINDS: dict[str, str] = {
    '"': "string",
    "-": "number",
    **{ch: "number" for ch in "0123456789"},
    "t": "boolean",
    "f": "boolean",
    "n": "null",
}


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
        "_params_object_seen",
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
        #: True si el recorrido pasó por PARAMS_OBJECT (el '{' de "parameters").
        #: Lo setea update() y lo consume el pase fino del generator (Inciso
        #: 4.1.1): cierra el gap del plan "output object sin parameters".
        self._params_object_seen = False
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
        if state.phase is DecoderPhase.PARAMS_OBJECT:
            self._params_object_seen = True
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

    def has_seen_params_object(self) -> bool:
        """True si el recorrido pasó por el '{' del objeto "parameters".

        LO CONSUME EL PASO FINO del generator (Inciso 4.1.1): cuando el
        estado llega a COMPLETE, exige que este flag esté encendido. Cierra
        el gap del plan "output object sin parameters" ('{"name":"fn"}'
        completo sin el objeto parameters jamás se emite). El flag lo setea
        update() cuando ve PARAMS_OBJECT (solo el '{' lo produce).
        """
        return self._params_object_seen

    # ------------------------------------------------ Fase 3 del filter (3.4)

    def allows_token(
        self, token_text: str, new_state: DecoderState, trie: TrieNode
    ) -> bool:
        """Semántica de UN token candidato: True si el schema lo permite.

        QUÉ ES (por dentro):
        - PURA: NO muta este SchemaContext. El filter (Task 3.4) la llama por
          cada candidato en Fase 3; acá SOLO se lee el snapshot commiteado del
          último update(state) (self._phase/_current_key/_depth/_keys_enclosed)
          + el estado SIMULADO new_state que el filter acaba de producir.
        - CUATRO cláusulas ANDed, cada una con su trigger:
            1. _allows_name_value  → trie contra el value de "name"
            2. _allows_param_key   → keys del objeto parameters (depth 1)
            3. _allows_value_type  → tipo del value de un parámetro
            4. _allows_params_close→ el '}' de cierre de parameters
        - El parámetro token_text (firma del plan) lo usa SOLO la cláusula de
          tipo (primer char de un value que abre Y cierra en el mismo token).
          El resto trabaja con fases y buffers: justamente el punto del
          desvío name_buffer (Task 3.3) — el schema no necesita re-parsear.

        GAPS (scope del plan, deliberados):
        - El '}' de cierre del OUTPUT object (depth 0 → COMPLETE) NO se gatea
          acá: el plan solo bloquea el cierre de parameters. Consecuencia:
          '{"name":"fn"}' sin "parameters" se completa (el schema no exige
          presence del key "parameters").
        - Las keys del output object ("name"/"parameters") NO se validan como
          keys: "parameters" con value string, o un "extra" al nivel superior,
          pasan la semántica (la sintaxis y el trie hacen lo suyo).
        - Values anidados (depth >= 2) escapan a estas cláusulas: el plan solo
          modela parámetros escalares de UN nivel (B8 futuro). La state
          machine ni siquiera sensa bien el depth a partir de 2.
        - Un name con ESCAPES queda "incompleto" en el buffer (los escapes se
          skippean en state.py): el trie lo bloquea por construcción. Es
          conservador y correcto: los nombres reales no usan escapes.
        """
        if not self._allows_name_value(new_state, trie):
            return False
        if not self._allows_param_key(new_state):
            return False
        if not self._allows_value_type(token_text, new_state):
            return False
        if not self._allows_params_close(new_state):
            return False
        return True

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

    # ------------------------------------------------ cláusulas de Fase 3

    def _allows_name_value(
        self, new_state: DecoderState, trie: TrieNode
    ) -> bool:
        """Cláusula 1: el value de "name" debe ser un nombre del trie.

        CÓMO FUNCIONA (por dentro):
        - El buffer lo acumula LA STATE MACHINE (state.name_buffer), con los
          escapes skippeados (nombre "decodificado").
        - DOS ramas, elegidas por dónde termina el token:
            * Termina DENTRO del value de "name" (post ∈ _NAME_READ_PHASES,
              key "name", depth 0): el buffer acumulado — incluido lo que
              este token agregó — debe ser un PREFIJO de algún nombre
              (find_node ≠ None). Es la rama "el name sigue construyéndose".
            * SALIÓ del value en este token (pre ∈ _NAME_READ_PHASES, key
              "name", depth 0, y post no): este token cerró el string (o
              avanzó a estructura, p.ej. 'greet", "parameters": {'). El
              buffer FINAL debe ser un nombre COMPLETO (is_complete_name).
        - El caso '{"name": {' (value que arranca con '{'): pre=COLON cae en
          la rama 2 con buffer "" → is_complete_name("") = False → bloqueado.
          Un number/literal como value de "name" se bloquea igual: el buffer
          queda "" y "" no es un nombre completo.
        - ¿Por qué estas dos ramas y no más? Un token que EMPIEZA fuera del
          name y lo atraviesa COMPLETO (key + value + cierre, p.ej.
          '}, "name": "fn_greet",') no gatilla ninguna: el contenido del
          name en ese token no se valida contra el trie (quedaría validado el
          de tokens posteriores... que ya no existen). Raro en vocabularios
          BPE reales; limitación documentada.
        """
        if (
            new_state.phase in _NAME_READ_PHASES
            and new_state.current_key == "name"
            and new_state.depth == 0
        ):
            return find_node(trie, new_state.name_buffer) is not None
        if (
            self._phase in _NAME_READ_PHASES
            and self._current_key == "name"
            and self._depth == 0
        ):
            return is_complete_name(trie, new_state.name_buffer)
        return True

    def _allows_param_key(self, new_state: DecoderState) -> bool:
        """Cláusula 2: las keys de parameters (depth 1) valen contra el schema.

        CÓMO FUNCIONA (por dentro):
        - Trigger por CAMBIO, no por fase: gatilla si el texto de la key
          (new_state.current_key) cambió durante este token — incluye el
          reset a "" cuando el token abre una key nueva. Esto cubre el caso
          que las fases no ven: un token como ', "b": 3.0' arranca DENTRO
          del value anterior, cierra, y lee la key "b" a mitad de token
          (termina en IN_NUMBER_VALUE, fuera de cualquier fase de key).
        - "Abierta" (post en KEY_START/IN_KEY): la key se está construyendo,
          cualquier prefijo de una key disponible es válido
          (any(k.startswith(key))). "Cerrada" (el resto): membership exacta.
        - available se calcula contra keys_enclosed COMMITEADO (self._keys_
          enclosed), NO contra el del estado simulado. Motivo: si el token
          cierra el value de la key nueva en el MISMO paso (', "b": 2,'), el
          set simulado ya la contiene y la membership la bloquearía siendo
          perfectamente legítima. Contra el commiteado, "b" sigue siendo
          válida; y un DUPLICADO exacto ('"b": 3' con "b" ya emitida) queda
          bloqueado porque su key ya estaba en el set commiteado.
        - Sin función seleccionada → available vacío → TODA key de params se
          bloquea. Refuerzo deliberado (más fuerte que el plan): fuerza
          name-antes-de-parameters en la práctica, porque el '}' de cierre
          también se bloquea sin función (cláusula 4) y el generador no puede
          salir del objeto parameters sin haber nombrado la función.
        - Limitación residual 1 (duplicado): una key re-emitida con texto
          IDÉNTICO al commiteado (la máquina la resetea a "" y la reconstruye
          igual) no se detecta como cambio. Duplicados exactos en el MISMO
          texto de key que ya estuviera commiteado sí se bloquean; el caso
          "reset y reconstrucción idéntica en un solo token" escapa (raro en
          BPE).
        - Limitación residual 2 (DESCUBIERTA): el trigger exige self._depth
          == 1 COMMITEADO; un token que ENTRA a parameters y abre la PRIMERA
          key en el MISMO paso ('", "parameters": {"a') arranca en depth 0 →
          `self._depth != 1` abstiene y la key jamás se valida, ni en este
          token ni en los siguientes (el trigger por cambio no vuelve a
          disparar: current_key sigue siendo "a"). La key entra de contrabando
          y el schema la acepta de por vida. Raro en BPE real (token de ~20
          chars); el pase fino post-argmax (viable en Task 4.1) lo cierra.
        """
        if self._depth != 1:
            return True  # output keys (depth 0): fuera del scope del schema
        if new_state.current_key == self._current_key:
            return True  # este token no tocó el texto de la key
        if self.selected_function is None:
            return False
        available = set(self.selected_function.parameters) - self._keys_enclosed
        key = new_state.current_key
        if new_state.phase in (DecoderPhase.KEY_START, DecoderPhase.IN_KEY):
            return any(k.startswith(key) for k in available)
        return key in available

    def _allows_value_type(
        self, token_text: str, new_state: DecoderState
    ) -> bool:
        """Cláusula 3: el value de un parámetro debe declarar su tipo.

        CÓMO FUNCIONA (por dentro):
        - El tipo se lee de DOS lugares, según cómo termina el token:
            * Termina DENTRO de una fase de value (post ∈ _VALUE_PHASES): la
              fase declara el tipo (_PHASE_KIND). Como la fase de un value en
              curso NO cambia token a token, re-chequear continuaciones es
              idempotente: mismo phase → mismo veredicto.
            * Arrancó en COLON y el value se abrió y cerró EN ESTE token
              ('2,' / 'true}' / '"x",'): el post-state ya no está en fase de
              value; el tipo se lee del PRIMER carácter del texto
              (_VALUE_START_KINDS, con lstrip por el ws inicial).
        - Solo aplica a depth 1 con función seleccionada: el value de un
          parámetro conocido. El value de "name" (depth 0) NO pasa por acá:
          lo restringe el trie (cláusula 1). Keys sin tipo conocido en el
          schema → default allow. '{' (objeto anidado) no está en
          _VALUE_START_KINDS → sin constraint (gap de depth >= 2).
        - Limitación residual (documentada): un token que abra, complete y
          cierre un value NUEVO arrancando desde DENTRO del value anterior
          (', "b": "x",' — key nueva + value string + cierre, todo en uno)
          ni termina en fase de value ni arranca en COLON: esquiva la
          cláusula. Raro en vocabularios BPE reales; parsear key:value
          múltiples por token es trabajo de un parser paramétrico (B8).
        """
        kind: str | None = None
        if new_state.phase in _VALUE_PHASES:
            kind = _PHASE_KIND[new_state.phase]
        elif self._phase is DecoderPhase.COLON:
            kind = _VALUE_START_KINDS.get(token_text.lstrip()[:1])
        if kind is None:
            return True  # el token no abrió un value (o es de tipo no escalar)
        if new_state.depth != 1 or self.selected_function is None:
            return True  # solo se tipa el value de un parámetro (depth 1)
        param = self.selected_function.parameters.get(new_state.current_key)
        if param is None:
            return True  # key sin tipo conocido: default allow
        return kind == param.type

    def _allows_params_close(self, new_state: DecoderState) -> bool:
        """Cláusula 4: el '}' de cierre de parameters exige los required.

        CÓMO FUNCIONA (por dentro):
        - Trigger por depths: self._depth == 1 (commiteado, DENTRO de
          parameters) y new_state.depth == 0 → este token cerró el objeto
          parameters (la máquina solo baja de depth 1 con '}'). El '}' de
          cierre del output object (depth 0 → COMPLETE) NO gatilla: gap del
          plan (ver docstring de allows_token).
        - keys_enclosed del ESTADO SIMULADO (new_state): si el cierre viene
          junto al value final ('"b": 3}'), ese value ya se registró en la
          copia simulada y cuenta para el ⊆.
        - Sin función seleccionada → False: no se puede afirmar que puede
          cerrar sin conocer los required (conservador, alineado con
          can_close_params).
        """
        if self._depth != 1 or new_state.depth != 0:
            return True
        if self.selected_function is None:
            return False
        return set(self.selected_function.parameters) <= new_state.keys_enclosed
