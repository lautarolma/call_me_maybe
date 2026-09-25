"""Constrained generator (Task 4.1): el loop de generación.

POR QUÉ EXISTE ESTE MÓDULO (por dentro):
- Es la cinta transportadora del plan didáctico: por cada step toma los
  logits del modelo, los filtra a los ids que mantienen el output como JSON
  válido (compute_allowed_ids), elige por argmax el mejor token permitido,
  commitea el estado y repite hasta COMPLETE o el límite de tokens.
- Es el ÚNICO consumidor de compute_allowed_ids: el filter (Task 3.4) reduce
  los ~151K ids del vocab a un set pequeño; acá vive el argmax que la firma
  del filter tenía reservado (el filter NO consume logits).

INCISO 4.1.1 — PASO FINO POST-ARgMAX (corrección documentada en el plan):
- Los gaps de las cláusulas del schema (documentados en sus docstrings) son
  abstención por no-coincidencia de bordes pre/post: el filter valida el
  token como un TODO (snapshot commiteado + snapshot simulado), no el
  recorrido char-por-char. Un token que entra Y sale de estructuras en un
  solo step escapa a las cláusulas: name completo atravesado (cláusula 1),
  entrada a parameters + primera key (cláusula 2 residual 2), key+value+
  cierre completos (cláusula 3), entrada y salida de parameters (cláusula 4).
- Acá se cierra con COSTE DESPRECIABLE: re-simular SOLO el token GANADOR
  char-por-char con un SchemaContext FRESCO (no el compartido, que quedaría
  contaminado si el candidato falla) y validar en CADA carácter. En el
  recorrido por carácter las cláusulas gatillan donde el pase por token no
  las gatillaba: el reset de current_key, el COLON intermedio, el depth 0→1→0.
- Si el ganador no pasa el pase fino, se descarta y se prueba el SIGUIENTE
  mejor de allowed (argmax repetido). Normalmente el primero pasa (~1
  re-simulación por step); el peor caso es degradación controlada.
- Cierra además el gap del plan "output object sin parameters": cuando el
  estado llega a COMPLETE, el pase fino exige que el recorrido haya pasado
  por PARAMS_OBJECT (SchemaContext.has_seen_params_object). El subject V.4.1
  SIEMPRE emite el objeto (aún vacío para fn_empty).

DESVÍO DOCUMENTADO del pseudocódigo didáctico (L1745):
- El commit usa vocab.id2decoded, NO vocab.id2token. Mismo desvío que el
  filter (token_filter.py desvío 1): la state machine trabaja con el texto
  DECODIFICADO ('Ġthe' → ' the'); id2token aportaría un token byte-mapheado
  que rompería la validación sintáctica. El decode final para el output se
  hace con model.decode() sobre los ids (el SDK aplica la tabla inversa).

QUÉ NO HACE (separación de concerns):
- No orquesta el pipeline completo (loaders, output file): eso vive en Task
  4.2/4.3 + pipeline.py. Acá solo el bucle de generación pura.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from time import perf_counter

from llm_sdk import Small_LLM_Model

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.token_filter import compute_allowed_ids
from src.decoder.trie import TrieNode
from src.loader.vocab_loader import Vocab
from src.models.function_definition import FunctionDef
from src.utils.metrics import MetricsRun

MAX_TOKENS = 200  # Safety net — el output esperado es ~30-60 tokens

# Opt2 (Anexo de Latencia §2.2, CONTEXTO_REFACTOR.md): prefijo 100%
# determinista por el schema — TODO output válido arranca con esta
# estructura antes de que el LLM tenga que elegir el nombre real de la
# función. Se tokeniza UNA vez con encode() y se inyecta sin forward ni
# filtro (ver _inject_static_header): recorta los forwards estructurales
# de ROOT/OBJECT_OPEN/KEY_START/IN_KEY/KEY_END/COLON medidos en el Anexo
# (~8-11 por prompt, ~4.5-4.8s cada uno — el costo del forward() del
# modelo es uniforme por step, no depende del tamaño del candidate set,
# así que evitar el forward es la única palanca real acá).
# ⚠ BUG-012 (2026-09-23): el header DEBE ser BYTE-EXACTO al formato natural
# que el modelo produce solo (con las 2 newlines iniciales antes de '{',
# como en el Anexo — la política compacta ya había fallado antes por lo
# mismo, commit abortado 7bf38ed). Recortar esas newlines "porque total no
# cuestan forward" (inyectar más texto no cuesta nada extra: TODO el header
# se salta el modelo por igual) rompió P2 ('Greet shrek'): el modelo quedó
# en un estado fuera de distribución y tokenizó el name como un "f" suelto
# en vez de un chunk natural, dejando sin candidatos válidos el step
# siguiente (ningún token entre los top-2000 por logit mantenía "f..." como
# prefijo del trie). Con las newlines restauradas, P2 vuelve a completar.
STATIC_HEADER = '\n\n{\n  "name": "'

# ─── Oráculo por estado — Fase 2 (Anexo §"Registro de decisiones 25/09") ──
# Generaliza el 2º tramo lineal (que trajo BUG-013) a una tabla de tramos
# por estado (NIVEL 1 del registro formal): cada tramo es una FUNCIÓN PURA
# (state, schema, emitted) -> texto canónico alineado, o None (None = se
# cae al camino normal: M5 → filtro → forward). El oráculo NUNCA fuerza:
# solo responde cuando el tramo es la única continuación determinista del
# schema (Riesgo 2 del 2º tramo cubierto POR DISEÑO).
#
# Definiciones del registro (Anexo D1-D4, verificadas contra state.py):
#   WS = " \t\n\r" · E = ws final de emitted (SOLO en fases ciegas al ws:
#   VALUE_END/IN_OBJECT/PARAMS_OBJECT/KEY_END/COLON — en KEY_START/IN_KEY
#   el espacio es contenido de la key y no se usa).
#   N = depth==0 ∧ current_key=="name" ∧ keys_enclosed==∅  → gate del
#   tramo de apertura de parameters. Reemplaza a ¬has_seen_params_object:
#   ese flag es sticky y SOLO se enciende si el token termina en
#   PARAMS_OBJECT (update() corre con el estado post-token) → un token BPE
#   que cruza el '{' (ej. '{"') lo deja apagado con parameters ya abierto:
#   falso negativo EN LA DIRECCIÓN INSEGURA (duplicaría el tramo). N, en
#   cambio, es derivable por casos: BUG-013 (param interno "name" de
#   fn_greet) deja keys_enclosed≠∅ → N=0; fn_empty (0 params) deja
#   current_key=="parameters" → N=0; N=1 ⟺ entre el value de "name" y la
#   key siguiente.
#   ord = tuple(F.parameters)  (insertion order — test dedicado)
#   ρ = (R != ∅) con R = schema.required_keys_remaining()
#   K1 = ord[0] · Knext = primer k de ord con k∈R
#   OP(k) = '"' si F.parameters[k].type=="string" sino ''  (comilla de
#   apertura del value: los tramos dejan el estado en COLON o
#   IN_STRING_VALUE d1 según el tipo)
#   KEY(k) = '"' + k + '": ' + OP(k) · ENTRY(k) = '\n    ' + KEY(k)
#   align(C) = C[len(E):] si C.startswith(E) sino None; '' → None.
# Tramo GRATIS (dec. 25/09): con ord=∅, T1/T2 devuelven None → el modelo
# genera "parameters": {} con su token FUSIONADO {} (tid 6257) — inyectar
# '{' suelto sería una costura tipo BUG-012 sobre un formato que el modelo
# nunca produce. Ninguna función del subject cae ahí (todas tienen params);
# el caso fn_empty queda verificado por el probe real (formato INLINE).
# Nivel 2 (T7-T10, tokens fusionados) y B′ (completar fn_name por trie):
# DIFERIDOS hasta medir el residuo del Nivel 1 (Anexo D4/D7).
_WS = " \t\n\r"
_WS_BLIND_PHASES = frozenset(
    {
        DecoderPhase.VALUE_END,
        DecoderPhase.IN_OBJECT,
        DecoderPhase.PARAMS_OBJECT,
        DecoderPhase.KEY_END,
        DecoderPhase.COLON,
    }
)


def _trailing_ws(text: str) -> str:
    """E del registro: el whitespace FINAL de ``text`` ('' si no hay)."""
    return text[len(text.rstrip(_WS)):]


def _align(canon: str, emitted: str, blind: bool) -> str | None:
    """Alinea el canónico contra el ws ya emitido. '' → None (spec Anexo).

    Evita la duplicación de whitespace (la clase de BUG-012): si el modelo
    ya emitió el ws del canónico (E), se corta y se inyecta solo el resto.
    Si el canónico NO arranca con E, el alineamiento es imposible → None →
    el tramo no aplica y la generación cae al forward (seguro).
    """
    e = _trailing_ws(emitted) if blind else ""
    if not canon.startswith(e):
        return None
    rest = canon[len(e):]
    return rest or None


def _gate_n(state: DecoderState) -> bool:
    """N del registro: entre el value de 'name' y la key siguiente."""
    return (
        state.depth == 0
        and state.current_key == "name"
        and not state.keys_enclosed
    )


def _op(schema: SchemaContext, key: str) -> str:
    """OP(k): comilla de apertura del value si el parámetro es string."""
    f = schema.selected_function
    if f is None:
        return ""
    param = f.parameters.get(key)
    return '"' if param is not None and param.type == "string" else ""


def _key(schema: SchemaContext, key: str) -> str:
    """KEY(k): '"' + k + '": ' + OP(k) — key más apertura de su value."""
    return f'"{key}": ' + _op(schema, key)


def _entry(schema: SchemaContext, key: str) -> str:
    """ENTRY(k): '\n    ' + KEY(k) — formato natural de Qwen por key."""
    return "\n    " + _key(schema, key)


def _knext(schema: SchemaContext) -> str:
    """Knext: primer key de ord que sigue requerida (ρ=1 ya validado)."""
    f = schema.selected_function
    if f is None:
        raise AssertionError("_knext() sin función seleccionada")
    remaining = schema.required_keys_remaining()
    for k in tuple(f.parameters):
        if k in remaining:
            return k
    raise AssertionError("_knext() sin keys pendientes (ρ=1 violado)")


def _tramp_t1(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T1: VALUE_END, d0, N=1 → la coma + apertura de parameters + 1ra key."""
    if not (state.phase is DecoderPhase.VALUE_END and _gate_n(state)):
        return None
    f = schema.selected_function
    if f is None:
        return None
    ord_ = tuple(f.parameters)
    if not ord_:
        return None  # ord=∅ → forward (token {} fusionado del modelo)
    canon = ",\n  \"parameters\": {" + _entry(schema, ord_[0])
    return _align(canon, emitted, True)


def _tramp_t2(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T2: IN_OBJECT, d0, N=1 → apertura + 1ra key (token fusionado '",')."""
    if not (state.phase is DecoderPhase.IN_OBJECT and _gate_n(state)):
        return None
    f = schema.selected_function
    if f is None:
        return None
    ord_ = tuple(f.parameters)
    if not ord_:
        return None
    canon = "\n  \"parameters\": {" + _entry(schema, ord_[0])
    return _align(canon, emitted, True)


def _tramp_t3(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T3: PARAMS_OBJECT, d1, ρ=1 → la primera key requerida que falta.

    PARAMS_OBJECT es la fase "entre values" de este decoder (verificado:
    VALUE_END + ',' → PARAMS_OBJECT, NO IN_OBJECT). Por eso T3 cubre DOS
    entradas: (a) el '{' de apertura llegó por forward sin key fusionada, y
    (b) la coma que cierra el value ANTERIOR — el caso número: IN_NUMBER_VALUE
    (número abierto, sin VALUE_END) → la coma del modelo cierra el número y
    deja PARAMS_OBJECT directamente. El alineamiento E absorbe el ws que el
    modelo haya emitido tras la coma ('2.0,' + '\n    ' → e='\n    ').
    """
    if not (state.phase is DecoderPhase.PARAMS_OBJECT and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if not schema.required_keys_remaining():
        return None
    return _align(_entry(schema, _knext(schema)), emitted, True)


def _tramp_t4(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T4: VALUE_END, d1, ρ=1 → coma + próxima key requerida.

    Solo se alcanza con un value que CIERRA en su propio token (strings,
    y tokens que terminan en cierre): los números/booleans/nulls quedan
    ABIERTOS (IN_NUMBER/BOOL/NULL_VALUE) hasta el siguiente carácter →
    ese flujo retoma en T3 (la coma del modelo deja PARAMS_OBJECT).
    """
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if not schema.required_keys_remaining():
        return None
    canon = ",\n    " + _key(schema, _knext(schema))
    return _align(canon, emitted, True)


def _tramp_t5(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T5: VALUE_END, d1, ρ=0 → cierre de parameters + cierre del ROOT."""
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 1):
        return None
    if schema.selected_function is None:
        return None
    if schema.required_keys_remaining():
        return None
    return _align("\n  }\n}", emitted, True)


def _tramp_t6(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """T6: VALUE_END, d0, N=0, ρ=0 → cierre del ROOT (fn_empty incl.).

    NOTA — P NO se exige (corrección al contrato, probe 25/09): el token
    FUSIONADO del modelo '"parameters": {}' trae '{'+'}' en un solo
    tocho y el flag sticky del schema (update POST-token) NUNCA ve el
    PARAMS_OBJECT intermedio → has_seen_params_object() queda False en el
    caso real de fn_empty. El gate N=0 ∧ ρ=0 ya garantiza post-parameters:
    el ÚNICO VALUE_END d0 sin parameters abiertos es el post-name
    (current_key=="name" → N=1, bloqueado arriba); una vez N=0 ∧ ρ=0, el
    ROOT solo puede cerrarse.
    """
    if not (state.phase is DecoderPhase.VALUE_END and state.depth == 0):
        return None
    if _gate_n(state):
        return None  # N=0: dominios disjuntos con T1/T2
    if schema.selected_function is None:
        return None
    if schema.required_keys_remaining():
        return None
    return _align("\n}", emitted, True)


_TRAMPS: tuple[
    Callable[[DecoderState, SchemaContext, str], str | None], ...
] = (
    _tramp_t1,
    _tramp_t2,
    _tramp_t3,
    _tramp_t4,
    _tramp_t5,
    _tramp_t6,
)


def _next_static_text(
    state: DecoderState, schema: SchemaContext, emitted: str
) -> str | None:
    """Devuelve el texto a inyectar para ``state``, o None (camino normal).

    CÓMO FUNCIONA (oráculo por estado, Nivel 1):
    - Recorre _TRAMPS en orden y devuelve el texto del PRIMER tramo cuyo
      dominio matchea el estado. Los dominios son disjuntos por
      construcción (phase × depth × gates), así que el orden es de
      claridad, no de precedencia.
    - ``emitted`` es el texto del output generado hasta el momento (sin el
      prompt): alimenta E para el alineamiento anti-duplicación (BUG-012).
    - Pureza: no muta ni state ni schema, es determinista y no consulta el
      modelo (contrato del diseño consultado).
    """
    for tramp in _TRAMPS:
        text = tramp(state, schema, emitted)
        if text is not None:
            return text
    return None


def generate(
    model: Small_LLM_Model,
    prompt: str,
    vocab: Vocab,
    functions: list[FunctionDef],
    trie: TrieNode,
    max_tokens: int = MAX_TOKENS,
    metrics: MetricsRun | None = None,
) -> tuple[str, bool]:
    """Genera output JSON constrained para un prompt.

    Args:
        model: Modelo del SDK (encode/get_logits_from_input_ids/decode).
        prompt: Texto del prompt a responder con un function call.
        vocab: Vocabulario pre-indexado (id2token + id2decoded + buckets).
        functions: Definiciones de función del schema.
        trie: Trie de nombres de función (build_trie(functions)).
        max_tokens: Límite de seguridad del bucle.
        metrics: Acumulador opcional de métricas por fase; si se provee,
            registra forwards, skips-if-single y tiempo por DecoderPhase.

    Returns:
        (texto_generado, éxito): éxito True si el estado llegó a COMPLETE.

    CÓMO FUNCIONA (por dentro):
    - Mismo esqueleto que el pseudocódigo del plan (PLAN_DIDACTICO L1696):
      tokenizar prompt → estado/schema iniciales → por step: logits →
      compute_allowed_ids → argmax → append → commit → COMPLETE?.
    - El pase fino del Inciso 4.1.1 vive en el argmax: _pick_best_token()
      descarta los candidatos que no pasan _passes_fine_validation().
    """
    # ⚠ BUG-005 (2026-09-18): el SDK devuelve un tensor 2D [1, N]; [0].tolist()
    # lo aplana a list[int] — el contrato que espera get_logits_from_input_ids.
    input_ids = model.encode(prompt)[0].tolist()
    # Longitud del prompt ANTES del loop: input_ids después solo crece con los
    # best_id generados, así el slice final separa prompt de generados sin
    # re-encodear (antes se re-encodeaba y len() contaba FILAS del tensor 2D).
    prompt_length = len(input_ids)
    state = DecoderState()
    schema = SchemaContext(functions)
    # emitted: el texto GENERADO del output (sin el prompt), acumulado por
    # iteración (dec. 25/09). Alimenta E (ws final) en el oráculo para el
    # alineamiento anti-duplicación; se actualiza en CADA punto de commit
    # (header, oráculo, M5 y forward) — nunca se resetea ni se deriva de
    # estructuras aparte.
    emitted_parts: list[str] = []
    _inject_static_header(
        STATIC_HEADER, model, vocab, input_ids, state, schema, emitted_parts
    )

    for _ in range(max_tokens):
        step_phase = state.phase
        step_start = perf_counter()
        try:
            # ─── Oráculo por estado (Fase 2): tramos estáticos Nivel 1 ───
            # ANTES del filter y del forward: si el estado matchea un tramo
            # determinista (_TRAMPS), inyectarlo pre-tokenizado. Si la
            # inyección no avanza el estado, se cae al camino normal — el
            # continue SOLO ocurre con avance real (sin eso, re-matchear el
            # mismo tramo en el step siguiente sería un loop infinito).
            tail = _next_static_text(state, schema, "".join(emitted_parts))
            if tail is not None and _inject_static_header(
                tail, model, vocab, input_ids, state, schema, emitted_parts
            ):
                if state.phase is DecoderPhase.COMPLETE:
                    break
                continue

            # ─── M5: Skip-if-single (Anexo de Latencia) ───
            # Check first WITHOUT model call. In non-wildcard phases, the
            # candidate set is small (~10-100 tokens) so the full filter is
            # fast. If exactly 1 candidate exists, we can skip the forward
            # call entirely.

            expected_chars = state.expected_first_chars()
            if "*" not in expected_chars:
                allowed_check = compute_allowed_ids(state, schema, vocab, trie)
                if len(allowed_check) == 1:
                    if metrics is not None:
                        metrics.add_skips(step_phase, 1)
                    single_id = next(iter(allowed_check))
                    token_text = vocab.id2decoded.get(single_id)
                    if token_text is None:
                        break
                    input_ids.append(single_id)
                    if not state.update_from_text(token_text):
                        break
                    schema.update(state)
                    emitted_parts.append(token_text)
                    if state.phase is DecoderPhase.COMPLETE:
                        break
                    continue

            # ─── Ambiguous step: consult model ───
            logits = model.get_logits_from_input_ids(input_ids)
            if metrics is not None:
                metrics.add_forward(step_phase)
            # M1/M2: Top-1 opportunistic + Top-K masking via logits
            allowed = compute_allowed_ids(state, schema, vocab, trie, logits)

            if not allowed:
                # Empty set handling: el plan dice "attempt repair or break".
                # MVP: break — el output queda truncado y success=False.
                break

            # Argmax sobre allowed + pase fino (Inciso 4.1.1): descarta el
            # mejor candidato si no supera la re-simulación char-por-char.
            best_id, token_text = _pick_best_token(
                allowed, logits, state, schema, functions, vocab, trie
            )
            if best_id is None:
                # Ningún candidato de allowed pasó el pase fino.
                break

            input_ids.append(best_id)

            # ⚠ DESVÍO del plan (ver docstring del módulo): se commitea con
            # el texto DECODIFICADO, el mismo que vio la state machine en
            # simulate().
            if not state.update_from_text(token_text):
                # No debería pasar: el filter ya validó este token en Fase 2.
                # Defensivo: si ocurriera, el estado queda atómico.
                break
            schema.update(state)
            emitted_parts.append(token_text)

            if state.phase is DecoderPhase.COMPLETE:
                break
        finally:
            # El timing del step se acumula en la fase en la que ARRANCÓ
            # (step_phase); correr SIGUE el contrato con continue/break.
            if metrics is not None:
                metrics.add_elapsed(
                    step_phase, (perf_counter() - step_start) * 1000.0
                )

    # Solo los tokens GENERADOS (no el prompt): prompt_length se calculó antes
    # del loop sobre los ids reales del prompt (BUG-005). Sin el [0].tolist(),
    # len() contaría las FILAS del tensor 2D y el slice arrastraría tokens.
    generated_ids = input_ids[prompt_length:]
    generated = model.decode(generated_ids)

    return generated, state.phase is DecoderPhase.COMPLETE


def _inject_static_header(
    header: str,
    model: Small_LLM_Model,
    vocab: Vocab,
    input_ids: list[int],
    state: DecoderState,
    schema: SchemaContext,
    emitted_parts: list[str] | None = None,
) -> bool:
    """Precarga ``header`` en input_ids/state/schema sin llamar al modelo.

    CÓMO FUNCIONA (por dentro):
    - `model.encode(header)` tokeniza el prefijo UNA vez; cada id resultante
      se commitea con el mismo `state.update_from_text` del loop principal
      (mismo contrato: atómico, mueve la state machine char por char).
    - `emitted_parts` (opcional): cuando se pasa, cada decoded commitado se
      acumula en él. El oráculo lo usa para computar E (ws final emitido) y
      alinear sus canónicos — sin esto, `_next_static_text` no podría saber
      qué ws ya salió por forward y duplicaría whitespace (clase BUG-012).
    - Best-effort defensivo: el texto es JSON válido por construcción
      (mismo grammar que valida `state.py`), así que no debería fallar. Si
      algún id no decodifica o `update_from_text` rechaza el texto (p.ej.
      un split de tokenizer inesperado), se corta la inyección ahí mismo —
      el estado queda atómico (sin ese id) y el loop principal retoma
      generando ese tramo por forward normal, sin crashear.

    Returns:
        True si se inyectó AL MENOS UN id (el estado avanzó); False si el
        texto no aportó ids o todos fueron rechazados. El caller de los
        tramos del oráculo DEBE consultar esto antes de `continue`: si no
        avanza y se continúa, el step siguiente ve el MISMO estado →
        matchea el mismo trigger → inyección fallida otra vez → loop
        infinito. (El header 1 pre-loop ignora el retorno: se llama una
        sola vez, fuera del loop — no tiene ese riesgo.)
    """
    header_ids = model.encode(header)[0].tolist()
    advanced = False
    for token_id in header_ids:
        decoded = vocab.id2decoded.get(token_id)
        if decoded is None or not state.update_from_text(decoded):
            break
        input_ids.append(token_id)
        schema.update(state)
        if emitted_parts is not None:
            emitted_parts.append(decoded)
        advanced = True
    return advanced


def _pick_best_token(
    allowed: set[int],
    logits: list[float],
    state: DecoderState,
    schema: SchemaContext,
    functions: list[FunctionDef],
    vocab: Vocab,
    trie: TrieNode,
) -> tuple[int | None, str]:
    """Argmax sobre allowed, con pase fino del ganador (Inciso 4.1.1).

    CÓMO FUNCIONA (por dentro):
    - `max(allowed, key=lambda tid: logits[tid])` = argmax restringido al
      set permitido: el token con logit más alto que el modelo prefiere.
    - Si el ganador NO pasa _passes_fine_validation(), se descarta del set
      y se elige el siguiente mejor. Devolver (None, "") = allowed agotado.
    """
    while allowed:
        best_id = max(allowed, key=lambda tid: logits[tid])
        token_text = vocab.id2decoded.get(best_id)
        if token_text is not None and _passes_fine_validation(
            functions, schema, state, trie, token_text
        ):
            return best_id, token_text
        allowed.discard(best_id)
    return None, ""


def _passes_fine_validation(
    functions: list[FunctionDef],
    schema: SchemaContext,
    state: DecoderState,
    trie: TrieNode,
    token_text: str,
) -> bool:
    """Re-simulación char-por-char del token GANADOR con schema fresco.

    QUÉ ES (por dentro):
    - Se copia el estado commiteado (copy barata, slots) y se construye un
      SchemaContext FRESCO (el compartido NO se toca: si el candidato falla
      a mitad de camino, quedaría contaminado). El flag _params_object_seen
      se SIEMBRA desde el schema real: el historial de tokens anteriores
      ("¿ya abrimos parameters?") no puede re-derivarse de este token solo.
    - ORDEN CRÍTICO por carácter (el mismo contrato del filter): avanzar la
      máquina (muta trial) → allows_token(char, trial) con el schema fino
      TODAVÍA sincronizado con el estado PRE-char → recién después
      fine.update(trial). Si el update fuera primero, self.* == new_state.*
      en allows_token y las cláusulas de cambio/depth (2 y 4) jamás gatillan
      — el pase fino quedaría en no-op.
    - ¿Por qué esto cierra los gaps? En el pase por TOKEN el schema solo ve
      pre (commiteado) y post (simulado); acá ve TODOS los estados
      intermedios, así las cláusulas gatillan donde antes no:
        * Cláusula 2: el reset de current_key ("a" → "" → "b") se ve como
          cambio y el prefix check bloquea keys inexistentes/duplicadas.
        * Cláusula 3: el COLON intermedio expone el tipo esperado antes de
          que el value se abra ('", "b": "x",' → string para un number se
          bloquea en el '"' que abre el string).
        * Cláusula 1: el cierre del name en el MISMO token gatilla la rama
          is_complete_name (el pase por token la perdía si arrancaba fuera).
        * Cláusula 4: el depth 0→1→0 dentro de un token dispara el ⊆ de
          required contra keys_enclosed del estado intermedio.
    - COMPLETE sin haber pasado por PARAMS_OBJECT → False (gap del plan:
      el output SIEMPRE lleva "parameters", aún vacío para fn_empty).
    """
    trial = copy(state)
    fine = SchemaContext(functions)
    # ⚠ El historial de tokens anteriores (¿se abrió parameters?) viene del
    # schema real; el del token actual NO alcanza para decidir COMPLETE.
    fine._params_object_seen = schema.has_seen_params_object()
    fine.update(trial)

    for char in token_text:
        if not trial.update_from_text(char):
            # Sintaxis: el filter ya validó el token completo; si un char
            # fallara acá sería un bug del filter. Falla cerrada.
            return False
        # allows_token con schema PRE-char + estado POST-char (contrato 3.4)
        if not fine.allows_token(char, trial, trie):
            return False
        # Recién acá el schema fino avanza al estado de este carácter.
        fine.update(trial)

    if trial.phase is DecoderPhase.COMPLETE and not fine.has_seen_params_object():
        return False
    return True
