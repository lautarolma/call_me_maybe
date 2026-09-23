"""Constrained generator (Task 4.1): el loop de generación.

POR QUÉ EXISTE ESTE MÓDULO (por dentro):
- Es la cinta transportadora del plan didáctico: por cada step toma los
  logits del modelo, los filtra a los ids que mantienen el output como JSON
  válido (compute_allowed_ids), elige por argmax el mejor token permitido,
  commitea el estado y repite hasta COMPLETE o el límite de tokens.
- Es el consumidor del filtro: usa compute_allowed_ids para pasos
  deterministas y el iterador rankeado para pasos ambiguos.

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

from copy import copy

from llm_sdk import Small_LLM_Model

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.token_filter import compute_allowed_ids, iter_ranked_allowed_ids
from src.decoder.trie import TrieNode
from src.loader.vocab_loader import Vocab
from src.models.function_definition import FunctionDef

MAX_TOKENS = 200  # Safety net — el output esperado es ~30-60 tokens


def generate(
    model: Small_LLM_Model,
    prompt: str,
    vocab: Vocab,
    functions: list[FunctionDef],
    trie: TrieNode,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, bool]:
    """Genera output JSON constrained para un prompt.

    Args:
        model: Modelo del SDK (encode/get_logits_from_input_ids/decode).
        prompt: Texto del prompt a responder con un function call.
        vocab: Vocabulario pre-indexado (id2token + id2decoded + buckets).
        functions: Definiciones de función del schema.
        trie: Trie de nombres de función (build_trie(functions)).
        max_tokens: Límite de seguridad del bucle.

    Returns:
        (texto_generado, éxito): éxito True si el estado llegó a COMPLETE.

    CÓMO FUNCIONA (por dentro):
    - Mismo esqueleto que el pseudocódigo del plan (PLAN_DIDACTICO L1696):
      tokenizar prompt → estado/schema iniciales → por step: logits →
      filtro → selección → append → commit → COMPLETE?.
    - El pase fino del Inciso 4.1.1 consume candidatos ordenados y descarta
      los que no pasan _passes_fine_validation().
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

    for _ in range(max_tokens):
        # M5: skip the model only when the state is not wildcard and the
        # exact deterministic filter leaves a single candidate.
        is_wildcard = "*" in state.expected_first_chars()
        if not is_wildcard:
            allowed_check = compute_allowed_ids(state, schema, vocab, trie)
            if not allowed_check:
                break
            if len(allowed_check) == 1:
                best_id = next(iter(allowed_check))
                token_text = vocab.id2decoded.get(best_id)
                if token_text is None:
                    break
                input_ids.append(best_id)
                if not state.update_from_text(token_text):
                    break
                schema.update(state)
                if state.phase is DecoderPhase.COMPLETE:
                    break
                continue

        # Ambiguous step: consult the model exactly once, then search local
        # candidates in descending logit order. Wildcards come here directly
        # without first scanning the full vocabulary in the deterministic path.
        logits = model.get_logits_from_input_ids(input_ids)
        selected: tuple[int, str] | None = None
        for candidate_id, candidate_text in iter_ranked_allowed_ids(
            state, schema, vocab, trie, logits
        ):
            if _passes_fine_validation(
                functions, schema, state, trie, candidate_text
            ):
                selected = candidate_id, candidate_text
                break

        if selected is None:
            break

        best_id, token_text = selected
        input_ids.append(best_id)
        if not state.update_from_text(token_text):
            break
        schema.update(state)

        if state.phase is DecoderPhase.COMPLETE:
            break

    # Solo los tokens GENERADOS (no el prompt): prompt_length se calculó antes
    # del loop sobre los ids reales del prompt (BUG-005). Sin el [0].tolist(),
    # len() contaría las FILAS del tensor 2D y el slice arrastraría tokens.
    generated_ids = input_ids[prompt_length:]
    generated = model.decode(generated_ids)

    return generated, state.phase is DecoderPhase.COMPLETE


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
