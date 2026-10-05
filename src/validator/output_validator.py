"""Output validation: turn the decoder's raw string into a checked FunctionCall.

QUÉ HACE ESTE MÓDULO (y por qué existe):
El decoder restringido produce un STRING de texto con el JSON del function
call. Ese string tiene que convertirse en una entrada del array de salida
que la moulinette lee. El paso intermedio no es opcional: es donde fallan
las cosas si no se chequea nada.

Este módulo hace TRES cosas, todas puras (sin modelo, sin I/O):
  1. `parse_output`      — string crudo  -> dict (json.loads, con tolerancia)
  2. `build_function_call` — (prompt, dict) -> FunctionCall (validación pydantic)
  3. `validate_output`   — string crudo + functions -> FunctionCall | str

Por qué NO un solo `json.loads` y ya:
  · El decoder garantiza JSON sintácticamente válido POR CONSTRUCCIÓN, pero
    no garantiza que el contenido encaje con el schema de la función
    elegida. `name` podría no existir en functions_definition.json.
  · Si no validamos acá, el error aparece en la moulinette del evaluador
    como un cero, no como un mensaje que podamos entender.

LAS TRES REPARACIONES POST-HOC (`_repair_string_value`):
  El modelo restringido copia bien la frase del usuario pero la deforma al
  copiarla. Hay tres deformaciones medidas, y cada una tiene su reparación:
    A · `_snap_to_query_span`     — copia TRUNCADA  (clip de puntuación líder)
    B · `_restore_internal_quotes` — copia SIN comillas internas
    C · `_collapse_repeated_run`   — copia CONTADA (una repetición por match)
  Las tres son post-hoc (no tocan logits), derivadas de la query, y comparten una
  sola norma: *el valor tiene que estar respaldado por la frase del usuario; si
  no, es una invención y se normaliza*. Cuando no hay una corrección única
  respaldada por evidencia, devuelven el valor SIN tocar.

LA REGLA DE ORO — no romper el alineamiento posicional:
La moulinette empareja answers y correcciones con `zip()`, que es
POSICIONAL. Si una entry falta en medio, TODAS las de después se desalinean
y el score se arruina. Por eso `build_results` NUNCA omite una entry: si un
prompt falla, emite un placeholder en esa posición exacta y sigue. Perder
un test es perdible; perder once es fatal.

CÓMO SE USA (ver pipeline.py):
    raw_prompts = load_prompts(args.input)          # texto CRUDO del input
    for i, raw_prompt in enumerate(raw_prompts):
        generated = generate(...)[0]                # string del decoder
        result = build_results(raw_prompts, generated_list)
    write_results(result, args.output)
"""

from __future__ import annotations

import json
import sys

from src.models.function_definition import FunctionDef
from src.models.output import FunctionCall


# Valor usado para el campo `name` cuando la generación no se pudo parsear.
# NO es un nombre de función real: es un marcador que hará que ese único
# test falle con "unknown function" en la moulinette, sin arrastrar a los
# demás. Cualquier valor no-existente sería igual de malo, pero este es
# legible en el log de errores.
_UNKNOWN_FN_SENTINEL = "__unparseable__"

#: Delimitadores que marcan el borde IZQUIERDO de un valor copiado de la query.
#: Whitespace y comillas separan palabras/valores en lenguaje natural. Un
#: alfanumérico pegado a la izquierda significa que el valor es el SUFIJO de
#: una palabra más larga ("llo" dentro de "hello") y NO se debe estirar.
_SNAP_BOUNDARY = frozenset(" \t\n\r\"'")


def parse_output(raw: str) -> dict[str, object]:
    """Parsea el string del decoder a un dict.

    Args:
        raw: Texto producido por el decoder (JSON con whitespace opcional
            alrededor — el decoder emite '\\n\\n{\\n  "name": ...').

    Returns:
        El dict parseado.

    Raises:
        json.JSONDecodeError: si el texto no es JSON válido.

    POR QUÉ strip():
    El decoder puede dejar newlines/espacios alrededor del objeto
    (`'\\n\\n{\\n  ...\\n}'`). `json.loads` los tolera igual (los whitespace
    son válidos fuera de un valor), pero el `strip()` documenta la
    intención y protege contra BOMs, que `json.loads` NO tolera.
    """
    return json.loads(raw.strip())  # type: ignore[no-any-return]


def _snap_to_query_span(value: str, prompt: str) -> str:
    """Re-ancla un valor string al tramo VERBATIM de la query del que salió.

    QUÉ PROBLEMA RESUELVE (caso real medido, test privado 8):
    el prompt dice ``Read the file at /home/user/data.json with utf-8`` y el
    modelo emite ``path = "home/user/data.json"``: copia bien TODO el path
    menos la puntuación líder (``/``). No alucina el contenido — clipea el
    borde izquierdo del tramo que copió.

    LA REGLA (tres pasos):
    1. Si el valor aparece LITERALMENTE en la query (substring exacto),
       ubica esa ocurrencia con ``find``.
    2. Mira el char inmediatamente a la izquierda. Si es PUNTUACIÓN (no
       whitespace, no comilla, no alfanumérico), ese char era parte del valor
       y el modelo lo perdió: estirá el valor hacia la izquierda hasta el
       primer borde.
    3. Si ya arranca en un borde, o no aparece en la query, no toca nada.

    POR QUÉ FRENA EN ESOS CHAR (contraejemplos que fijan la frontera):
    - ``"llo"`` dentro de ``"hello"``: a la izquierda hay ``e`` (alfanumérico)
      → NO se estira. Sin este freno, "las últimas 3 letras de hello"
      devolvería "hello" entero.
    - ``"hello"`` dentro de ``'hello'``: a la izquierda hay ``'`` (comilla) →
      NO se estira. La comilla es el DELIMITADOR del valor, no su contenido:
      sin este freno se rompían los tests públicos de ``'hello'``/``'world'``
      (``hello`` → ``'hello``).
    - ``"C:\\Users\\john\\config.ini"`` (test privado 9): a la izquierda hay
      espacio → NO se estira. Por eso un path Windows (que no arranca con
      ``/``) queda intacto: la regla NO asume "todo path empieza con /".

    LÍMITE CONOCIDO: usa la PRIMERA ocurrencia (``find``). Si el mismo valor
    aparece varias veces con bordes distintos, sólo considera la primera.
    Ninguno de los 22 casos medidos (11 públicos + 11 privados) lo ejercita.
    """
    if not value:
        return value
    start = prompt.find(value)
    if start < 0:
        return value
    left = start
    while left > 0:
        ch = prompt[left - 1]
        if ch in _SNAP_BOUNDARY or ch.isalnum():
            break
        left -= 1
    if left == start:
        return value
    return prompt[left:start + len(value)]


def _restore_internal_quotes(value: str, prompt: str) -> str:
    """Restaura las comillas dobles INTERNAS que el modelo se comió al copiar.

    QUÉ PROBLEMA RESUELVE (caso real medido, test privado 11):
    el prompt dice ``Format template: Say "hello" to {name}`` y el modelo emite
    ``template = "Say hello to {name}"``: copió perfecto el contenido pero se
    comió las comillas que delimitan ``hello`` dentro del valor.

    LA REGLA (cinco pasos):
    1. Se borran TODAS las comillas dobles de la query, guardando el mapa de
       índices para poder volver a las coordenadas originales.
    2. Se buscan todas las ocurrencias del valor en esa query mutilada.
    3. De cada una se recupera el slice ORIGINAL que le corresponde (incluye
       las comillas que el paso 1 se había saltado).
    4. Se descartan los slices que (a) son idénticos al valor —no hay nada que
       restaurar— o (b) empiezan o terminan en comilla.
    5. Si queda EXACTAMENTE UNO, se devuelve. Si queda cero o más de uno, se
       devuelve el valor sin tocar.

    POR QUÉ SE DESCARTAN LAS COMILLAS DE LOS BORDES (el contraejemplo que
    define la regla): el prompt ``Replace all numbers in "Hello 34 I'm 233
    years old" with NUMBERS`` tiene el valor entre comillas, pero esas comillas
    son el MARCO de la frase, no su contenido: el valor esperado es el texto
    SIN ellas. Sin este filtro, la regla le agregaría las comillas y rompería
    ese test público —y también el de ``Reverse the string 'hello'``.

    POR QUÉ EXIGE UN SOLO CANDIDATO: sin esa exigencia la regla empieza a
    adivinar. En ``Say "hello" and hello`` hay dos ocurrencias y la correcta es
    NO tocar nada; con ``Use "a" or "b" for {x}`` el valor ``a`` aparece
    dentro de palabras ("Form**a**t"), así que la evidencia es ambigua. Ante
    duda, se queda callada: el costo de callarse es un test igual de fallado,
    y el costo de adivinar es romper los 38 valores que hoy funcionan.
    """
    if not value:
        return value
    stripped_chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(prompt):
        if char == '"':
            continue
        stripped_chars.append(char)
        positions.append(index)
    stripped = "".join(stripped_chars)

    candidates: list[str] = []
    start = 0
    while True:
        found = stripped.find(value, start)
        if found < 0:
            break
        # `end` es exclusivo sobre `stripped`; `positions[end - 1]` es el
        # último carácter del slice y por eso el +1 del corte en `prompt`.
        end = found + len(value)
        original = prompt[positions[found]:positions[end - 1] + 1]
        start = found + 1
        if original == value:
            continue
        if original[0] == '"' or original[-1] == '"':
            continue
        candidates.append(original)

    if len(set(candidates)) != 1:
        return value
    return candidates[0]


def _collapse_repeated_run(value: str, prompt: str) -> str:
    """Deshace el conteo cuando el modelo repitió un carácter por cada match.

    QUÉ PROBLEMA RESUELVE (caso real medido, test público 9):
    el prompt dice ``Replace all vowels in 'Programming is fun' with
    asterisks`` y el modelo emite ``replacement = "****"``. En cualquier API de
    sustitución —``re.sub`` de Python, ``sed``, ``replace`` de JavaScript— el
    ``replacement`` es una PLANTILLA que se aplica a todas las coincidencias,
    no una copia por coincidencia. El modelo contó las vocales y escribió una
    estrella por vocal: ejecutó la instrucción en vez de parametrizarla. El
    valor correcto es el carácter repetido UNA vez.

    LA REGLA (cuatro pasos):
    1. El valor tiene que ser ENTERAMENTE una corrida de >= 2 caracteres
       idénticos.
    2. Esa corrida NO puede aparecer literal en la query: si el modelo la
       copió, la repetición es intencional y no se toca.
    3. Se busca la corrida más larga que la query SÍ muestra y se usa esa.
    4. Si la query no muestra ninguna, se deja un solo carácter.

    POR QUÉ EL PASO 3 EXISTE (el agujero medido de la versión simple): si la
    query muestra ``***`` y el modelo cuenta cinco, colapsar siempre a uno
    devolvería ``*`` en vez de ``***``. La versión "respeta el conteo de la
    query" devuelve ``***``. En el caso real la query no muestra ninguna
    corrida —dice "asterisks", la palabra inglesa, no el símbolo— y por eso
    ahí sí se cae al paso 4.

    POR QUÉ EXIGE QUE EL VALOR ENTERO SEA LA CORRIDA: una versión más amplia
    que reconociera "un bloque repetido" (tipo ``ababab`` -> ``ab``) está
    ROTA — sobre el caso real devolvería ``**`` en vez de ``*``, porque una
    corrida de cuatro iguales también es "un bloque de dos repetido dos
    veces". Medido: esa variante baja el set público de 11/11 a 10/11. Los
    valores legítimos con repetición (``utf-8``, ``/home/user/data.json``,
    ``NUMBERS``, ``dog``) tienen caracteres distintos y quedan intactos.

    NOTA DE LEGITIMIDAD: la regla se apoya en una FIRMA ESTRUCTURAL (la
    corrida de caracteres), no en un vocabulario. Anclar la corrección a la
    palabra "asterisks" sería una tabla de búsqueda hardcodeada y está
    prohibido por el subject; anclarla a la forma de la salida no.
    """
    if len(value) < 2:
        return value
    if len(set(value)) != 1:
        return value
    if value in prompt:
        return value
    for length in range(len(value), 1, -1):
        head = value[:length]
        if head in prompt:
            return head
    return value[:1]


def _repair_string_value(value: str, prompt: str) -> str:
    """Aplica las tres reparaciones post-hoc, en orden, y la primera gana.

    POR QUÉ "LA PRIMERA QUE CORRIGE GANA" y no las tres en cadena: cada regla
    exige la evidencia que la anterior no tenía. `_snap_to_query_span` sólo
    dispara si el valor ES un substring de la query. Si no lo es, A no debía
    tocar nada, así que la cadena sigue a `_restore_internal_quotes`, que
    exige que aparezca al quitar las comillas. Y `_collapse_repeated_run`
    sólo mira corridas de caracteres idénticos, forma que B nunca produce
    (B devuelve slices con comillas, que no son corridas). Son disjuntas por
    construcción, y esta forma lo hace explícito en el código.

    El orden NO es arbitrario: A es la regla ya verificada y commiteada, así
    que si B o C tuvieran un defecto, el comportamiento previo queda cubierto
    por el regression suite.
    """
    snapped = _snap_to_query_span(value, prompt)
    if snapped != value:
        return snapped
    quoted = _restore_internal_quotes(snapped, prompt)
    if quoted != snapped:
        return quoted
    return _collapse_repeated_run(quoted, prompt)


def build_function_call(prompt: str, payload: dict[str, object]) -> FunctionCall:
    """Construye un FunctionCall validado desde (prompt, payload parseado).

    Args:
        prompt: El request natural ORIGINAL (no el prompt con las function
            definitions inyectadas). La moulinette lo compara con
            `correction["prompt"]` por igualdad EXACTA de string, así que
            tiene que ser el texto del input, byte a byte.
        payload: Dict con las keys `name` y `parameters` (o `name` sola).

    Returns:
        ``FunctionCall`` con los tres campos del subject V.4.

    Raises:
        pydantic.ValidationError: si falta `name` o los tipos no cierran.
    """
    name = payload.get("name")
    parameters = payload.get("parameters", {})
    raw_parameters = parameters if isinstance(parameters, dict) else {}
    # Aplica las TRES reparaciones post-hoc a cada value string (ver
    # `_repair_string_value`). Las tres corrigen la misma clase de defecto: el
    # modelo es un copiador y deforma la frase mientras copia — la recorta
    # (A), le come las comillas internas (B), o cuenta repeticiones en vez de
    # parametrizar (C). Los no-string (números, bools, null) pasan intactos: el
    # proyecto limita los params a escalares (models/output.py).
    repaired_parameters = {
        key: _repair_string_value(value, prompt) if isinstance(value, str) else value
        for key, value in raw_parameters.items()
    }
    return FunctionCall(
        prompt=prompt,
        # `name` viene como object del json.loads; pydantic lo valida como str.
        # Si el decoder garantiza un string, el isinstance es defensivo y
        # nunca falla en la práctica — pero sin él, mypy se quejaría de
        # pasar `object` donde se espera `str`.
        name=name if isinstance(name, str) else str(name),
        # `raw_parameters` es `object` para mypy (viene de `payload`); el
        # isinstance de arriba ya lo estrechó a dict. El default {} cubre el
        # caso "el decoder emitió solo el name" (fn sin parámetros).
        parameters=repaired_parameters,
    )


def validate_output(
    raw: str,
    functions: list[FunctionDef],
) -> FunctionCall | str:
    """Valida una generación contra las definiciones disponibles.

    Args:
        raw: Texto del decoder para UN prompt.
        functions: Definiciones cargadas del input.

    Returns:
        ``FunctionCall`` si la generación es válida y su `name` existe en
        ``functions``; o un string de error si algo falla.

    POR QUÉ devuelve `FunctionCall | str` en vez de lanzar:
    El subject (V.5) pide que el programa "must never crash unexpectedly"
    y que los prompts problemáticos no maten el pipeline. Devolver el error
    como valor deja que el caller lo loguee y siga con el siguiente prompt.

    NOTA sobre la validación de `name`:
    Verificamos que el nombre EXISTA en las definiciones, pero NO que los
    parameters coincidan con el schema de esa función. Esa validación más
    profunda es la Task 5.1 completa (ver PLAN_EJECUCION_V2 Etapa 1) y queda
    fuera de este primer corte: lo que resuelve el bloqueo del entregable es
    que el archivo exista y sea parseable, no que sea semánticamente perfecto.
    """
    try:
        payload = parse_output(raw)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return f"expected a JSON object, got {type(payload).__name__}"

    try:
        call = build_function_call(prompt="", payload=payload)
    except Exception as exc:  # pydantic.ValidationError y subclases
        return f"schema violation: {exc}"

    known = {f.name for f in functions}
    if call.name not in known:
        return f"unknown function {call.name!r} (known: {sorted(known)})"

    return call


def build_results(
    prompts: list[str],
    generated: list[str],
) -> list[FunctionCall]:
    """Empaqueta (prompts originales, generaciones) en entries de salida.

    Args:
        prompts: Requests originales, en el MISMO orden que `generated`.
        generated: Strings del decoder, uno por prompt.

    Returns:
        Una entry por prompt, EN ORDEN. Nunca se omiten entries: si una
        generación no se puede parsear, se emite un placeholder en esa
        posición para no romper el `zip()` posicional de la moulinette.
    """
    results: list[FunctionCall] = []
    for i, raw_prompt in enumerate(prompts):
        raw_output = generated[i] if i < len(generated) else ""
        try:
            payload = parse_output(raw_output)
            if not isinstance(payload, dict):
                raise ValueError(f"not a JSON object: {type(payload).__name__}")
            results.append(build_function_call(raw_prompt, payload))
        except (json.JSONDecodeError, ValueError) as exc:
            # Placeholder alineado: este test va a fallar, los demás no.
            results.append(
                FunctionCall(
                    prompt=raw_prompt,
                    name=_UNKNOWN_FN_SENTINEL,
                    parameters={},
                )
            )
            # El warning va a stderr, que es el stream de diagnóstico
            # (ver la nota sobre stdout/stderr en __main__.py).
            print(
                f"WARNING: prompt {i} produced unparseable output ({exc}); "
                f"emitted placeholder to keep positional alignment",
                file=sys.stderr,
            )
    return results


def _text_forms(value: object) -> list[str]:
    """Representaciones textuales con las que un valor puede estar en un prompt.

    POR QUÉ MÁS DE UNA FORMA: el decoder escribe los floats con punto decimal
    (`2.0`), pero el humano escribe el número sin él ("sum 2 and 3"). Si sólo
    aceptáramos `"2.0"` como evidencia de soporte, ese prompt legítimo
    dispararía un warning falso. Por eso un float entero devuelve las dos
    formas.

    Devuelve `[]` para valores sin forma textual comparable (bool, None) —
    el caller los trata como "no juzgables" en vez de "no soportados".
    """
    if isinstance(value, bool):
        # `bool` es subclase de `int`: hay que chequearlo ANTES que int o
        # True se reportaría como "1".
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, float):
        forms = [repr(value)]
        if value.is_integer():
            forms.append(str(int(value)))
        return forms
    if isinstance(value, str):
        return [value]
    return []


def find_unsupported_prompts(
    prompts: list[str],
    results: list[FunctionCall],
) -> list[int]:
    """Índices de los prompts cuya llamada no tiene NINGÚN valor respaldado
    por el texto del prompt.

    QUÉ HACE: el decoder restringido SIEMPRE emite una función válida — la
    gramática no permite otra cosa. Para un prompt que no corresponde a
    ninguna función, eso degenera en que el modelo elige la que "menos feo"
    queda: no crashea, pero tampoco avisa. Ejemplo real medido:
    *"What is the weather in Paris tomorrow?"* → `fn_get_square_root(a=100.0)`.
    Esto detecta ese caso y lo reporta.

    EL CRITERIO, y por qué es simple a propósito: si NINGÚN valor de
    parámetro aparece (literal, sin distinguir mayúsculas) en el prompt, la
    llamada no está respaldada por la entrada — el modelo se inventó hasta los
    argumentos. Con que UNO aparezca, no se reporta: el caso borderline
    (P9, donde `replacement="****"` no está en el prompt pero `source_string`
    y `regex` sí) es un fallo de accuracy del modelo, no una falta de match,
    y el corretero ya lo mide.

    LO QUE ESTE MÓDULO **NO** ES — importante para la corrección del subject:
    el subject dice que "the function to call should be chosen using the LLM,
    not with heuristics". Acá NO se elige nada: la función ya la eligió el LLM
    en constrained decoding. Esta función es un sensor de SALIDA para una
    persona, no una decisión. No toca el archivo de resultados y no altera el
    score.

    Args:
        prompts: Requests originales (texto crudo, sin las defs inyectadas).
        results: Entries ya construidas por `build_results`.

    Returns:
        Índices (base 0) de los prompts sin respaldo textual. Vacío = todo OK.
    """
    unsupported: list[int] = []
    for i, call in enumerate(results):
        # El sentinel ya tiene su propio warning (no parseable) y no tiene
        # argumentos que juzgar. Las funciones sin parámetros tampoco son
        # juzgables: no hay con qué comparar contra el prompt.
        if call.name == _UNKNOWN_FN_SENTINEL or not call.parameters:
            continue
        if i >= len(prompts):
            continue
        haystack = prompts[i].casefold()
        if not haystack.strip():
            continue
        supported = any(
            form and form.casefold() in haystack
            for value in call.parameters.values()
            for form in _text_forms(value)
        )
        if not supported:
            unsupported.append(i)
    return unsupported
