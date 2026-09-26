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
    return FunctionCall(
        prompt=prompt,
        # `name` viene como object del json.loads; pydantic lo valida como str.
        # Si el decoder garantiza un string, el isinstance es defensivo y
        # nunca falla en la práctica — pero sin él, mypy se quejaría de
        # pasar `object` donde se espera `str`.
        name=name if isinstance(name, str) else str(name),
        # Mismo motivo: `parameters` es `object` para mypy, y el campo del
        # modelo es `dict[str, JSONValue]`. El default {} cubre el caso
        # "el decoder emitió solo el name" (fn sin parámetros).
        parameters=parameters if isinstance(parameters, dict) else {},
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
