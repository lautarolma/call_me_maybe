"""Pydantic models for generated output (I/O layer)."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Any JSON value that can appear inside the ``parameters`` object.
#
# Union type con sintaxis PEP 604 (Python 3.10+): equivale a
# Union[str, int, float, bool, None] de typing. Describe EXACTAMENTE el
# conjunto de valores que JSON puede representar como escalar (JSON no tiene
# tipos fecha/tupla/etc.: todo son estos 5 + arrays + objects).
# Nota de tipado: en Python `bool` es subclase de `int`, así que un validador
# estricto que distinga True de 1 debe chequear bool ANTES que int.
JSONValue = str | int | float | bool | None


class FunctionCall(BaseModel):
    """A single function call produced for one input prompt.

    Este modelo representa la SALIDA del sistema: lo que el decoder
    restringido va a producir por cada prompt y que después se serializa a
    function_calls.json. Al modelarlo con pydantic ganamos gratis:
    - validación al construir (si el generador produce basura, explota acá)
    - serialización a dict/JSON con .model_dump() / .model_dump_json()
    """

    name: str = Field(description="Name of the function to call")

    # ¿Por qué default_factory=dict y NO parameters: dict = {}? Porque los
    # defaults mutables se evalúan UNA vez (al definir la clase) y serían
    # COMPARTIDOS entre todas las instancias: mutar el dict de una "instancia"
    # mutaría el de todas — bug clásico de Python (el famoso mutable default).
    # default_factory recibe el callable y lo LLAMA en cada construcción,
    # produciendo un dict fresco por instancia.
    #
    # El tipo del value es JSONValue (la union de arriba): un parámetro puede
    # valer "pepe", 42, 3.14, true o null, pero nunca una lista anidada ni
    # otro objeto — el scope del proyecto limita parámetros a escalares.
    parameters: dict[str, JSONValue] = Field(
        default_factory=dict,
        description="Function arguments",
    )
