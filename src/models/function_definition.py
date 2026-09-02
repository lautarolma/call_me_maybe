"""Pydantic models for function definitions (I/O layer)."""

from __future__ import annotations

# `Literal` (PEP 586, typing): restringe un valor a un conjunto CERRADO de
# literales exactos. A diferencia de `str` (acepta cualquier string),
# Literal["string", "number"] solo acepta esos 4 valores EXACTOS.
# Doble función: documentación para humanos + chequeo estático (mypy) +
# validación en RUNTIME cuando lo usa pydantic. Si el JSON trae "integer",
# la validación explota con un mensaje que lista los valores válidos.
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: Allowed JSON types for function parameters.
ParameterType = Literal["string", "number", "boolean", "null"]


class ParameterDef(BaseModel):
    """A single function parameter.

    QUÉ ES BaseModel (por dentro):
    - Cuando definís una clase que hereda de BaseModel, la metaclass de
      pydantic inspecciona las anotaciones en tiempo de DEFINICIÓN y construye
      un schema interno (core schema en Rust en pydantic v2). Con eso genera
      un __init__ que VALIDA Y COERCIONA cada argumento: si pasa `"type": 42`
      a ParameterDef, no llega a asignarse — lanza ValidationError.
    - Los modelos son (por default) inmutables-ish: validar después de crear
      requiere model_copy o desactivar validate_assignment.
    """

    # ``name`` mirrors the dict key in ``FunctionDef.parameters`` and is kept
    # in sync by ``FunctionDef._sync_parameter_names`` after validation.
    #
    # ¿Por qué existe este campo si ya es la key del dict? DENORMALIZACIÓN
    # deliberada: al recorrer luego los parámetros como lista de objetos,
    # cada objeto se describe completo sin necesitar contexto externo
    # ("¿cómo me llamo?"). El costo es mantener la copia sincronizada — y
    # ese trabajo lo hace el validator de FunctionDef, no el consumidor.
    #
    # Field() es el configurador de campos de pydantic: default, alias,
    # constraints (ge/le/min_length...), description (que termina en el JSON
    # schema generado). Acá: default="" porque el JSON de entrada NO trae
    # "name" dentro del parameter (viene como key del dict padre); sin el
    # default, la validación fallaría antes de llegar al sincronizador.
    name: str = Field(default="", description="Parameter name")
    type: ParameterType = Field(
        description="Parameter type: 'string', 'number', 'boolean', 'null'"
    )


class FunctionDef(BaseModel):
    """A function definition as found in functions_definition.json."""

    name: str = Field(description="Function name, e.g. 'fn_add_numbers'")
    description: str = Field(description="Human-readable description")
    # dict[str, ParameterDef]: pydantic valida RECURSIVAMENTE. Cada valor del
    # dict se valida contra el modelo ParameterDef completo. Un dict anidado
    # malformado reporta el error con la ruta exacta (parameters -> altura).
    parameters: dict[str, ParameterDef] = Field(
        description="Parameter name -> validated {type: ...} definition"
    )
    returns: dict[str, str] = Field(description="Return type info")

    # @field_validator("parameters", mode="after"):
    #   Registra este método como validador del campo "parameters".
    #   mode="after" significa: se ejecuta DESPUÉS de que pydantic validó y
    #   convirtió el valor crudo (dict de dicts -> dict de ParameterDef).
    #   Por eso la firma recibe dict[str, ParameterDef] ya tipado y podemos
    #   mutar objetos ParameterDef reales, no dicts crudos.
    #   (mode="before" recibiría el JSON crudo y serviría para pre-procesar.)
    #
    # @classmethod es OBLIGATORIO en la API de pydantic v2 para validators:
    # el validador se invoca sobre la clase (cls) porque puede correr antes
    # de que exista la instancia.
    @field_validator("parameters", mode="after")
    @classmethod
    def _sync_parameter_names(
        cls, params: dict[str, ParameterDef]
    ) -> dict[str, ParameterDef]:
        """Keep each ParameterDef.name in sync with its dict key."""
        # items() devuelve pares (key, value) del dict; asignamos param.name
        # pisando el default "". Como el validador corre dentro del proceso
        # de construcción del modelo, NINGÚN consumidor puede ver el estado
        # dessincronizado: o el modelo se construye bien, o no se construye.
        for key, param in params.items():
            param.name = key
        return params
