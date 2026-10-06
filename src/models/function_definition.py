"""Pydantic models for function definitions (I/O layer)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: Allowed JSON scalar types for function parameters.
#: Only scalars are in scope: constrained-decoding nested objects/arrays
#: needs a recursive schema and a much larger state machine (the subject's
#: "complex nested function arguments" bonus turns them on — leave it off
#: until the MVP is green).
#:
#: Why "integer" is here, although JSON has no such type: the moulinette's
#: extractor spells a Python int as "integer" (float -> "number") and the
#: private function set uses it. Without this member ``load_functions``
#: fails Literal validation and the program cannot start on the private
#: half of the evaluation. Domino effect: accepting it here is not enough —
#: every numeric token declares kind "number", so an equality check would
#: accept no token and hang the decoder; ``schema_validator`` therefore
#: compares declared types by compatibility, not equality.
ParameterType = Literal["string", "number", "integer", "boolean", "null"]


class ParameterDef(BaseModel):
    """A single function parameter: name and declared JSON type."""

    # name mirrors the dict key in FunctionDef.parameters, synced by
    # FunctionDef._sync_parameter_names: deliberate denormalization so a
    # parameter object is self-describing when iterated as a list. The
    # default "" matters because the input JSON carries the name as the
    # parent dict's key, not inside the parameter — without it, validation
    # would fail before the syncer runs.
    name: str = Field(default="", description="Parameter name")
    type: ParameterType = Field(
        description=(
            "Parameter type: 'string', 'number', 'integer', 'boolean', 'null'. "
            "'integer' is not a JSON type: it is the moulinette's spelling for "
            "'this parameter is a Python int' (see ParameterType)"
        )
    )


class FunctionDef(BaseModel):
    """A function definition as found in ``functions_definition.json``."""

    name: str = Field(description="Function name, e.g. 'fn_add_numbers'")
    description: str = Field(description="Human-readable description")
    parameters: dict[str, ParameterDef] = Field(
        description="Parameter name -> validated {type: ...} definition"
    )
    returns: dict[str, str] = Field(description="Return type info")

    @field_validator("parameters", mode="after")
    @classmethod
    def _sync_parameter_names(
        cls, params: dict[str, ParameterDef]
    ) -> dict[str, ParameterDef]:
        """Keep each ParameterDef.name in sync with its dict key."""
        # mode="after" runs AFTER pydantic's own validation has turned the
        # raw JSON {"n": {...}} into real ParameterDef instances — which is
        # exactly what this loop needs, since it assigns param.name on
        # those instances. mode="before" would still see plain dicts.
        for key, param in params.items():
            param.name = key
        return params
