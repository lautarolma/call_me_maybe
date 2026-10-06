"""Pydantic models for generated output (I/O layer)."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Any JSON scalar that can appear inside ``parameters``: string, int,
#: float, bool or null (scalars only — nested lists/objects are out of
#: scope). Note bool is a subclass of int in Python, so a strict
#: validator must check bool before int.
JSONValue = str | int | float | bool | None


class FunctionCall(BaseModel):
    """A single function call produced for one input prompt.

    The output side of the system: what the constrained decoder emits per
    prompt and what gets serialized to ``function_calling_results.json``.
    Pydantic buys validation at construction (garbage from the generator
    fails here) and serialization via ``model_dump()``. Field order is
    readability only — it mirrors the subject's example output; JSON
    object key order carries no semantics.
    """

    prompt: str = Field(
        description="Original natural-language request, verbatim from the input file"
    )
    name: str = Field(description="Name of the function to call")
    parameters: dict[str, JSONValue] = Field(
        default_factory=dict,
        description="Function arguments",
    )

    def echo_view(self) -> dict[str, object]:
        """Project the entry to the echo's ``name`` + ``parameters`` view.

        Deliberately not ``model_dump()``: that would also print
        ``prompt``, which the console already showed, doubling every echo
        block. And not the decoder's raw dict either — this runs after
        validation, so what is on screen equals what is persisted,
        post-hoc repairs included.

        Returns:
            A two-key dict, in the order the echo prints it.
        """
        return {"name": self.name, "parameters": self.parameters}
