"""Build the model prompt from validated function definitions.

The prompt is the contract with the model: a small model (Qwen3-0.6B)
does not "understand" functions, it understands text. This layer
translates pydantic-validated ``FunctionDef`` objects into one stable
text block in three parts — system instruction, numbered function
list, user query.

The rationale for the fixed layout is accuracy: if the format varies
between calls, the model must re-discover the pattern every time; a
stable format wastes fewer tokens and gives it nothing to rediscover.
"""

from __future__ import annotations

from src.models.function_definition import FunctionDef

#: Root system instruction: ``{function_list}`` is a ``str.format()``
#: placeholder (not an f-string — the list is only built per call);
#: adjacent literals keep each physical line under flake8's 120 chars.
SYSTEM_PROMPT = (
    "You are a function calling assistant. Given the user's query, you must "
    "output a JSON object that calls the most appropriate function.\n\n"
    "Available functions:\n\n"
    "{function_list}\n\n"
    'Output ONLY a JSON object with "name" and "parameters" fields. No explanation.'
)


def build_function_list(functions: list[FunctionDef]) -> str:
    """Build a numbered list of function descriptions for the prompt.

    Each entry is order number, name, description and parameters with
    their type in parentheses. The type words — "number", "string",
    "boolean", "null" — are spelled exactly as the constrained decoder
    spells them: model and decoder must share one type vocabulary.

    Args:
        functions: FunctionDef list, already validated by the loader.

    Returns:
        One entry per function separated by a blank line (``\\n\\n``);
        an empty list yields an empty string.
    """
    parts = []
    for index, fn in enumerate(functions, start=1):
        params = ", ".join(
            f"{pname} ({pinfo.type})" for pname, pinfo in fn.parameters.items()
        )
        parts.append(f"{index}. {fn.name}: {fn.description}\n   Parameters: {params}")
    return "\n\n".join(parts)


def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Build the complete prompt for a single user query.

    Interpolates the function list into ``SYSTEM_PROMPT`` and appends
    the user query. The model only ever sees this exact layout — never
    a hand-assembled free-form string.

    Args:
        functions: FunctionDef list to expose in the prompt.
        user_prompt: The user's phrase, verbatim from the tests JSON.

    Returns:
        The full prompt, ready for ``model.encode(prompt)``.
    """
    function_list = build_function_list(functions)
    return SYSTEM_PROMPT.format(function_list=function_list) + f"\nUser query: {user_prompt}"
