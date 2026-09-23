"""Build the model prompt from validated function definitions.

El prompt es EL CONTRATO entre nosotros y el modelo. Un modelo chico
(Qwen3-0.6B) no "entiende" funciones: lo que entiende es texto. Esta capa
traduce las FunctionDef (que ya vinieron validadas por pydantic) a un bloque
de texto estable y reproducible, en 3 partes:

    1. Instrucción del sistema  -> "sos un asistente de function calling"
    2. Lista de funciones       -> build_function_list() (numerada, con tipos)
    3. Query del usuario        -> "User query: ..."

Por qué esto importa (accuracy): si el formato de la lista varía entre
llamadas, el modelo tiene que re-descubrir el patrón cada vez. Formato
fijo = menos tokens desperdiciados = mejor accuracy con modelo chico.
"""

from __future__ import annotations

from src.models.function_definition import FunctionDef

#: Instrucción raíz del sistema. El marcador ``{function_list}`` es un
#: placeholder que se rellena con ``str.format()``; NO usar f-string acá,
#: porque el contenido de la lista se genera recién por llamada.
#: Se arma con concatenación implícita de strings adyacentes (PEP 8): el
#: resultado en runtime es UN único string, pero cada línea física queda
#: dentro del límite de 120 chars que exige flake8.
SYSTEM_PROMPT = (
    "You are a function calling assistant. Given the user's query, you must "
    "output a JSON object that calls the most appropriate function.\n\n"
    "Available functions:\n\n"
    "{function_list}\n\n"
    'Output ONLY a JSON object with "name" and "parameters" fields. No explanation.'
)


def build_function_list(functions: list[FunctionDef]) -> str:
    """Build a numbered list of function descriptions for the prompt.

    Cada función se describe con: número de orden, nombre, descripción y
    sus parámetros con tipo entre paréntesis. El formato ES parte del prompt:
    "number", "string", "boolean", "null" (los tipos del subject) escritos
    EXACTAMENTE como los usa el constrained decoder después — el modelo y el
    decoder deben hablar el mismo vocabulario de tipos.

    Args:
        functions: Lista de FunctionDef ya validada por el loader.

    Returns:
        String con una entrada por función, separadas por línea en blanco
        (``\n\n``), lista vacía -> string vacío.
    """
    parts = []
    for index, fn in enumerate(functions, start=1):
        # fn.parameters es dict[str, ParameterDef] (una vez que pydantic
        # validó en el loader). Accedemos a pinfo.type por ATRIBUTO, no por
        # pinfo["type"]: para un BaseModel el acceso por atributo es la vía
        # natural y evita ambigüedades del __getitem__ de pydantic.
        params = ", ".join(
            f"{pname} ({pinfo.type})" for pname, pinfo in fn.parameters.items()
        )
        parts.append(f"{index}. {fn.name}: {fn.description}\n   Parameters: {params}")
    return "\n\n".join(parts)


def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Build the complete prompt for a single user query.

    Arma el prompt final: SYSTEM_PROMPT con la lista de funciones
    interpolada, seguido de la query del usuario. El modelo solo ve SIEMPRE
    este formato — nunca texto libre armado a mano.

    Args:
        functions: Lista de FunctionDef a exponer en el prompt.
        user_prompt: La frase del usuario, tal cual viene del JSON de tests.

    Returns:
        El prompt completo listo para ``model.encode(prompt)``.
    """
    function_list = build_function_list(functions)
    return SYSTEM_PROMPT.format(function_list=function_list) + f"\nUser query: {user_prompt}"
