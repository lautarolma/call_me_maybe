"""Load the model vocabulary and build pre-indexed token structures.

Byte-level BPE vocabularies (GPT-2 / tiktoken style) store raw UTF-8 bytes as
latin-1-ish text: the token ``'Ġthe'`` really represents the text ``' the'``
(``Ġ`` is the byte-encoded space). To make the pre-index match what the
constrained decoder will actually see (the *decoded* text), every token is
decoded once through the SDK's public ``decode`` API at startup and indexed
by its first decoded character.

FONDO TÉCNICO — ¿por qué existe 'Ġ'?:
- Los tokenizadores byte-level BPE NO trabajan sobre caracteres, sino sobre
  BYTES UTF-8 crudos (0-255). Como los bytes arbitrarios son invisibles o no
  imprimibles, GPT-2 ideó un truco: mapear cada byte a un carácter unicode
  "visible" (byte-to-unicode table). El espacio ASCII (0x20) cae en 'Ġ'
  (U+0120, mayúscula g con punto). Por eso el vocabulario guarda 'Ġthe' y no
  ' the': el texto real se recupera invirtiendo esa tabla al decodificar.
- Consecuencia para ESTE proyecto: si indexáramos por la primera letra del
  token CRUDO ('Ġthe' empieza con 'Ġ'), el índice no serviría de nada. Hay
  que indexar por el primer carácter del texto DECODIFICADO (' the' -> ' '),
  que es lo que el decoder restringido realmente compara.
"""

from __future__ import annotations

import json

# `@dataclass` es un decorador que GENERA código boilerplate en tiempo de
# definición de la clase: __init__, __repr__ y __eq__ automáticos a partir
# de las anotaciones de clase. Es azúcar sintáctico de PEP 557.
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model

#: Bucket key for tokens whose text cannot be decoded cleanly (invalid byte
#: sequences, special tokens that decode to empty). These tokens are never
#: reachable while generating JSON text.
BYTE_CATEGORY = "<byte>"


# `slots=True` (Python 3.10+) agrega __slots__ a la dataclass: Python deja de
# crear el __dict__ por instancia y reserva exactamente un slot por campo.
# Resultado: ~50% menos RAM por instancia y acceso a atributos más rápido
# (el atributo se resuelve por offset fijo en memoria, sin lookup de dict).
# El costo: no podés agregar atributos nuevos dinámicamente ni usar
# class-level defaults sin cuidado. Para una estructura de datos fija como
# esta, es puro beneficio.
@dataclass(slots=True)
class Vocab:
    """Pre-indexed vocabulary for constrained decoding.

    Attributes:
        token2id: Token text (vocab.json key) -> token id.
        id2token: Token id -> token text (vocab.json key).
        id2decoded: Token id -> decoded text (via the SDK tokenizer).
        tokens_starting_with: First decoded character -> set of token ids
            whose decoded text starts with that character.
        vocab_size: Total number of tokens in the vocabulary.

    NOTA DE DISEÑO: mantenemos DOS vistas (texto crudo vs decodificado) a
    propósito. La cruda (token2id/id2token) es lo que entiende el tokenizador;
    la decodificada (id2decoded) es lo que entiende el validador de JSON.
    Son dos mundos que conviven y este objeto es la frontera entre ambos.
    """

    token2id: dict[str, int]
    id2token: dict[int, str]
    id2decoded: dict[int, str]
    # ¿Por qué SET de ids y no list? Porque el decoder restringido va a
    # preguntar constantemente "¿este id está permitido acá?" — membership
    # en set es O(1) promedio (hash lookup); en list sería O(n) con n hasta
    # miles. Esa diferencia define si la generación es usable o no.
    tokens_starting_with: dict[str, set[int]]
    vocab_size: int


def load_vocab(model: Small_LLM_Model) -> Vocab:
    """Load the model vocabulary and build pre-indexed structures.

    Args:
        model: Initialized ``Small_LLM_Model`` whose vocabulary file is read
            and whose tokenizer decodes every token text.

    Returns:
        A :class:`Vocab` with token mappings, decoded text per token and the
        first-character pre-index used by the constrained decoder.

    CÓMO FUNCIONA (por dentro):
    - Complejidad total: O(V * D) donde V = tamaño del vocab (~150k para
      Qwen) y D = costo de decodificar 1 token. Se paga UNA vez al inicio
      para que cada paso de generación después sea O(1).
    """
    # get_path_to_vocab_file() (del SDK):
    #   QUÉ: devuelve la ruta absoluta al vocab.json del modelo.
    #   CÓMO: consulta tokenizer.vocab_files_names (dict que cada tokenizador
    #   declara; para Qwen/GPT-2-style es {'vocab_file': 'vocab.json'}) y se
    #   lo pasa a hf_hub_download(repo_id=..., filename=...). Esa función de
    #   huggingface_hub NO descarga siempre: primero busca en el cache local
    #   (~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/...); si el file
    #   ya está (y su hash coincide), devuelve la ruta local; si no, baja el
    #   file del Hub con reintentos. Por eso la segunda ejecución es instantánea.
    vocab_path = model.get_path_to_vocab_file()
    # El vocab.json es literalmente {"token_text": id, ...}: el mapping que
    # usa el tokenizador para partir texto en ids.
    with open(vocab_path, encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    token2id: dict[str, int] = raw_vocab
    # Dict comprehension invertida. Ojo: solo es segura porque el JSON garantiza
    # ids únicos por token; si hubiera colisiones, la última entrada pisaría
    # a las anteriores silenciosamente (comportamiento estándar de dict).
    id2token: dict[int, str] = {token_id: token_text for token_text, token_id in token2id.items()}

    id2decoded: dict[int, str] = {}
    tokens_starting_with: dict[str, set[int]] = {}
    for token_text, token_id in token2id.items():
        try:
            # model.decode([token_id]) (del SDK):
            #   QUÉ: convierte UN id a su texto real ("Ġthe" -> " the").
            #   CÓMO: acepta Tensor O lista (normaliza con .tolist() si hace
            #   falta) y llama a HF tokenizer.decode(ids, skip_special_tokens=True)
            #   — ese flag oculta tokens especiales como <|endoftext|>.
            #   Internamente: id -> string de bytes-mapeados (tabla inversa
            #   del byte-to-unicode) -> join -> decode UTF-8 real.
            decoded = model.decode([token_id])
        except Exception:
            # Defensivo: UN token malformado no debe tirar abajo el arranque
            # entero. Los tokens que explotan acá van al bucket <byte> y quedan
            # fuera del camino de generación (nunca son válidos dentro de JSON).
            decoded = ""
        if not decoded:
            # Tokens especiales (<|endoftext|>, etc.) decodifican a ''.
            # setdefault(key, default): devuelve el valor si la key existe,
            # si no inserta `default` y lo devuelve. Es el idiom get-or-create
            # que evita el doble lookup de "if key not in d: d[key] = ..." .
            # .add(token_id): los sets son mutables; setdefault nos da el MISMO
            # set vivo, así que mutarlo alcanza.
            tokens_starting_with.setdefault(BYTE_CATEGORY, set()).add(token_id)
            continue
        # decoded[0]: PRIMER carácter del texto real. OJO: como los strings de
        # Python iteran por CODE POINT unicode, esto es correcto para el índice
        # aunque el token empiece con un carácter multibyte en UTF-8 (p.ej. 'ñ').
        first_char = decoded[0]
        id2decoded[token_id] = decoded
        tokens_starting_with.setdefault(first_char, set()).add(token_id)

    return Vocab(
        token2id=token2id,
        id2token=id2token,
        id2decoded=id2decoded,
        tokens_starting_with=tokens_starting_with,
        # len(dict) es O(1): los dicts de Python cachean su tamaño.
        vocab_size=len(token2id),
    )
