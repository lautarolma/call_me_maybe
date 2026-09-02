"""Pipeline orchestration for call_me_maybe (Phase 1 skeleton).

Loads function definitions, prompts and the model vocabulary, initializes
the model, prints a summary and returns a success exit code. Actual
generation is implemented in later phases.
"""

from __future__ import annotations

import argparse

# Import del SDK provisto por la cátedra (vive en llm_sdk/llm_sdk/__init__.py).
# `Small_LLM_Model` envuelve un modelo causal de Hugging Face para inferencia.
from llm_sdk import Small_LLM_Model

from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import load_vocab


def run(args: argparse.Namespace) -> int:
    """Run the Phase 1 pipeline: load inputs + model, print summary.

    Args:
        args: Parsed CLI arguments (``functions_definition``, ``input``,
            ``output`` paths).

    Returns:
        ``0`` on success, ``1`` on failure.

    CÓMO FUNCIONA (por dentro):
    - Es el ORQUESTADOR: no sabe cargar JSONs ni correr tensores, solo
      coordina a los especialistas en orden y propaga excepciones hacia
      `__main__.py`, que es quien las convierte en exit code 1.
    """
    print(f"[1/4] Loading function definitions from {args.functions_definition} ...")
    # load_functions(path) -> list[FunctionDef]:
    #   abre el JSON, lo parsea con json.load, valida cada entrada contra el
    #   modelo pydantic FunctionDef (tipos, campos requeridos) y verifica que
    #   no haya nombres duplicados. Si algo falla, lanza ValueError con
    #   mensaje descriptivo (ver function_loader.py para el detalle).
    functions = load_functions(args.functions_definition)

    print(f"[2/4] Loading prompts from {args.input} ...")
    # load_prompts(path) -> list[str]:
    #   mismo mecanismo de parseo JSON, pero acepta dos formatos: array de
    #   strings planos o array de objetos {"prompt": "..."}. Rechaza listas
    #   vacías (no tiene sentido correr un pipeline sin inputs).
    prompts = load_prompts(args.input)

    print("[3/4] Initializing model (first run downloads weights from the HF Hub) ...")
    # Small_LLM_Model() SIN argumentos usa el default Qwen/Qwen3-0.6B.
    # QUÉ HACE POR DENTRO (llm_sdk):
    #   1. Elige device con prioridad mps > cuda > cpu:
    #      - mps = Metal Performance Shaders (GPU de Apple Silicon).
    #      - cuda = GPU NVIDIA.
    #      - cpu = fallback universal, más lento pero siempre disponible.
    #   2. Elige dtype: float16 en GPU/MPS (mitad de memoria, ~1.2 GB para
    #      600M parámetros vs ~2.4 GB en float32), float32 en CPU por
    #      compatibilidad numérica.
    #   3. AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"): descarga (la
    #      primera vez) o lee del cache (~/.cache/huggingface/hub) los files
    #      del tokenizador (vocab.json, merges.txt, tokenizer.json) y arma
    #      el objeto que traduce texto <-> ids de tokens.
    #   4. AutoModelForCausalLM.from_pretrained(...): ídem pero con los
    #      PESOS del transformer (safetensors). `torch_dtype=self._dtype`
    #      carga los pesos ya en float16 directamente.
    #   5. .eval(): pone el modelo en modo inferencia (desactiva dropout y
    #      otras capas con comportamiento distinto en entrenamiento).
    #   6. requires_grad=False en todos los parámetros: le dice a autograd
    #      de PyTorch que no registre operaciones para calcular gradientes,
    #      lo que ahorra memoria y acelera cada forward pass.
    model = Small_LLM_Model()

    print("[4/4] Building vocabulary index ...")
    # load_vocab(model) -> Vocab:
    #   lee el vocab.json DEL MODELO (via get_path_to_vocab_file, que resuelve
    #   la ruta en el cache de HF) y pre-indexa cada token por su PRIMER
    #   CARÁCTER DECODIFICADO. Ese índice es la base del decoder restringido:
    #   cuando el generador necesite "todos los tokens que empiezan con `"",
    #   responde en O(1) con un set de ids en vez de escanear 150k+ tokens.
    vocab = load_vocab(model)

    print()
    print("=== Phase 1 summary ===")
    print(f"  functions : {len(functions)}")
    print(f"  prompts   : {len(prompts)}")
    print(f"  vocab size: {vocab.vocab_size}")
    print(f"  first-char buckets : {len(vocab.tokens_starting_with)}")
    print(f"  output path: {args.output}")
    print()
    print("All components loaded OK. Generation arrives in Phase 3.")
    return 0
