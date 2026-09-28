"""Pipeline orchestration for call_me_maybe.

Loads function definitions, prompts and the model vocabulary, initializes
the model, runs constrained generation over every prompt, validates each
result and persists the output JSON required by the subject (V.4).
"""

from __future__ import annotations
import json
from pathlib import Path

from src.models.output import FunctionCall
from src.prompt.prompt_builder import build_prompt
from src.utils.metrics import (
    MetricsRun,
    measure_time,
    report_prompt_metrics,
    track_prompt,
)
from src.decoder.constrained_generator import generate
from src.decoder.trie import build_trie
from src.validator import build_results

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
    print(f"[1/5] Loading function definitions from {args.functions_definition} ...")
    # load_functions(path) -> list[FunctionDef]:
    #   abre el JSON, lo parsea con json.load, valida cada entrada contra el
    #   modelo pydantic FunctionDef (tipos, campos requeridos) y verifica que
    #   no haya nombres duplicados. Si algo falla, lanza ValueError con
    #   mensaje descriptivo (ver function_loader.py para el detalle).
    functions = load_functions(args.functions_definition)

    print(f"[2/5] Loading prompts from {args.input} ...")
    # load_prompts(path) -> list[str]:
    #   mismo mecanismo de parseo JSON, pero acepta dos formatos: array de
    #   strings planos o array de objetos {"prompt": "..."}. Rechaza listas
    #   vacías (no tiene sentido correr un pipeline sin inputs).
    #
    # POR QUÉ GUARDAMOS LAS DOS LISTAS: `build_prompt` inyecta las function
    # definitions y produce el texto que ve el MODELO. El campo `prompt` de
    # la salida tiene que ser el request ORIGINAL, sin las definitions —
    # la moulinette lo compara con `correction["prompt"]` por igualdad
    # EXACTA de string. Si guardáramos solo los prompts ya construidos,
    # escribiríamos en el output el prompt con las definitions inyectadas, y
    # la comparación fallaría en los 11 tests.
    raw_prompts = load_prompts(args.input)
    prompts = [build_prompt(functions, prompt) for prompt in raw_prompts]

    print("[3/5] Initializing model (first run downloads weights from the HF Hub) ...")
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

    print("[4/5] Building vocabulary index ...")
    # load_vocab(model) -> Vocab:
    #   lee el vocab.json DEL MODELO (via get_path_to_vocab_file, que resuelve
    #   la ruta en el cache de HF) y pre-indexa cada token por su PRIMER
    #   CARÁCTER DECODIFICADO. Ese índice es la base del decoder restringido:
    #   cuando el generador necesite "todos los tokens que empiezan con `"",
    #   responde en O(1) con un set de ids en vez de escanear 150k+ tokens.
    vocab = load_vocab(model)
    print("[5/5] Building trie ...")
    trie_node = build_trie([function.name for function in functions])

    print()
    print("=== Phase 1 summary ===")
    print(f"  functions : {len(functions)}")
    print(f"  prompts   : {len(prompts)}")
    print(f"  trie_node : {len(trie_node.children)}")

    print(f"  vocab size: {vocab.vocab_size}")
    print(f"  first-char buckets : {len(vocab.tokens_starting_with)}")
    print(f"  output path: {args.output}")
    print()
    print("All components loaded OK. Starting constrained generation.")

    # Los resultados se acumulan para imprimirlos fuera del loop: el ciclo de
    # generación queda libre de prints y mediciones directas (eso vive en
    # src/utils/metrics.py).
    generated: list[str] = []
    # Un MetricsRun POR PROMPT, no uno acumulado: el conteo de forwards es la
    # única magnitud que NO depende del hardware, así que es la que permite
    # separar "el código hizo más trabajo" de "la máquina estuvo más lenta"
    # (ver report_prompt_metrics). El loop sigue sin prints: sólo acumula.
    prompt_metrics: list[MetricsRun] = []
    with measure_time("Prueba completa"):
        for i, prompt in enumerate(prompts):
            with track_prompt(i):
                prompt_run = MetricsRun()
                generated.append(
                    generate(
                        model,
                        prompt,
                        vocab,
                        functions,
                        trie_node,
                        metrics=prompt_run,
                    )[0]
                )
                prompt_metrics.append(prompt_run)

    # El conteo de forwards vive en data/output/ (git-ignored, junto al
    # entregable) y NO en /tmp: la evidencia de medición tiene que sobrevivir
    # al reinicio de la VM.
    report_prompt_metrics(prompt_metrics, args.output.parent / "decode_metrics.json")

    for i, result in enumerate(generated):
        print(f"  result : {result}")

    # --- Persistencia del entregable (subject V.4) -------------------------
    # Convertimos cada string del decoder en una FunctionCall validada,
    # emparejada con su prompt ORIGINAL, y escribimos el array a disco.
    # `build_results` conserva el orden 1:1 con los prompts de entrada: la
    # moulinette empareja con `zip()`, que es posicional.
    results: list[FunctionCall] = build_results(raw_prompts, generated)
    written = write_results(results, args.output)

    print()
    print(f"  wrote {len(results)} entries -> {written}")
    return 0


def write_results(results: list[FunctionCall], path: Path) -> Path:
    """Serializa los resultados al JSON de salida y lo escribe en disco.

    Args:
        results: Entries validadas, en el orden de los prompts de entrada.
        path: Destino. El parent se crea si no existe (el subject exige que
            el programa cree el directorio output/ durante la ejecución).

    Returns:
        El path efectivamente escrito.

    POR QUÉ `model_dump()` y no `model_dump_json()`:
    `json.dumps` sobre una lista de dicts es más simple de testear (el
    resultado es texto plano, no bytes) y mantiene el control del indent.
    `ensure_ascii=False` importa: los prompts contienen acentos y comillas
    tipográficas; sin esto el JSON los escapa como \\uXXXX, sigue siendo
    válido pero ilegible para un revisor humano.
    """
    payload = [call.model_dump() for call in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
