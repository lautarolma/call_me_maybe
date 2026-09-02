# PLAN_IMPLEMENTACION — call_me_maybe

> **Autor**: lama-design (big-pickle engine)
> **Fecha**: 2026-08-18
> **Proyecto**: call_me_maybe (42 school — LLM function calling con constrained decoding)
> **Estado**: Plan completo — listo para fase de implementación (sdd-tasks / lama-apply)

---

## Parte A: Guía de Arquitectura

### A1. Contexto y estado del repositorio

**Repo**: `/home/laviles/Python/call_me_maybe`

**Estado actual (verificado)**:

| Elemento | Estado |
|----------|--------|
| `data/input/functions_definition.json` | ✅ Presente — 5 functions, formato verificado |
| `data/input/function_calling_tests.json` | ✅ Presente — 11 prompts |
| `llm_sdk/` | ✅ Presente — SDK proporcionado, NO modificar |
| `llm_sdk/pyproject.toml` | ✅ Presente — dependencias: torch, transformers, huggingface-hub |
| `llm_sdk/llm_sdk/__init__.py` | ✅ Presente — 126 líneas, clase `Small_LLM_Model` |
| `en.subject.pdf` | ✅ Presente — subject del proyecto |
| `src/` | ❌ FALTA — debe crearse desde cero |
| `pyproject.toml` (root) | ❌ FALTA — configuración del proyecto raíz |
| `uv.lock` | ❌ FALTA — se genera con `uv sync` |
| `Makefile` | ❌ FALTA |
| `README.md` | ❌ FALTA |
| `.gitignore` | ❌ FALTA |
| `data/output/` | ❌ FALTA — se crea en runtime, gitignored |

**Public API del SDK (las únicas que podemos usar)**:
- `encode(text: str) -> Tensor` — tokeniza texto a tensor
- `decode(token_ids) -> str` — detokeniza tokens a texto
- `get_logits_from_input_ids(input_ids: list[int]) -> list[float]` — logits del próximo token para la secuencia completa (sin KV-cache)
- `get_path_to_vocab_file() -> str` — ruta a vocab.json
- `get_path_to_merges_file() -> str` — ruta a merges.txt
- `get_path_to_tokenizer_file() -> str` — ruta a tokenizer.json

**Modelo default**: `Qwen/Qwen3-0.6B` (~151,643 tokens en vocabulario, BPE byte-level).

**LO QUE NO PODEMOS HACER**: importar torch, transformers, huggingface_hub, dspy, outlines en nuestro código. Tampoco acceder a `_model`, `_tokenizer`, ni ningún atributo privado del SDK.

---

### A2. Requisitos

#### MUST (obligatorios para MVP)

| # | Requisito | Fuente |
|---|-----------|--------|
| M1 | Python ≥ 3.10 | Subject |
| M2 | Constrained decoding en cada generación | Subject + tesis-tokens |
| M3 | Output JSON: `{"name": "<fn>", "parameters": {<args>}}` | Subject V.4.1 |
| M4 | Modelo ELIGE la función vía inferencia LLM (no heurísticas) | Subject |
| M5 | Todas las clases usan Pydantic para validación | Subject |
| M6 | Entry point: `uv run python -m src` | Subject |
| M7 | `uv sync` instala todo (runner ejecuta solo eso) | Subject |
| M8 | Output file: `data/output/function_calls.json` | Subject V.4.1 |
| M9 | CLI args: `--functions_definition`, `--input`, `--output` | Subject |
| M10 | Makefile con targets: install, run, debug, clean, lint | Subject |
| M11 | Type hints y docstrings en todo | Subject |
| M12 | flake8 + mypy pasan | Subject |
| M13 | 100% JSON válido en output | Subject quality bar |
| M14 | 90%+ accuracy (function + args correctos) | Subject quality bar |
| M15 | <5 min total en CPU | Subject quality bar |
| M16 | NO importar dspy, outlines, torch, transformers, huggingface_hub en nuestro código | Subject |

#### SHOULD (altamente recomendados)

| # | Requisito |
|---|-----------|
| S1 | Context managers donde aplique |
| S2 | Error handling graceful: prompts problemáticos no matan el pipeline |
| S3 | Logging útil (no DEBUG spew) |
| S4 | Tests unitarios sin modelo (mock vocab) |
| S5 | Proceso exitoso = exit code 0, falla = exit code != 0 |

#### BONUS (ANEXO — no parte del main process)

| # | Requisito |
|---|-----------|
| B1 | lint-strict Makefile target |
| B2 | Soporte multi-modelo |
| B3 | Recode tokenizer |
| B4 | Error recovery avanzado |
| B5 | Performance optimizations |
| B6 | Test suite comprehensiva |
| B7 | Visualización de generación |
| B8 | Nested arguments complejos |
| B9 | Public encode/decode API |

---

### A3. Restricciones críticas del subject

1. **FORBIDDEN**: dspy, outlines, standalone pytorch/huggingface/transformers en nuestro código. El SDK los usa internamente — eso está bien.
2. **ALLOWED**: numpy, json (stdlib), pydantic. Nada más.
3. **NO private SDK attributes**: No se puede acceder a `_model`, `_tokenizer`, etc.
4. **Constrained decoding es OBLIGATORIO** — no es opcional, no es bonus.
5. **El modelo DEBE elegir la función** — inferencia LLM, NO heurísticas tipo "si contiene 'sum' → fn_add_numbers".
6. **Reviewers**: ejecutan `uv sync` y nada más. El proyecto debe funcionar con solo eso.
7. **Calidad**: 100% JSON válido, 90%+ accuracy, <5 min total CPU.
8. **Input/Output**: el subject V.4 muestra {"name": "...", "parameters": {...}} — el input DEBE ser un array (output de LLMs como OpenAI).

---

### A4. Arquitectura objetivo

```
call_me_maybe/
├── pyproject.toml               # Root project config — dependencias del proyecto (llm_sdk como local dep)
├── uv.lock                      # Generado por uv sync
├── Makefile                     # install/run/debug/clean/lint targets
├── README.md                    # Descripción, usage, arquitectura
├── .gitignore                   # data/output/, __pycache__, .venv, .mypy_cache
├── llm_sdk/                     # PROPORCIONADO — NO modificar
├── data/
│   ├── input/                   # JSONs provistos
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/                  # Creado en runtime, gitignored
│       └── function_calls.json
├── src/
│   ├── __init__.py              # Package marker (vacío o docstring)
│   ├── __main__.py              # Entry point: CLI parsing + orquestación, error handling top-level
│   ├── cli.py                   # argparse: --functions_definition, --input, --output con defaults
│   ├── models/
│   │   ├── __init__.py
│   │   ├── function_definition.py  # Pydantic: ParameterDef, FunctionDef, FunctionsDefinitionFile
│   │   └── output.py               # Pydantic: FunctionCall, FunctionCallOutput
│   ├── loader/
│   │   ├── __init__.py
│   │   ├── input_loader.py         # Carga + valida prompts (lista de strings o dicts con "prompt")
│   │   ├── function_loader.py      # Carga + valida funciones (nombres duplicados → error)
│   │   └── vocab_loader.py         # Carga vocab.json, construye token→id, id→token, índices pre-computados
│   ├── prompt/
│   │   ├── __init__.py
│   │   └── prompt_builder.py       # System prefix con funciones + user prompt → tokenizable
│   ├── decoder/
│   │   ├── __init__.py
│   │   ├── state.py                # State machine del decoder JSON (dataclass, NO Pydantic)
│   │   ├── trie.py                 # Trie de nombres de funciones para prefix-constrained selection
│   │   ├── schema_validator.py     # Mapa posición → tipo de schema esperado
│   │   ├── token_filter.py         # Computación de allowed-tokens: pre-filtro + post-filtro schema-aware
│   │   └── constrained_generator.py  # Loop principal: logits → mask → argmax → append → update state → termination
│   ├── validator/
│   │   ├── __init__.py
│   │   └── output_validator.py     # Post-generación: parse JSON, Pydantic validate, name existe, types match
│   └── pipeline.py                 # Orquestador: load defs → load vocab → init model → per prompt generate+validate → write output
└── tests/                          # Unit tests SIN modelo (mock vocab)
    ├── __init__.py
    ├── test_state.py
    ├── test_trie.py
    ├── test_token_filter.py
    ├── test_prompt_builder.py
    └── test_output_validator.py
```

**Principio rector**: Separación clara entre I/O (Pydantic) y generación (dataclasses nativos). Pydantic tiene overhead de instanciación que en el inner loop de generación es costoso (~200μs × 50 tokens × 11 prompts = 110ms extra, aceptable pero innecesario).

---

### A5. Flujo de datos (end-to-end)

```
CLI (__main__.py)
  │
  ├─→ Parse args (cli.py)
  │     --functions_definition → path
  │     --input                → path
  │     --output               → path
  │
  ├─→ Load functions (function_loader.py)
  │     file → JSON → [FunctionDef] → validate (no dup names)
  │
  ├─→ Load prompts (input_loader.py)
  │     file → JSON → [str] → validate (non-empty)
  │
  ├─→ Load vocab (vocab_loader.py)
  │     SDK.get_path_to_vocab_file() → json.load → {token: id}
  │     Build: token2id, id2token, tokens_starting_with (pre-indexed)
  │
  ├─→ Init model (Small_LLM_Model)
  │
  ├─→ Per prompt:
  │     │
  │     ├─→ Build prompt (prompt_builder.py)
  │     │     system_prefix (all functions desc) + user_prompt
  │     │
  │     ├─→ Tokenize: model.encode(full_prompt) → input_ids
  │     │
  │     ├─→ Constrained generation loop (constrained_generator.py)
  │     │     │
  │     │     │ for each step:
  │     │     │   logits = model.get_logits_from_input_ids(current_ids)
  │     │     │   allowed_ids = token_filter.compute(logits, state, schema)
  │     │     │   best_id = argmax over allowed_ids
  │     │     │   append best_id
  │     │     │   decoded_text = model.decode(best_id)
  │     │     │   state.update(decoded_text)  ← char-by-char through state machine
  │     │     │   if state == COMPLETE: break
  │     │     │
  │     │     └─→ return generated_text
  │     │
  │     ├─→ Validate output (output_validator.py)
  │     │     parse JSON → FunctionCall Pydantic model → check name exists → check param types
  │     │
  │     └─→ Append to results list
  │
  ├─→ Create data/output/ if needed
  │
  ├─→ Write output (function_calls.json)
  │
  └─→ Print summary (N prompts, success rate, timing)
        exit 0 on success, exit 1 on critical failure
```

**Nota sobre flujo de validación**: El validador post-generación es una REDUNDANCIA defense-in-depth. El constrained decoder YA produce JSON válido por construcción. Pero:
- El decoder garantiza **syntactic** validity (JSON parseable, keys conocidos)
- El validador garantiza **semantic** validity (nombre existe, tipos correctos, no hay extras)
- Si el decoder falla (e.g. allowed set vacío), el validador reporta el error

---

### A6. Diseño del constrained decoding

#### A6.1. El problema fundamental

Tokens de BPE NO son caracteres. Un solo token puede representar múltiples caracteres:
- Token `Ġthe` = espacio + "the" (4 chars)
- Token `hello` = 5 chars
- Token `Ã©` = bytes de UTF-8 para 'é' (2 bytes, incomplete)

**Solución**: al decodificar cada token candidato, procesamos TODOS sus caracteres character-by-character a través de la state machine. El token es válido solo si TODOS sus caracteres mantienen el JSON válido + schema-compliant.

#### A6.2. Estrategia de vocabulario

```
vocab_loader.py construye al startup:

1. Carga vocab.json → dict[str, int] (token_text → token_id)
2. Construye id2token: dict[int, str] (inverso)
3. Pre-indexa por primer carácter:
   tokens_starting_with: dict[str, set[int]]
   - Para cada token, toma el primer char de decoded_text
   - Agrupa ids por ese char
   - Incluye categoría especial "byte" para tokens que son secuencias UTF-8 incompletas
4. Computa token_lengths: dict[int, int] (longitud en chars de cada token decodificado)
```

**Por qué pre-indexar**: El vocabulario tiene ~151,643 tokens. Sin pre-indexar, cada step del loop escanea TODOS para filtrar por primer char — ~151K comparaciones × ~50 steps × 11 prompts = ~83M operaciones en CPython. Con pre-indexado, cada step solo examina los tokens de la categoría relevante (típicamente 2,000-15,000).

#### A6.3. State Machine del decoder JSON

**Estados** (implementados como `@dataclass` con `__slots__`):

```
ROOT                  → Estado inicial, esperando '{' de output object
OBJECT_OPEN           → '{' leído, dentro del output object
IN_OBJECT             → Dentro del output object, esperando key o '}'
KEY_START             → '"' leído, empezando a leer key name
IN_KEY                → Leyendo caracteres de key name
KEY_END               → '"' leído al final de key name
COLON                 → ':' leído, esperando value
VALUE_START           → Primer char del value (determina tipo)
IN_STRING_VALUE       → Dentro de un value string (func name o string param)
IN_NUMBER_VALUE       → Dentro de un value number
IN_BOOL_VALUE         → Dentro de un value boolean (true/false/null)
IN_NULL_VALUE         → Dentro de null
ESCAPE_IN_STRING      → '\' leído dentro de string, esperando char de escape
VALUE_END             → Value terminado, esperando ',' o '}'
COMPLETE              → '}' de cierre del output object leído — STOP
```

**Transiciones detalladas**:

```
ROOT ──────────────{──────────────→ OBJECT_OPEN

OBJECT_OPEN ───────"──────────────→ KEY_START

IN_OBJECT ─────────"──────────────→ KEY_START
IN_OBJECT ─────────}──────────────→ COMPLETE

KEY_START ─────────char──────────→ IN_KEY  (acumula key name)
IN_KEY ────────────"──────────────→ KEY_END
IN_KEY ────────────char──────────→ IN_KEY  (acumula)

KEY_END ───────────:──────────────→ COLON

COLON ─────────────"──────────────→ VALUE_START → IN_STRING_VALUE
COLON ─────────────- or digit────→ VALUE_START → IN_NUMBER_VALUE
COLON ─────────────t──────────────→ VALUE_START → IN_BOOL_VALUE
COLON ─────────────f──────────────→ VALUE_START → IN_BOOL_VALUE
COLON ─────────────n──────────────→ VALUE_START → IN_NULL_VALUE
COLON ─────────────{──────────────→ VALUE_START → PARAMS_OBJECT (nested)

IN_STRING_VALUE ───"──────────────→ VALUE_END
IN_STRING_VALUE ───\──────────────→ ESCAPE_IN_STRING
IN_STRING_VALUE ───char──────────→ IN_STRING_VALUE

IN_NUMBER_VALUE ───digit or .────→ IN_NUMBER_VALUE
IN_NUMBER_VALUE ───e or E────────→ IN_NUMBER_VALUE
IN_NUMBER_VALUE ───, or } or ws──→ VALUE_END

IN_BOOL_VALUE ─────char──────────→ IN_BOOL_VALUE (acumula hasta match "true"/"false"/"null")
IN_BOOL_VALUE ─────terminal──────→ VALUE_END

ESCAPE_IN_STRING ──char──────────→ IN_STRING_VALUE (\", \\, \/, \n, \t, \r, \b, \f, \uXXXX)

VALUE_END ─────────,──────────────→ IN_OBJECT
VALUE_END ─────────}──────────────→ COMPLETE
```

**Estados adicionales para nested parameters object**:

```
PARAMS_OBJECT ─────"──────────────→ KEY_START (recursive para parameters keys)
PARAMS_OBJECT ─────}──────────────→ VALUE_END (cierre de parameters)
```

#### A6.4. Algoritmo de computación de allowed-tokens (3 fases)

**Fase 1: Pre-filtro por categoría de primer carácter**
```python
# En token_filter.py
current_char_category = state.expected_first_chars()  # retorna set de chars válidos
candidate_ids = set()
for char in current_char_category:
    candidate_ids.update(vocab.tokens_starting_with.get(char, set()))
```

**Fase 2: Decodificación parcial + validación char-by-char**
```python
# Para cada candidate_id en candidate_ids:
for token_id in candidate_ids:
    token_text = vocab.id2token[token_id]
    valid, new_state = state.simulate(token_text)
    if valid:
        # Verificar schema constraints
        if schema.allows(token_text, new_state):
            allowed_ids.add(token_id)
```

**Fase 3: Post-filtro de schema (para keys y values)**
```python
# state.simulate() procesa cada char del token a través de la state machine
# Retorna (is_valid: bool, resulting_state: DecoderState)
# El token es allowed SOLO SI:
#   1. Todos sus chars mantienen JSON válido
#   2. Si estamos en KEY_START/IN_KEY: el key parcial es prefix de un key conocido
#   3. Si estamos en IN_STRING_VALUE para function name: el parcial es prefix de un nombre válido
#   4. Si estamos en VALUE para number: cumple JSON number grammar
#   5. Si estamos en VALUE para string: no hay restricción adicional (strings libres)
#   6. Si el token cierra '}' del parameters object: TODOS los required keys ya presentes
```

**Detalles críticos del Phase 2**:

```python
def simulate(self, token_text: str) -> tuple[bool, "DecoderState"]:
    """Simula procesar un token completo a través de la state machine.
    Retorna (is_valid, new_state)."""
    state = copy(self)  # deep copy del estado actual
    for char in token_text:
        valid = state._advance_char(char)
        if not valid:
            return False, self  # retorna estado original si falla
    return True, state
```

`_advance_char` implementa las transiciones de la state machine. Cada char se procesa individualmente.

#### A6.5. Procesamiento de tokens

```
Para cada step del loop de generación:

1. logits = model.get_logits_from_input_ids(current_ids)  # lista de ~151K floats
2. allowed_ids = token_filter.compute_allowed(logits, current_state, current_schema)
3. Si allowed_ids vacío:
   a. Log warning
   b. Intentar cerrar estructuras abiertas (append '}'s necesarios)
   c. Break con resultado parcial
4. best_id = argmax(logits[i] for i in allowed_ids)  # softmax-free, solo max logit
5. current_ids.append(best_id)
6. token_text = vocab.id2token[best_id]
7. current_state.update_from_text(token_text)  # char-by-char
8. current_schema.update(current_state)  # refresca expected types
9. Si current_state == COMPLETE: break
10. Si len(current_ids) > MAX_TOKENS: break (safety)
```

**MAX_TOKENS = 200** (safety net — el output esperado es ~30-60 tokens).

#### A6.6. Terminación

- **Primary**: state == COMPLETE (root object '}' leído)
- **Safety**: max_tokens counter (200)
- **NO usar EOS**: el subject advierte explícitamente que modelos pequeños emiten EOS prematuramente
- El loop PARA cuando COMPLETE — no genera tokens extra después

#### A6.7. Schema enforcement detallado

**Para la key "name"**:
- En VALUE_START/IN_STRING_VALUE: solo tokens que son prefixes de nombres de funciones conocidos
- Cuando el string se cierra: el nombre completo DEBE existir en functions_definition
- Después de "name": esperando ',' para ir a "parameters"

**Para las keys del objeto parameters**:
- En KEY_START/IN_KEY: solo tokens que son prefixes de keys del schema de la función seleccionada
- Cuando el string se cierra: el key DEBE existir en el schema

**Para los values del objeto parameters**:
- `type: "string"` → IN_STRING_VALUE hasta '"' — contenido libre
- `type: "number"` → IN_NUMBER_VALUE — JSON number grammar:
  - Primer char: dígito o '-'
  - Después de dígito: dígito, '.', 'e', 'E', ',', '}'
  - Después de '.': al menos un dígito (no ".." ni ".}")
  - Después de 'e'/'E': dígito, '+', '-'
  - NO leading zeros: "0" OK, "01" NO (excepto "0.5")

**Para cerrar parameters**:
- `}` de cierre del parameters object SOLO permitido si TODOS los required keys del schema están presentes
- Esto previene outputs como `{"name": "fn_add_numbers", "parameters": {"a": 2.0}}` sin "b"

**Para cerrar el output object**:
- `}` de cierre del root object → COMPLETE

#### A6.8. Manejo de allowed set vacío

```
Si allowed_ids está vacío después de las 3 fases:
1. Log: "WARNING: Empty allowed set at state {state}, key={current_key}, step={step}"
2. Intentar repair: simular cierre de estructuras abiertas
   - Si estamos en IN_KEY: cerrar '"' → KEY_END
   - Si estamos en VALUE_START: agregar value por defecto según tipo
   - Si estamos en IN_OBJECT: cerrar con '}'
3. Si repair exitoso: continuar con next step
4. Si repair falla: marcar prompt como error, continuar con siguiente prompt
5. Al final: reportar prompts fallidos en summary
```

---

### A7. Estrategia de selección de función (trie)

**Estructura**: Trie (prefix tree) construido desde los nombres de funciones en `functions_definition.json`.

```python
@dataclass
class TrieNode:
    __slots__ = ('children', 'function_name', 'is_end')
    children: dict[str, "TrieNode"]
    function_name: str | None  # Non-None solo en nodos finales
    is_end: bool
```

**Construcción**:
```python
def build_trie(function_names: list[str]) -> TrieNode:
    root = TrieNode()
    for name in function_names:
        node = root
        for char in name:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.function_name = name
        node.is_end = True
    return root
```

**Uso en constrained decoding**:
1. Cuando estamos en `IN_STRING_VALUE` para el campo "name" del output:
   - Tenemos el prefijo acumulado (e.g., "fn_a")
   - Recorremos el trie con ese prefijo → llegamos a un nodo
   - Si el nodo tiene children: los chars válidos son las keys de `children`
   - Si el nodo es terminal: '"' también es válido (cerrar el nombre)
   - Si no existe el prefijo en el trie:允许 set vacío → repair o error

2. **Zero hardcoding**: el trie se construye dinámicamente desde el JSON de entrada. Si mañana se agregan funciones, el trie se adapta automáticamente.

**Nombres de las 5 funciones actuales**:
- `fn_add_numbers` (prefix común: `fn_`)
- `fn_greet`
- `fn_reverse_string`
- `fn_get_square_root`
- `fn_substitute_string_with_regex`

Los prefijos compartidos (`fn_`) se optimizan naturalmente por el trie — solo una rama para `fn_`, luego se bifurca.

---

### A8. Manejo de errores

#### Tabla de condiciones de error

| Condición | Severidad | Acción | Continúa? |
|-----------|-----------|--------|-----------|
| Archivo de funciones no existe | FATAL | stderr + exit 1 | NO |
| JSON de funciones malformado | FATAL | stderr + exit 1 | NO |
| Funciones con nombres duplicados | FATAL | stderr + exit 1 | NO |
| Archivo de prompts no existe | FATAL | stderr + exit 1 | NO |
| JSON de prompts malformado | FATAL | stderr + exit 1 | NO |
| Lista de prompts vacía | FATAL | stderr + exit 1 | NO |
| Error cargando modelo | FATAL | stderr + exit 1 | NO |
| Allowed set vacío en step | WARNING | repair + log, skip prompt si no repairable | SÍ |
| JSON output malformado post-gen | WARNING | log + skip prompt | SÍ |
| Nombre de función no encontrado | WARNING | log + skip prompt | SÍ |
| Tipo de parámetro incorrecto | WARNING | log + skip prompt | SÍ |
| Output dir no existe | INFO | crear data/output/ | SÍ |
| Token con bytes incompletos | INFO | skip en pre-filter | SÍ |

#### Principios de error handling

1. **Fail fast en inputs**: si los archivos de entrada son inválidos, no tiene sentido continuar
2. **Per-prompt resilience**: un prompt fallido no debe matar el pipeline para los demás
3. **Exit code**: 0 = éxito (todos los prompts procesados), 1 = falla crítica (no se pudo empezar)
4. **Logging**: errores en stderr, info en stdout
5. **Reparación de output**: si el decoder produce JSON parcial, intentar cerrar estructuras

---

### A9. Diseño de performance

#### Análisis de costos

```
11 prompts × ~50 tokens generados × ~200ms/token = ~110s total
Margen para 5 min (300s): 190s de overhead para I/O, validación, etc.
```

**Costo por token step**:
- `get_logits_from_input_ids`: ~150-200ms (CPU, Qwen3-0.6B, sin KV-cache)
- `token_filter.compute_allowed`: ~0.5-2ms (con pre-indexing)
- `state.update`: ~0.01ms (trivial)
- **Total por step**: ~150-200ms

#### Optimizaciones OBLIGATORIAS (Gemini refinements)

1. **Vocab pre-indexing por primer carácter**: `tokens_starting_with[char] → set[int]`
   - SIN esto: ~80-150ms/step scanneando 151K tokens en CPython
   - CON esto: ~0.1-0.5ms/step (solo examinar tokens de la categoría)

2. **dataclass con `__slots__` para DecoderState**: evita `__dict__` overhead
   - NO usar Pydantic en el inner loop (heap allocation per step)

3. **Token simulation eficiente**: `str` iteration es O(n) por token, tokens son cortos (~3-5 chars avg)

4. **Evitar `copy.deepcopy`** en simulate: usar `copy()` del dataclass (shallow, ~50ns vs ~5μs)

#### Qué NO optimizar

- NO KV-cache (SDK no lo expone)
- NO batching (SDK es single-sequence)
- NO GPU-specific optimizations
- NO cuantización
- NO flash attention
- NO ONNX runtime

**Riesgo de performance**: el scanneo de vocabulario es el bottleneck principal. Sin pre-indexing, el proyecto probablemente NO cumple 5 min. Con pre-indexing, estamos holgados.

---

### A10. Estrategia de testing

#### Unit tests (sin modelo)

| Test | Qué verifica | Mock approach |
|------|-------------|---------------|
| `test_state.py` | Transiciones de la state machine | Input manual de chars |
| `test_trie.py` | Construcción, prefix matching, validación | Nombres hardcodeados |
| `test_token_filter.py` | Filtro de tokens permitidos | Vocabularios small mock |
| `test_prompt_builder.py` | Formato de prompt | Funciones mock |
| `test_output_validator.py` | Validación post-generación | JSONs válidos/inválidos |

**Estructura de mocks**:
```python
# Mock vocab para tests — vocabulario tiny
MOCK_VOCAB = {
    '{"name":': 1,
    '"fn_add_numbers"': 2,
    ', "parameters":': 3,
    '{"a":': 4,
    '2.0': 5,
    ', "b":': 6,
    '3.0': 7,
    '}': 8,
    '}': 9,  # cierre root
}
```

#### Integration test (con modelo)

- Un solo prompt de smoke test: "What is 2+3?"
- Verificar que output es JSON válido y contiene "fn_add_numbers"
- NO incluir en CI — solo para verificación manual

#### Test execution

```bash
make test          # pytest tests/ -v
make lint          # flake8 + mypy
make test-cov      # pytest --cov=src --cov-report=term-missing
```

---

### A11. Riesgos arquitectónicos

| # | Riesgo | Severidad | Probabilidad | Mitigación |
|---|--------|-----------|-------------|------------|
| R1 | BPE tokens con bytes incompletos causan UnicodeDecodeError | ALTA | ALTA | Manejo en vocab_loader: skip tokens que no decodifican limpio |
| R2 | Allowed set vacío → generación falla | ALTA | MEDIA | Repair heuristic + graceful skip |
| R3 | Modelo emite EOS prematuramente (50% según subject) | ALTA | ALTA | Ignorar EOS token, usar solo constrained termination |
| R4 | Performance >5min en CPU | MEDIA | BAJA | Pre-indexing obligatorio; budget holgado con optimización |
| R5 | keys faltantes en parameters (JSON válido pero semánticamente inválido) | ALTA | MEDIA | Bloquear '}' de parameters hasta required keys presentes |
| R6 | Nombres de función no reconocidos por el modelo | MEDIA | BAJA | Trie constraint fuerza nombres válidos |
| R7 | Python bytes en BPE tokens (Qwen usa byte-level BPE) | MEDIA | ALTA | Pre-index handling special bytes |
| R8 | Mypy o flake8 rechazan patterns del inner loop | BAJA | BAJA | Type: ignore comment y documentar por qué |

---

### A12. Decisiones de diseño

#### Decisión 1: Pydantic SOLO en I/O, dataclasses en inner loop

**Elección**: Pydantic en `models/`, `loader/`, `validator/`. Dataclasses nativos en `decoder/`.
**Alternativas rechazadas**: Pydantic en todo (overhead ~200μs/instancia en inner loop); dataclasses en todo (pierde validación declarativa en I/O).
**Razón**: Pydantic es excelente para validación de input/output (schema declarativo, error messages claros). Pero en el inner loop de generación, cada step instanciaría un objeto Pydantic — innecesario y costoso. El decoder necesita velocidad, no validación schema en cada token.

#### Decisión 2: Trie separado (no fused en token_filter)

**Elección**: `trie.py` como módulo independiente.
**Alternativas rechazadas**: Fused dentro de `token_filter.py`.
**Razón**: Separación de concerns. El trie es una estructura de datos genérica que se puede testear aisladamente. El token_filter lo usa como componente, no lo contiene. Mejor para testing y mantenibilidad.

#### Decisión 3: Prompt system-like prefix (no chat template)

**Elección**: Un solo string con system-like prefix + user prompt, tokenizado de una vez.
**Alternativas rechazadas**: Chat template (requiere tokens especiales de chat del modelo, más complejo).
**Razón**: El subject dice "model receives a prompt containing the function definitions and the user's query". Un prefix plano es más portable y testeable. El modelo Qwen3 entiende bien instructions en texto plano.

#### Decisión 4: Salida flat `{"name": ..., "parameters": {...}}` (no wrapper array)

**Elección**: Cada prompt genera un JSON object independiente. El output final es un array de estos objects.
**Alternativas rechazadas**: Un solo object con keys "function_calls": [...].
**Razón**: El subject V.4 muestra exactamente `{"name": "...", "parameters": {...}}` como formato de output por prompt. El output file es un array de estos objects (uno por prompt input).

#### Decisión 5: int vs float para number params

**Elección**: Preservar semántica JSON — emitir int cuando integral (2.0 → 2), float cuando no (2.5 → 2.5).
**Alternativas rechazadas**: Siempre float (pierde información); siempre int (pierde precisión).
**Razón**: JSON number semantics. `2.0` se emite como `2` porque `2` es un JSON number válido. `2.5` se emite como `2.5`. El validador Pydantic acepta ambos para `float`.

**Nota**: hay ambigüedad en el subject — "a": 2.0 vs "a": 2. La estrategia de emitir int cuando integral es la más robusta.

#### Decisión 6: `returns` field es input-only

**Elección**: El campo `returns` de functions_definition.json se usa SOLO para prompt building (informar al modelo qué retorna cada función). NO se incluye en el output schema del decoder.
**Alternativas rechazadas**: Incluir `returns` en el output JSON.
**Razón**: El formato de output del subject es `{"name": ..., "parameters": {...}}`. No hay campo `returns`. Es metadata de input.

#### Decisión 7: Prompt template default

**Elección**: Template con system prefix descriptivo + listado de funciones con sus schemas + user prompt. Un solo prompt concatenado.

Template default:
```
You are a function calling assistant. Given the user's query, you must output a JSON object that calls the most appropriate function.

Available functions:

1. fn_add_numbers: Add two numbers together and return their sum.
   Parameters: a (number), b (number)

2. fn_greet: Generate a greeting message for a person by name.
   Parameters: name (string)

3. fn_reverse_string: Reverse a string and return the reversed result.
   Parameters: s (string)

4. fn_get_square_root: Calculate the square root of a number.
   Parameters: a (number)

5. fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.
   Parameters: source_string (string), regex (string), replacement (string)

Output ONLY a JSON object with "name" and "parameters" fields. No explanation.
User query: {user_prompt}
```

**Alternativas rechazadas**: Few-shot (requiere examples, más tokens); XML format (más tokens, modelo pequeño puede no entender); JSON schema format (demasiado verbose para modelo pequeño).

**Validación experimental**: testear 2-3 variantes (con/sin "Output ONLY", con/sin numeración, con/sin "No explanation") y medir accuracy.

#### Decisión 8: Decoder state como dataclass con copy

**Elección**: DecoderState es un `@dataclass(slots=True)` con método `simulate(token_text) -> tuple[bool, DecoderState]` que hace shallow copy.
**Alternativas rechazadas**: Estado como tupla inmutable (inconveniente para updates); estado mutable sin copy (simulación corrupta el estado real).
**Razón**: `simulate` necesita explorar sin afectar el estado actual. Shallow copy de dataclass con `__slots__` es ~50ns vs ~5μs de deepcopy. Funciona porque los campos son types inmutables (str, int, bool) o structures simples (dict, set).

---

### A13. Ambigüedades resueltas

| Ambigüedad | Resolución | Razón |
|------------|-----------|-------|
| Output filename: `function_calls.json` vs `function_calling_results.json` | `function_calls.json` | Matchea el default del CLI del subject y la convención más corta |
| int vs float para numbers | emitir int cuando integral, float cuando no | JSON number semantics, más robusto |
| Formato de parameters: flat object vs nested | flat object `{"a": 2.0, "b": 3.0}` | Subject V.4.1 muestra exactamente esto |
| `returns` en output | NO — es solo metadata de input | Subject format solo tiene "name" + "parameters" |
| Prompt template exacto | Default candidate + variantes para testing | Necesita validación empírica con el modelo |
| Función no determinable | graceful error per prompt, continuar con otros | Subject pide 90%+ accuracy, no 100% |
| Tokens con bytes UTF-8 incompletos | skip en pre-filter, log | BPE byte-level es inevitable con Qwen |
| Parameters closing brace timing | BLOCK hasta todos required keys | Subject warning explícito en tesis-tokens |

---

### A14. Readiness

#### Lo que queda ambiguo (requiere decisión en implementación)

1. **Prompt template exacto**: el default es un starting point. La accuracy real dependerá de cómo el modelo Qwen3-0.6B responde. Hay que testear 2-3 variantes.
2. **Token limit máximo**: 200 es un safety net razonable, pero puede ajustarse.
3. **Manejo de `thought` tags**: Qwen3 puede emitir `<|begin_of_thought|>...<|end_of_thought|>` antes de la respuesta. El decoder debe manejar esto (skip hasta encontrar '{').

#### Lo que necesita validación empírica

- El prompt template exacto que maximiza accuracy con Qwen3-0.6B
- Si el modelo emite tokens de thinking o no (y cómo manejarlos)
- El timing real en CPU del pipeline completo
- Si los 11 prompts son todos resolubles por el modelo

#### El componente más difícil

**token_filter.py** — específicamente la interacción entre:
- El trie de nombres de función
- La validación char-by-char del token contra la state machine
- El manejo de tokens multi-char BPE
- El schema enforcement para keys y values

Esto es ~300-400 líneas de código cuidadoso con mucho edge cases.

#### El mayor riesgo de evaluación

**Accuracy < 90%** — el modelo Qwen3-0.6B es pequeño (0.6B params). Puede:
- No entender bien el prompt
- Elegir la función incorrecta
- Emitir parámetros con valores incorrectos

Mitigación: el constrained decoding GARANTIZA JSON válido y nombres/keys correctos. La accuracy depende del modelo eligiendo la función correcta y passando los valores correctos — ahí no podemos hacer mucho más allá del prompt engineering.

---

## Parte B: Anexo BONUS

> **⚠️ NO PART OF MAIN PROCESS — evaluar solo después del MVP funcional**

### B1. lint-strict Makefile target

**Cambio arquitectónico**: agregar target `lint-strict` que ejecuta flake8 con reglas extendidas (errors, warnings, max-line-length=100, banned-functions, etc.) + mypy --strict.

**Mini plan**:
1. Agregar sección `[tool.flake8]` o `.flake8` config
2. Agregar `[tool.mypy]` con strict en pyproject.toml
3. Target: `make lint-strict` ejecuta ambos con flags estrictos

### B2. Soporte multi-modelo

**Cambio arquitectónico**: abstraer el modelo detrás de una interfaz `LLMProvider` que el CLI selecciona.

**Mini plan**:
1. Crear `src/model/provider.py` con protocol `LLMProvider`
2. `Small_LLM_Model` se adapta como implementación
3. CLI: `--model <model_name>` argumento adicional
4. factory pattern en `pipeline.py`

### B3. Recode tokenizer

**Cambio arquitectónico**: en vez de usar `get_path_to_vocab_file()` y parsear JSON, usar `get_path_to_tokenizer_file()` y parsear el tokenizer.json completo (que tiene más metadata).

**Mini plan**:
1. `vocab_loader.py` parsea tokenizer.json en vez de vocab.json
2. Extrae merges, added_tokens, normalizer info
3. Más robusto para edge cases de BPE

### B4. Error recovery avanzado

**Cambio arquitectónico**: en vez de skip on failure, intentar regenerar con temperatura o con prompt modificado.

**Mini plan**:
1. Detectar exactamente dónde falló (state, key, value)
2. Reintentar con prompt que incluye contexto del error
3. Agregar temperatura control (requiere acceso a logits processing del modelo)

### B5. Performance optimizations

**Cambios**:
1. Batch processing de prompts (si SDK lo permite)
2. KV-cache management (si SDK lo expone)
3. Profiling con `cProfile` y optimización de hot paths
4. `__slots__` en todas las data classes del decoder

**Mini plan**:
1. Benchmark pipeline completa
2. Identificar top-3 bottlenecks
3. Optimizar uno por uno

### B6. Test suite comprehensiva

**Cambios**:
1. Tests de integración con modelo mock
2. Property-based testing con Hypothesis
3. Coverage target 90%+
4. Tests de edge cases: strings vacíos, números negativos, null values, unicode

**Mini plan**:
1. `tests/test_integration.py` con fixtures completos
2. `tests/test_property.py` con Hypothesis
3. `pytest-cov` en Makefile

### B7. Visualización de generación

**Cambios**:
1. Flag `--visualize` que muestra step-by-step
2. Colores ANSI: green=allowed, red=blocked, yellow=current state
3. Muestra el trie siendo recorrido en tiempo real

**Mini plan**:
1. `src/visualizer.py` con ANSI output
2. `constrained_generator.py` accepta optional visualizer callback
3. `make run-visual` target

### B8. Nested arguments complejos

**Cambios**:
1. Soporte para parameters que son objetos anidados
2. State machine extensión para nested objects/arrays
3. Schema recursivo en Pydantic models

**Mini plan**:
1. Extender state machine con OBJECT_NESTED state
2. Extender schema_validator para nested paths
3. Actualizar function_definition.py con nested ParameterDef

### B9. Public encode/decode API

**Cambios**:
1. `src/api.py` con funciones públicas `encode()`, `decode()`, `generate()`
2. Facade sobre el pipeline para uso como librería
3. Type-safe API con Pydantic models

**Mini plan**:
1. `src/api.py` con `@dataclass` config y funciones públicas
2. Re-export en `src/__init__.py`
3. Docstrings completos con examples

---

## Parte C: Plan Ejecutable por Agentes

### Dependencias entre fases

```
Phase 1 (Foundation)
    ↓
Phase 2 (Prompt)
    ↓
Phase 3 (Decoder Core) ←─┐
    ↓                     │ (puede paralelizarse)
Phase 4 (Generation Loop) ←┘
    ↓
Phase 5 (Validation + Pipeline)
    ↓
Phase 6 (Error Handling + Polish)
    ↓
Phase 7 (Bonus — ANNEX ONLY)
```

**Regla**: Cada fase DEBE completar sus Definition of Done antes de que la siguiente pueda empezar.

---

### Phase 1: Foundation

**Objetivo**: Project structure, loading, CLI — todo funcional sin generar nada.

#### Task 1.1: Root pyproject.toml
- **Archivo**: `/home/laviles/Python/call_me_maybe/pyproject.toml`
- **Implementar**:
  ```toml
  [project]
  name = "call-me-maybe"
  version = "0.1.0"
  description = "LLM function calling with constrained decoding"
  requires-python = ">=3.10"
  dependencies = [
      "llm-sdk",
      "pydantic>=2.0.0",
      "numpy>=1.24.0",
  ]

  [tool.hatch.build.targets.wheel]
  packages = ["src"]
  ```

  Configurar llm_sdk como dependency local:
  ```toml
  [tool.uv.sources]
  llm-sdk = { path = "./llm_sdk", editable = true }

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```
- **Acceptance criteria**: `uv sync` instala todo sin errores
- **Dependencies**: ninguna

#### Task 1.2: .gitignore
- **Archivo**: `/home/laviles/Python/call_me_maybe/.gitignore`
- **Implementar**: data/output/, __pycache__, .venv, .mypy_cache, *.egg-info, uv.lock
- **Acceptance criteria**: `git status` no muestra archivos de build
- **Dependencies**: ninguna

#### Task 1.3: Makefile
- **Archivo**: `/home/laviles/Python/call_me_maybe/Makefile`
- **Implementar**:
  ```makefile
  .PHONY: install run debug clean lint test

  install:
  	uv sync

  run:
  	uv run python -m src

  debug:
  	uv run python -m src --help

  clean:
  	rm -rf __pycache__ .mypy_cache .pytest_cache
  	rm -rf src/__pycache__ src/*/__pycache__
  	rm -rf tests/__pycache__

  lint:
  	flake8 src/ tests/ --max-line-length=120
  	mypy src/ --ignore-missing-imports

  test:
  	pytest tests/ -v
  ```
- **Acceptance criteria**: `make install` funciona, `make run` ejecuta el pipeline
- **Dependencies**: Task 1.1

#### Task 1.4: src/ package init
- **Archivos**: `src/__init__.py`, `src/__main__.py`
- **Implementar**:
  - `src/__init__.py`: docstring del package
  - `src/__main__.py`: import argparse, parse args, print args (placeholder)
- **Acceptance criteria**: `uv run python -m src --help` muestra help del CLI
- **Dependencies**: Task 1.1, Task 1.3

#### Task 1.5: CLI parser
- **Archivo**: `src/cli.py`
- **Implementar**:
  ```python
  import argparse
  from pathlib import Path

  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
      parser = argparse.ArgumentParser(description="LLM function calling with constrained decoding")
      parser.add_argument(
          "--functions_definition",
          type=Path,
          default=Path("data/input/functions_definition.json"),
          help="Path to functions definition JSON file"
      )
      parser.add_argument(
          "--input",
          type=Path,
          default=Path("data/input/function_calling_tests.json"),
          help="Path to input prompts JSON file"
      )
      parser.add_argument(
          "--output",
          type=Path,
          default=Path("data/output/function_calls.json"),
          help="Path to output JSON file"
      )
      return parser.parse_args(argv)
  ```
- **Acceptance criteria**: `parse_args([])` retorna defaults correctos; `parse_args(["--input", "x.json"])` overridea
- **Dependencies**: Task 1.4

#### Task 1.6: Pydantic models
- **Archivos**: `src/models/__init__.py`, `src/models/function_definition.py`, `src/models/output.py`
- **Implementar**:
  ```python
  # src/models/function_definition.py
  from pydantic import BaseModel, Field

  class ParameterDef(BaseModel):
      name: str = Field(description="Parameter name")
      type: str = Field(description="Parameter type: 'string', 'number', 'boolean', 'null'")

  class FunctionDef(BaseModel):
      name: str = Field(description="Function name, e.g. 'fn_add_numbers'")
      description: str = Field(description="Human-readable description")
      parameters: dict[str, dict[str, str]] = Field(
          description="Parameter name -> {type: ...}"
      )
      returns: dict[str, str] = Field(
          description="Return type info"
      )
  ```
  ```python
  # src/models/output.py
  from pydantic import BaseModel, Field

  class FunctionCall(BaseModel):
      name: str = Field(description="Name of the function to call")
      parameters: dict[str, str | int | float | bool | None] = Field(
          default_factory=dict,
          description="Function arguments"
      )
  ```
- **Acceptance criteria**: `FunctionDef(**valid_json)` valida; `FunctionDef(**invalid)` lanza ValidationError
- **Dependencies**: Task 1.1

#### Task 1.7: Input loader
- **Archivo**: `src/loader/__init__.py`, `src/loader/input_loader.py`
- **Implementar**:
  ```python
  from pathlib import Path
  import json

  def load_prompts(path: Path) -> list[str]:
      """Load prompts from JSON file. Supports list of strings or list of objects with 'prompt' key."""
      with open(path) as f:
          data = json.load(f)
      if not isinstance(data, list) or len(data) == 0:
          raise ValueError(f"Expected non-empty list in {path}")
      prompts = []
      for item in data:
          if isinstance(item, str):
              prompts.append(item)
          elif isinstance(item, dict) and "prompt" in item:
              prompts.append(item["prompt"])
          else:
              raise ValueError(f"Invalid prompt format: {item}")
      return prompts
  ```
- **Acceptance criteria**: carga los 11 prompts del test file; lanza error en JSON vacío o malformado
- **Dependencies**: Task 1.6

#### Task 1.8: Function loader
- **Archivo**: `src/loader/function_loader.py`
- **Implementar**:
  ```python
  from pathlib import Path
  import json
  from src.models.function_definition import FunctionDef

  def load_functions(path: Path) -> list[FunctionDef]:
      """Load and validate function definitions. Raises on duplicate names."""
      with open(path) as f:
          data = json.load(f)
      functions = [FunctionDef(**item) for item in data]
      names = [fn.name for fn in functions]
      dupes = [n for n in names if names.count(n) > 1]
      if dupes:
          raise ValueError(f"Duplicate function names: {set(dupes)}")
      return functions
  ```
- **Acceptance criteria**: carga las 5 funciones; nombres duplicados → ValueError
- **Dependencies**: Task 1.6

#### Task 1.9: Vocab loader
- **Archivo**: `src/loader/vocab_loader.py`
- **Implementar**:
  ```python
  import json
  from dataclasses import dataclass, field
  from llm_sdk.llm_sdk import Small_LLM_Model

  @dataclass
  class Vocab:
      token2id: dict[str, int]
      id2token: dict[int, str]
      tokens_starting_with: dict[str, set[int]]  # char → set of token ids
      vocab_size: int

  def load_vocab(model: Small_LLM_Model) -> Vocab:
      """Load vocabulary from model and build pre-indexed structures."""
      vocab_path = model.get_path_to_vocab_file()
      with open(vocab_path) as f:
          raw_vocab = json.load(f)

      token2id = raw_vocab  # {token_text: token_id}
      id2token = {v: k for k, v in token2id.items()}

      # Pre-index by first decoded character
      tokens_starting_with: dict[str, set[int]] = {}
      for token_text, token_id in token2id.items():
          # Handle bytes-level BPE tokens
          try:
              decoded = token_text.encode('utf-8').decode('utf-8')
              first_char = decoded[0] if decoded else ""
          except (UnicodeDecodeError, IndexError):
              first_char = "<byte>"
          tokens_starting_with.setdefault(first_char, set()).add(token_id)

      return Vocab(
          token2id=token2id,
          id2token=id2token,
          tokens_starting_with=tokens_starting_with,
          vocab_size=len(token2id),
      )
  ```
- **Acceptance criteria**: vocab loadea 151K+ tokens; tokens_starting_with tiene entries; tokens bytes no crashean
- **Dependencies**: Task 1.1

#### Task 1.10: Pipeline skeleton
- **Archivo**: `src/pipeline.py`
- **Implementar**: función `run(args)` que:
  1. Parsea args
  2. Loada functions
  3. Loada prompts
  4. Loada vocab
  5. Init model
  6. Print summary
  7. Return success
- **Acceptance criteria**: `uv run python -m src` ejecuta sin errores, imprime que todo se cargó
- **Dependencies**: Tasks 1.4-1.9

#### Task 1.11: __main__.py final
- **Archivo**: `src/__main__.py`
- **Implementar**: import cli.parse_args, pipeline.run; try/except con exit code
- **Acceptance criteria**: `uv run python -m src` exit 0; args inválidos exit 1
- **Dependencies**: Task 1.10

#### Task 1.12: Lint
- **Archivos**: `.flake8` o config en pyproject.toml; `[tool.mypy]` config
- **Implementar**: configurar flake8 + mypy para pasar en todo el código
- **Acceptance criteria**: `make lint` pasa sin errores
- **Dependencies**: Tasks 1.4-1.11

#### Definition of Done — Phase 1
- [ ] `uv run python -m src --help` muestra help con los 3 args
- [ ] `uv run python -m src` carga functions + prompts + vocab, imprime OK
- [ ] `make lint` pasa (flake8 + mypy)
- [ ] `make test` pasa (tests básicos de loaders)
- [ ] Todos los archivos del directory tree existen excepto decoder/ y prompt/

---

### Phase 2: Prompt Engineering

**Objetivo**: Prompt builder que genera el input completo para el modelo.

#### Task 2.1: Prompt builder module
- **Archivo**: `src/prompt/__init__.py`, `src/prompt/prompt_builder.py`
- **Implementar**:
  ```python
  from src.models.function_definition import FunctionDef

  SYSTEM_PROMPT = """You are a function calling assistant. Given the user's query, you must output a JSON object that calls the most appropriate function.

Available functions:

{function_list}

Output ONLY a JSON object with "name" and "parameters" fields. No explanation."""

  def build_function_list(functions: list[FunctionDef]) -> str:
      """Build numbered list of function descriptions."""
      parts = []
      for i, fn in enumerate(functions, 1):
          params = ", ".join(
              f"{pname} ({pinfo['type']})"
              for pname, pinfo in fn.parameters.items()
          )
          parts.append(f"{i}. {fn.name}: {fn.description}\n   Parameters: {params}")
      return "\n\n".join(parts)

  def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
      """Build complete prompt for a single user query."""
      function_list = build_function_list(functions)
      return SYSTEM_PROMPT.format(function_list=function_list) + f"\nUser query: {user_prompt}"
  ```
- **Acceptance criteria**: prompt generado contiene todas las funciones; es tokenizable por el modelo; formato claro
- **Dependencies**: Phase 1 Task 1.6

#### Task 2.2: Prompt validation
- **Archivo**: `src/prompt/prompt_builder.py` (extender)
- **Implementar**: testear que el prompt generado:
  1. Contiene todos los nombres de funciones
  2. Contiene todos los parámetros
  3. Es una string no vacía
- **Acceptance criteria**: unit tests pasan
- **Dependencies**: Task 2.1

#### Task 2.3: Prompt tests
- **Archivo**: `tests/test_prompt_builder.py`
- **Implementar**: tests con las 5 funciones del proyecto
- **Acceptance criteria**: tests pasan
- **Dependencies**: Task 2.1

#### Definition of Done — Phase 2
- [ ] `build_prompt(functions, "What is 2+3?")` retorna string con las 5 funciones
- [ ] Prompt es tokenizable: `model.encode(prompt)` no falla
- [ ] Tests de prompt pasan
- [ ] `make lint` sigue pasando

---

### Phase 3: Decoder Core

**Objetivo**: State machine, trie, schema validator, token filter — todos testables sin modelo.

#### Task 3.1: Decoder state machine
- **Archivo**: `src/decoder/state.py`
- **Implementar**:
  - Dataclass `DecoderState` con `__slots__`
  - Enum `DecoderPhase` con todos los estados (ROOT, OBJECT_OPEN, IN_OBJECT, etc.)
  - Método `_advance_char(char) -> bool`: transiciones de la state machine
  - Método `simulate(token_text) -> tuple[bool, DecoderState]`: copy + advance chars
  - Método `update_from_text(token_text)`: avanza el estado real
  - Tracking de: current_key, keys_enclosed (set), depth (nesting level)
  - Método `expected_first_chars() -> set[str]`: chars válidos para el próximo token
- **Acceptance criteria**: transiciones correctas para `{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}` char-by-char; tests pasan
- **Dependencies**: Phase 1 Task 1.6

#### Task 3.2: Trie
- **Archivo**: `src/decoder/trie.py`
- **Implementar**:
  - Dataclass `TrieNode` con `__slots__`
  - Función `build_trie(names: list[str]) -> TrieNode`
  - Método `find_node(prefix: str) -> TrieNode | None`
  - Método `valid_next_chars(prefix: str) -> set[str]`: chars que mantienen al menos un nombre como prefix
  - Método `is_complete_name(prefix: str) -> bool`: el prefix es exactamente un nombre
- **Acceptance criteria**: `build_trie(["fn_add_numbers", "fn_greet"])` funciona; `valid_next_chars("fn_a")` retorna `{"d"}`; tests pasan
- **Dependencies**: Phase 1 Task 1.6

#### Task 3.3: Schema validator
- **Archivo**: `src/decoder/schema_validator.py`
- **Implementar**:
  - Clase `SchemaContext` que mantiene: función seleccionada, parámetros requeridos, parámetros ya seteados
  - Método `update(state: DecoderState)`: refresca el contexto basado en el estado actual
  - Método `current_expected_type() -> str | None`: tipo esperado para el value actual
  - Método `required_keys_remaining() -> set[str]`: keys que faltan
  - Método `all_required_present() -> bool`: si todos los required están seteados
  - Método `can_close_params() -> bool`: si se puede cerrar el objeto parameters
- **Acceptance criteria**: dado un state en VALUE_START de key "a" con type "number", retorna "number"; tests pasan
- **Dependencies**: Phase 1 Task 1.6, Phase 1 Task 1.8

#### Task 3.4: Token filter
- **Archivo**: `src/decoder/token_filter.py`
- **Implementar**:
  - Función `compute_allowed_ids(state, schema, vocab, trie, logits) -> set[int]`:
    1. Obtener expected_first_chars del state
    2. Pre-filter: traer tokens de tokens_starting_with por esos chars
    3. Para cada candidate: simulate token through state machine
    4. Post-filter: verificar schema constraints (types, trie matching, closing rules)
    5. Retornar set de allowed ids
  - Manejo especial para tokens con bytes UTF-8 incompletos
  - Bloqueo de '}' de parameters hasta all_required_present
- **Acceptance criteria**: dado vocab mock y state en ROOT, retorna tokens que empiezan con '{'; dado state en KEY_START de "name", solo retorna prefixes de function names; tests pasan
- **Dependencies**: Tasks 3.1, 3.2, 3.3

#### Task 3.5: Decoder unit tests
- **Archivos**: `tests/test_state.py`, `tests/test_trie.py`, `tests/test_token_filter.py`
- **Implementar**: tests comprehensivos con vocabularios mock
- **Acceptance criteria**: >80% coverage del decoder; todos los edge cases testeados
- **Dependencies**: Tasks 3.1-3.4

#### Definition of Done — Phase 3
- [ ] State machine procesa `{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}` sin errores
- [ ] Trie construido desde 5 funciones, prefix matching funciona
- [ ] Token filter con mock vocab retorna allowed tokens correctos
- [ ] Todos los tests de decoder pasan
- [ ] `make lint` sigue pasando
- [ ] El decoder es COMPLETAMENTE testeable sin modelo

---

### Phase 4: Generation Loop

**Objetivo**: Loop de generación constrained que conecta el decoder con el modelo.

#### Task 4.1: Constrained generator
- **Archivo**: `src/decoder/constrained_generator.py`
- **Implementar**:
  ```python
  from llm_sdk.llm_sdk import Small_LLM_Model
  from src.decoder.state import DecoderState
  from src.decoder.token_filter import compute_allowed_ids
  from src.loader.vocab_loader import Vocab

  MAX_TOKENS = 200

  def generate(
      model: Small_LLM_Model,
      prompt: str,
      vocab: Vocab,
      functions: list[FunctionDef],
      trie: TrieNode,
      max_tokens: int = MAX_TOKENS,
  ) -> tuple[str, bool]:
      """Generate constrained JSON output for a prompt.

      Returns:
          (generated_text, success_flag)
      """
      input_ids = model.encode(prompt).tolist()
      state = DecoderState()
      schema = SchemaContext(functions)

      for step in range(max_tokens):
          logits = model.get_logits_from_input_ids(input_ids)
          allowed = compute_allowed_ids(state, schema, vocab, trie, logits)

          if not allowed:
              # Empty set handling: attempt repair or break
              break

          # Argmax over allowed tokens
          best_id = max(allowed, key=lambda tid: logits[tid])
          input_ids.append(best_id)

          token_text = vocab.id2token[best_id]
          state.update_from_text(token_text)

          if state.phase == DecoderPhase.COMPLETE:
              break

      generated = model.decode(input_ids[len(model.encode(prompt).tolist()):])
      return generated, state.phase == DecoderPhase.COMPLETE
  ```
- **Acceptance criteria**: con prompt "What is 2+3?" genera JSON con "fn_add_numbers"; timing ~200ms per token
- **Dependencies**: Phase 3 (all tasks), Phase 2 (Task 2.1)

#### Task 4.2: Single-prompt smoke test
- **Archivo**: script temporal o test manual
- **Implementar**: ejecutar generate() con "What is 2+3?" y verificar output
- **Acceptance criteria**: output parseable como JSON, contiene "fn_add_numbers"
- **Dependencies**: Task 4.1

#### Task 4.3: All-prompts test
- **Archivo**: script temporal
- **Implementar**: ejecutar generate() con los 11 prompts, medir timing, imprimir resultados
- **Acceptance criteria**: todos los prompts producen output; total <5 min; accuracy ≥90%
- **Dependencies**: Task 4.2

#### Definition of Done — Phase 4
- [ ] Un prompt → JSON válido con function correcta
- [ ] 11 prompts → todos producen output
- [ ] Timing total <5 min en CPU
- [ ] Accuracy ≥90% (function + args correctos)
- [ ] `make lint` sigue pasando

---

### Phase 5: Validation + Pipeline

**Objetivo**: Validación post-generación + orquestación completa end-to-end.

#### Task 5.1: Output validator
- **Archivo**: `src/validator/__init__.py`, `src/validator/output_validator.py`
- **Implementar**:
  ```python
  import json
  from pydantic import ValidationError
  from src.models.output import FunctionCall
  from src.models.function_definition import FunctionDef

  def validate_output(
      raw_output: str,
      functions: list[FunctionDef],
  ) -> FunctionCall | str:
      """Validate generated output against function definitions.

      Returns FunctionCall on success, error message on failure.
      """
      try:
          parsed = json.loads(raw_output)
      except json.JSONDecodeError as e:
          return f"Invalid JSON: {e}"

      try:
          call = FunctionCall(**parsed)
      except ValidationError as e:
          return f"Schema validation error: {e}"

      # Check function name exists
      valid_names = {fn.name for fn in functions}
      if call.name not in valid_names:
          return f"Unknown function: {call.name}"

      # Check parameter types match schema
      fn_def = next(f for f in functions if f.name == call.name)
      for pname, pvalue in call.parameters.items():
          if pname not in fn_def.parameters:
              return f"Unknown parameter: {pname}"
          expected = fn_def.parameters[pname]["type"]
          if not _type_matches(pvalue, expected):
              return f"Type mismatch for {pname}: expected {expected}, got {type(pvalue).__name__}"

      # Check no extra parameters
      extra = set(call.parameters.keys()) - set(fn_def.parameters.keys())
      if extra:
          return f"Unexpected parameters: {extra}"

      return call

  def _type_matches(value, expected_type: str) -> bool:
      """Check if a value matches the expected JSON type."""
      if expected_type == "string":
          return isinstance(value, str)
      if expected_type == "number":
          return isinstance(value, (int, float))
      if expected_type == "boolean":
          return isinstance(value, bool)
      return True  # null or unknown types pass
  ```
- **Acceptance criteria**: valida JSONs válidos e inválidos; chequea names, types, no extras
- **Dependencies**: Phase 1 Task 1.6, Phase 1 Task 1.8

#### Task 5.2: Pipeline integration
- **Archivo**: `src/pipeline.py` (extender)
- **Implementar**: integrar todo:
  ```python
  def run(args) -> int:
      """Full pipeline: load → generate → validate → output."""
      functions = load_functions(args.functions_definition)
      prompts = load_prompts(args.input)
      model = Small_LLM_Model()
      vocab = load_vocab(model)
      trie = build_trie([fn.name for fn in functions])

      results = []
      success_count = 0

      for prompt_text in prompts:
          full_prompt = build_prompt(functions, prompt_text)
          generated, is_complete = generate(model, full_prompt, vocab, functions, trie)

          validation = validate_output(generated, functions)
          if isinstance(validation, FunctionCall):
              results.append(validation.model_dump())
              success_count += 1
          else:
              # Log error, append empty/error result
              results.append({"error": str(validation), "prompt": prompt_text})

      # Write output
      output_path = args.output
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, 'w') as f:
          json.dump(results, f, indent=2)

      # Summary
      print(f"Processed {len(prompts)} prompts, {success_count} successful")
      return 0 if success_count > 0 else 1
  ```
- **Acceptance criteria**: ejecución completa genera `data/output/function_calls.json` con 11 entries
- **Dependencies**: Tasks 4.1, 5.1

#### Task 5.3: Output format verification
- **Archivo**: verificación manual o test
- **Implementar**: verificar que el output tiene el formato exacto del subject V.4.1
- **Acceptance criteria**: cada entry tiene "name" (string) y "parameters" (object)
- **Dependencies**: Task 5.2

#### Definition of Done — Phase 5
- [ ] `uv run python -m src` genera `data/output/function_calls.json`
- [ ] Output tiene 11 entries con formato correcto
- [ ] Validación post-generación funciona
- [ ] Pipeline es robusto a prompts problemáticos
- [ ] `make lint` sigue pasando

---

### Phase 6: Error Handling + Polish

**Objetivo**: Todos los paths de error, README completo, tests comprehensivos.

#### Task 6.1: Complete error handling
- **Archivos**: todos los módulos
- **Implementar**:
  - FileNotFoundError para archivos que no existen
  - JSONDecodeError para JSON malformado
  - ValueError para duplicados
  - Graceful skip en prompts problemáticos
  - Exit code 1 en falla crítica
  - Logging consistente (warnings en stderr, info en stdout)
- **Acceptance criteria**: archivos faltantes → exit 1 con mensaje claro; prompts malos → skip + warning
- **Dependencies**: Phase 5

#### Task 6.2: README completo
- **Archivo**: `README.md`
- **Implementar**: descripción, quick start (`make install && make run`), arquitectura, output format
- **Acceptance criteria**: un dev nuevo puede correr el proyecto con solo el README
- **Dependencies**: Phase 5

#### Task 6.3: Comprehensive tests
- **Archivos**: `tests/` completos
- **Implementar**: tests para todos los módulos, edge cases, error paths
- **Acceptance criteria**: >70% coverage; `make test` pasa
- **Dependencies**: Phases 3-5

#### Task 6.4: Final lint
- **Archivos**: todo el proyecto
- **Implementar**: flake8 + mypy clean
- **Acceptance criteria**: `make lint` pasa sin warnings
- **Dependencies**: Tasks 6.1-6.3

#### Definition of Done — Phase 6
- [ ] Todos los paths de error manejan correctamente
- [ ] README completo y útil
- [ ] Tests comprehensivos pasan
- [ ] `make lint` pasa limpio
- [ ] `uv run python -m src` funciona end-to-end
- [ ] Output cumple todos los criterios del subject

---

### Phase 7: Bonus (ANNEX ONLY)

> **No parte del main process. Evaluar solo después del MVP funcional.**

Esta fase se detalla en la Parte B de este documento. Incluye:
- B1: lint-strict Makefile target
- B2: Multi-modelo support
- B3: Recode tokenizer
- B4: Error recovery avanzado
- B5: Performance optimizations
- B6: Test suite comprehensiva
- B7: Visualización de generación
- B8: Nested arguments
- B9: Public encode/decode API

Cada bonus requiere su propia tarea de diseño y evaluación de tradeoffs antes de implementar.

---

> **Fin del plan. lsto para sdd-tasks → lama-apply.**
