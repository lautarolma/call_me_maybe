# PLAN DIDÁCTICO — call_me_maybe

> **Proyecto**: call_me_maybe (42 school — LLM function calling con constrained decoding)
> **Ingeniero didáctico**: lama-onboard (mimo-v2.5-free engine)
> **Fecha**: 2026-08-18
> **Público**: Estudiante de 42 school, background en C (libft, push_swap), Python intermedio (piscine + A-maze-ing)
> **Formato**: Documento autocontenido para Google NotebookLM — cada módulo es independiente

---

## Cómo leer este documento

Cada módulo es **autocontenido**. Si NotebookLM toma un módulo solo, entiende todo lo que necesita sin depender de otros módulos. Los conceptos se re-explican brevemente cuando aparecen en módulos posteriores — no se asume que leíste los anteriores.

Los módulos siguen el **orden exacto de construcción**. Primero aprendés el concepto, después lo construís. Así de simple.

---

# M0: Mapa del viaje

## Qué vas a aprender acá

En este módulo entendés **qué carajo estamos construyendo**, por qué importa, y cuál es el mapa completo del viaje. No escribís código acá — solo armar el mapa en tu cabeza.

## La idea central: function calling con constrained decoding

Imaginate que le preguntás a una IA: "¿Cuánto es 2 + 3?" y la IA te responde con un JSON como este:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

Eso es **function calling**: en vez de responder "es 5", la IA **elige la función correcta** y **arma los parámetros**. Como si le dijeras "no me des la respuesta, decime a qué función la pasaría".

**El problema**: los modelos de LLM generan texto libre. Si le pedís JSON, puede generar basura: comas faltantes, keys inexistentes, strings sin cerrar. Es como pedirle a alguien que nunca vio un molde de ladrillo que fabrique un ladrillo perfecto — va a salir todo chueco.

**La solución**: **constrained decoding**. En vez de dejar que el modelo genere lo que quiera, **lo obligamos** a seguir las reglas del JSON en cada token que genera. Como poner rieles a un tren — puede ir para adelante, pero no puede salirse de la vía.

## ¿Qué es un MVP?

**MVP = Minimum Viable Product** (Producto Mínimo Viable). Es la versión más simple que **funciona**. No es la versión bonita, no es la versión optimizada — es la versión que hace lo básico sin romperse.

¿Por qué empezamos con un MVP? Porque en ingeniería de software, construir todo de una es una receta para el desastre. Preferimos tener algo que funciona **ahora** y mejorarlo después, que planear 6 meses y nunca entregar.

Nuestro MVP: un pipeline que lee funciones de un JSON, recibe prompts del usuario, genera JSON válido con constrained decoding, y escribe el output. Sin fancy stuff. Sin optimización. Solo funciona.

## Las 6 fases del proyecto

El proyecto se divide en 6 fases. Cada fase construye sobre la anterior, como pisos de un edificio:

```
Fase 1: FOUNDATION (cimientos)
  → Estructura del proyecto, carga de archivos, modelos de datos
  → Sin esto, no hay nada donde pararse

Fase 2: PROMPT ENGINEERING (el plano)
  → Diseñar el prompt que le dice al modelo qué hacer
  → Sin esto, el modelo no sabe qué función elegir

Fase 3: DECODER CORE (el motor)
  → State machine, trie, filtro de tokens — el corazón del constrained decoding
  → Sin esto, no podemos restringir al modelo

Fase 4: GENERATION LOOP (la cinta de producción)
  → Conectar el decoder con el modelo en un loop
  → Sin esto, el decoder no hace nada

Fase 5: VALIDATION + PIPELINE (control de calidad)
  → Validar output, orquestar todo, escribir resultados
  → Sin esto, no sabemos si funciona

Fase 6: HARDING + POLISH (pulir y asegurar)
  → Error handling, tests, README, lint
  → Sin esto, no está listo para entregar
```

## El flujo completo (vista de pájaro)

```
Usuario pasa un prompt: "What is 2+3?"
         │
         ▼
┌─────────────────────┐
│ 1. CARGAR FUNCIONES  │  ← Leo el JSON con las 5 funciones disponibles
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. CONSTRUIR PROMPT  │  ← Armo un string: "Tens estas funciones... User: What is 2+3?"
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. TOKENIZAR         │  ← Convierto el string a números (tokens) que el modelo entiende
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────────────┐
│ 4. LOOP DE GENERACIÓN CONSTRAINED (50 iteraciones)│
│                                                   │
│  ┌──────┐    ┌──────────┐    ┌─────────┐         │
│  │Modelo│───▶│  Logits   │───▶│ Filtro  │───▶ Token │
│  └──────┘    └──────────┘    └─────────┘    elegido │
│       ▲                                           │
│       └───────────────────────────────────────────┘
│                                                   │
│  En cada paso:                                    │
│  a) Modelo dice "creo que el siguiente token es X" │
│  b) Filtro dice "X no es válido, probá con Y"     │
│  c) Elegimos el mejor token válido                │
│  d) Actualizamos el estado del JSON               │
│  e) Si el JSON está completo, PARA                │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────┐
│ 5. VALIDAR OUTPUT    │  ← ¿El JSON es válido? ¿Los nombres existen? ¿Los tipos son correctos?
└─────────┬───────────┘
          ▲
          │
┌─────────┴───────────┐
│ 6. ESCRIBIR ARCHIVO  │  ← Guardo el resultado en data/output/function_calls.json
└─────────────────────┘
```

## ¿Por qué esto importa?

El constrained decoding no es un ejercicio académico. Empresas como OpenAI, Anthropic y Google lo usan en producción. Cuando usás ChatGPT y le pedís que use una herramienta, hay un constrained decoder asegurándose de que el JSON que genera sea válido.

Estás construyendo algo que se usa en el mundo real, desde cero, con un modelo de 0.6B parámetros (que es chiquitito). Si funciona acá, funciona en cualquier lado.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **LLM** | Large Language Model — un modelo de IA que genera texto (como ChatGPT, pero en versión chiquita) |
| **Function calling** | Que el LLM elija una función y arme sus parámetros en vez de responder texto libre |
| **Constrained decoding** | Restringir qué tokens puede generar el LLM en cada paso, forzándolo a cumplir reglas (como JSON válido) |
| **MVP** | Minimum Viable Product — la versión más simple que funciona |
| **Token** | La unidad más pequeña de texto que el modelo procesa. No es una palabra ni un carácter — es algo intermedio definido por el tokenizador |
| **Logit** | Un número que indica cuánto "prefiere" el modelo un token determinado. Más alto = más probable |
| **JSON** | JavaScript Object Notation — formato de datos como `{"nombre": "valor"}` |

## Checkpoint — ¿lo entendiste?

1. **¿Qué es function calling?** ¿Por qué el modelo no simplemente responde "es 5"?

<details><summary>Respuesta</summary>Function calling es que el LLM elija una función predefinida y arme sus parámetros en vez de generar texto libre. Es útil cuando queremos que la IA interactúe con sistemas externos (bases de datos, APIs, etc.) en vez de solo dar respuestas de texto.</details>

2. **¿Qué problema resuelve el constrained decoding?**

<details><summary>Respuesta</summary>Los LLMs generan texto libre y pueden producir JSON inválido (comas faltantes, keys incorrectas, etc.). El constrained decoding lo obliga a seguir las reglas del JSON en cada token, garantizando output válido por construcción.</details>

3. **¿Cuáles son las 6 fases del proyecto?** Nombralas en orden.

<details><summary>Respuesta</summary>Foundation → Prompt Engineering → Decoder Core → Generation Loop → Validation + Pipeline → Hardening + Polish</details>

4. **¿Qué es un MVP y por qué empezamos con eso?**

<details><summary>Respuesta</summary>MVP = Minimum Viable Product. Empezamos con eso porque es mejor tener algo que funciona ahora y mejorarlo después, que planificar todo y nunca entregar. El MVP hace lo básico sin romperse.</details>

5. **¿Por qué el constrained decoding es importante en la industria?**

<details><summary>Respuesta</summary>Empresas como OpenAI, Anthropic y Google lo usan en producción para garantizar que el output de los LLMs sea estructurado y confiable. Cuando usás ChatGPT con herramientas, hay un constrained decoder trabajando detrás.</details>

---

# M1: El terreno

## Qué vas a aprender acá

En este módulo armás la **estructura del proyecto**: archivos de configuración, carpetas, entry point. Es como preparar el terreno antes de construir un edificio — sin esto, no hay dónde poner los ladrillos.

## pyproject.toml: el acta de nacimiento del proyecto

Todo proyecto Python serio tiene un archivo que dice "este proyecto se llama así, necesita estas dependencias, y se construye así". Ese archivo es `pyproject.toml`.

**Analogía**: es como el plano registral de una propiedad. Dice qué es, dónde está, y qué tiene adentro.

```toml
# pyproject.toml — el archivo de configuración del proyecto
[project]
name = "call-me-maybe"
version = "0.1.0"
description = "LLM function calling with constrained decoding"
requires-python = ">=3.10"   # Necesitamos Python 3.10+ (para match/case y type hints modernos)
dependencies = [
    "llm-sdk",              # El SDK que nos dan en 42 — el modelo, tokenizer, todo
    "pydantic>=2.0.0",      # Para validar datos con tipos
    "numpy>=1.24.0",        # Para operaciones numéricas (argmax sobre logits)
]

# Cómo se construye el paquete
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# El SDK es una dependencia LOCAL, no de PyPI
[tool.uv.sources]
llm-sdk = { path = "./llm_sdk", editable = true }
```

**¿Qué es `requires-python = ">=3.10"`?** Es como decir "esta casa solo se puede habitar si tenés calefacción". Python 3.10 introdujo `match/case` (como switch de C pero mejorado) y tipos como `str | None` que usamos en todo el proyecto.

## uv: el gerente de dependencies

**uv** es un gestor de paquetes para Python. Piensa en él como `apt` o `brew` pero para bibliotecas Python. Cuando ejecutás `uv sync`, uv lee el `pyproject.toml`, descarga las dependencias, y arma un entorno virtual.

```bash
# Instalar dependencias
uv sync

# Correr el proyecto
uv run python -m src

# Correr un solo script
uv run python mi_script.py
```

**¿Por qué uv y no pip?** uv es 10-100x más rápido que pip. Para un proyecto con torch (que pesa ~2GB), la diferencia es enorme. Además, uv maneja los lock files automáticamente.

## Makefile: los atajos de teclado del proyecto

Un Makefile es una lista de comandos abreviados. En vez de escribir `uv run python -m src --help` cada vez, escribís `make debug`.

```makefile
# Makefile — atajos para el proyecto
.PHONY: install run debug clean lint test

install:          # make install → instala dependencias
	uv sync

run:              # make run → ejecuta el pipeline completo
	uv run python -m src

debug:            # make debug → muestra help del CLI
	uv run python -m src --help

clean:            # make clean → limpia archivos temporales
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ src/*/__pycache__
	rm -rf tests/__pycache__

lint:             # make lint → chequea estilo y tipos
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

test:             # make test → corre los tests
	pytest tests/ -v
```

**`.PHONY`** le dice a make "estos no son archivos reales, son comandos". Sin esto, si tenés un archivo llamado `clean`, make pensaría que ya está hecho y no haría nada.

## Estructura de paquetes Python

Python organiza el código en **paquetes** (carpetas con `__init__.py`) y **módulos** (archivos `.py`). Nuestro proyecto:

```
call_me_maybe/
├── pyproject.toml          # Config del proyecto
├── Makefile                # Atajos
├── llm_sdk/                # SDK proporcionado (NO tocar)
├── data/
│   ├── input/              # JSONs de entrada
│   └── output/             # Resultados (se crea en runtime)
├── src/                    # NUESTRO CÓDIGO
│   ├── __init__.py         # "Esto es un paquete Python"
│   ├── __main__.py         # Entry point: por dónde empieza todo
│   ├── cli.py              # Parser de argumentos de línea de comandos
│   ├── models/             # Modelos de datos (Pydantic)
│   ├── loader/             # Carga de archivos
│   ├── prompt/             # Construcción de prompts
│   ├── decoder/            # El motor de constrained decoding
│   ├── validator/          # Validación post-generación
│   └── pipeline.py         # Orquestador: conecta todo
└── tests/                  # Tests unitarios
```

## __init__.py y __main__.py

**`__init__.py`** es como el letrero de una tienda que dice "acá hay un paquete". Puede estar vacío o tener un docstring:

```python
# src/__init__.py
"""call_me_maybe — LLM function calling with constrained decoding."""
```

**`__main__.py`** es el **entry point** — el archivo que Python ejecuta cuando hacés `python -m src`. Es como el `main()` de C, pero para paquetes:

```python
# src/__main__.py
"""Entry point: ejecuta el pipeline cuando corás `python -m src`."""
import sys
from src.cli import parse_args
from src.pipeline import run

def main() -> None:
    """Parse args y ejecuta el pipeline."""
    try:
        args = parse_args()
        exit_code = run(args)
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**¿Qué es `if __name__ == "__main__"`?** Cuando Python ejecuta un archivo directamente, le pone el nombre `"__main__"`. Cuando lo importa desde otro archivo, le pone el nombre del módulo. Este bloque dice "ejecutá esto solo si me corren directamente, no si me importan".

## CLI: argumentos de línea de comandos

El **CLI** (Command Line Interface) es cómo el usuario le dice al programa qué hacer. Usamos `argparse` (viene con Python, no hay que instalar nada):

```python
# src/cli.py
import argparse
from pathlib import Path

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="LLM function calling with constrained decoding"
    )
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

**¿Qué es `Path`?** Es una forma moderna de manejar rutas de archivos en Python. En vez de usar strings como `"data/input/functions_definition.json"`, usás `Path("data/input/functions_definition.json")`. La ventaja: funciona igual en Windows, Linux, y Mac.

**¿Qué es `argparse.Namespace`?** Es básicamente un objeto con atributos. Si el usuario pasa `--input mi_archivo.json`, el namespace tiene `args.input == Path("mi_archivo.json")`.

## Type hints: el molde de los datos

Los **type hints** son anotaciones que dicen qué tipo de dato es cada variable, parámetro, y retorno. No afectan la ejecución (Python los ignora), pero ayudan a IDEs, mypy, y a vos a entender el código:

```python
def mi_funcion(nombre: str, edad: int) -> str:
    """Recibe nombre y edad, retorna un saludo."""
    return f"Hola {nombre}, tenés {edad} años"
```

**¿Por qué importan?** En un proyecto con 15+ archivos, los type hints son como los planos de un arquitecto — sin ellos, no sabés qué va dónde. Además, `mypy` (un chequeador de tipos estático) los usa para encontrar errores ANTES de correr el código.

## .gitignore: qué NO subir a git

El `.gitignore` le dice a git "ignorá estos archivos". Cosas que no queremos en el repo:

```
# .gitignore
data/output/         # Los resultados se generan, no se guardan
__pycache__/         # Archivos compilados de Python
.venv/               # Entorno virtual
.mypy_cache/         # Cache de mypy
*.egg-info/          # Info de paquetes
uv.lock             # Se genera con uv sync
.pytest_cache/       # Cache de pytest
```

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **pyproject.toml** | Archivo de configuración del proyecto Python: nombre, dependencias, cómo construir |
| **uv** | Gestor de paquetes Python rápido (reemplazo moderno de pip) |
| **Makefile** | Lista de comandos abreviados (`make install`, `make run`, etc.) |
| **Paquete** | Carpeta con `__init__.py` que Python trata como unidad de código |
| **Entry point** | Punto de entrada del programa — dónde empieza la ejecución |
| **CLI** | Command Line Interface — argumentos que el usuario pasa por terminal |
| **Type hints** | Anotaciones de tipo en el código (`def foo(x: int) -> str`) |
| **mypy** | Herramienta que chequea type hints estáticamente (encuentra errores sin correr el código) |
| **.gitignore** | Archivo que le dice a git qué archivos ignorar |
| **Path** | Clase de Python para manejar rutas de archivos de forma moderna |

## Checkpoint — ¿lo entendiste?

1. **¿Qué hace `uv sync`?**

<details><summary>Respuesta</summary>Lee el pyproject.toml, descarga todas las dependencias, y arma el entorno virtual del proyecto. Es como `npm install` pero para Python.</details>

2. **¿Cuál es la diferencia entre `__init__.py` y `__main__.py`?**

<details><summary>Respuesta</summary>`__init__.py` marca una carpeta como paquete Python. `__main__.py` es el entry point — el código que se ejecuta cuando hacés `python -m src`.</details>

3. **¿Por qué usamos `Path` en vez de strings para rutas?**

<details><summary>Respuesta</summary>Path funciona igual en todos los sistemas operativos (Windows, Linux, Mac) y tiene métodos útiles como `.parent`, `.exists()`, `.mkdir()`. Los strings son más propensos a errores con separadores de ruta.</details>

4. **¿Qué pasa si no existe `__init__.py` en una carpeta?**

<details><summary>Respuesta</summary>Python 3.3+ soporta "namespace packages" sin `__init__.py`, pero es mejor practice tenerlo. Sin él, algunos tools (como mypy o pytest) pueden tener problemas para encontrar el paquete.</details>

5. **¿Por qué `.PHONY` en el Makefile?**

<details><summary>Respuesta</summary>Le dice a make que esos targets no son archivos reales. Sin `.PHONY`, si existe un archivo llamado `clean`, make pensaría que el target ya está completo y no ejecutaría nada.</details>

---

# M2: Pydantic y modelos de datos

## Qué vas a aprender acá

En este módulo conocés **Pydantic**, la herramienta que usamos para definir y validar la estructura de los datos que maneja nuestro programa. Es como poner un molde a cada pieza de LEGO — si no entra en el molde, no va.

## ¿Qué es Pydantic?

**Pydantic** es una biblioteca Python que te deja definir **modelos de datos** con tipos y restricciones. Cuando creás un modelo Pydantic, Pydantic **valida automáticamente** que los datos cumplan las reglas. Si no cumplan, lanza un error claro.

**Analogía**: imaginá que estás armando un formularo de inscripción. Pydantic es como poner reglas en cada campo: "nombre tiene que ser texto", "edad tiene que ser un número entre 0 y 120", "email tiene que tener @". Si alguien llena mal el formulario, Pydantic le dice exactamente qué está mal.

```python
from pydantic import BaseModel, Field

class Persona(BaseModel):
    """Modelo de una persona."""
    nombre: str = Field(description="Nombre completo")
    edad: int = Field(ge=0, le=120, description="Edad en años")
    email: str = Field(description="Correo electrónico")

# Esto FUNCIONA — los datos son válidos
persona = Persona(nombre="Juan", edad=25, email="juan@mail.com")

# Esto FALLA — Pydantic lanza ValidationError
persona = Persona(nombre="Juan", edad=-5, email="juan@mail.com")
# Error: Input should be greater than or equal to 0
```

**¿Qué es `BaseModel`?** Es la clase base de Pydantic. Todos tus modelos heredan de ella. Cuando instanciás un modelo, Pydantic toma los argumentos, los valida contra los tipos y restricciones, y te da el objeto validado.

**¿Qué es `Field`?** Es una forma de agregar metadata a cada campo: descripciones, restricciones (como `ge=0` para "greater or equal"), valores por defecto, etc.

## Nuestros modelos de datos

Tenemos dos tipos de modelos: **input** (lo que leemos) y **output** (lo que generamos).

### Modelos de input: ¿qué funciones existen?

```python
# src/models/function_definition.py
from pydantic import BaseModel, Field

class ParameterDef(BaseModel):
    """Definición de un parámetro de función."""
    name: str = Field(description="Nombre del parámetro, ej: 'a'")
    type: str = Field(description="Tipo: 'string', 'number', 'boolean', 'null'")

class FunctionDef(BaseModel):
    """Definición completa de una función disponible."""
    name: str = Field(description="Nombre de la función, ej: 'fn_add_numbers'")
    description: str = Field(description="Descripción legible por humanos")
    parameters: dict[str, dict[str, str]] = Field(
        description="Parámetros: nombre → {type: ...}"
    )
    returns: dict[str, str] = Field(
        description="Info del tipo de retorno (solo para el prompt)"
    )
```

**¿Por qué `dict[str, dict[str, str]]` para parameters?** Porque el JSON de entrada tiene esta forma:

```json
{
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  }
}
```

Es un diccionario donde las keys son nombres de parámetros, y los values son objetos con la key "type". Pydantic valida que cada value tenga la estructura correcta.

### Modelos de output: ¿qué generamos?

```python
# src/models/output.py
from pydantic import BaseModel, Field

class FunctionCall(BaseModel):
    """Una llamada a función generada por el modelo."""
    name: str = Field(description="Nombre de la función a llamar")
    parameters: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Argumentos de la función"
    )
```

**¿Qué es `default_factory=dict`?** Significa que si no pasás `parameters`, se crea un diccionario vacío automáticamente. Es como decir "si no hay parámetros, poné un dict vacío".

**¿Por qué `str | int | float | bool | None`?** Porque los parámetros pueden ser de cualquier tipo JSON: strings, números, booleanos, o null. Python 3.10+ permite esta sintaxis de unión de tipos.

## Pydantic vs dataclasses: ¿cuándo usar cada uno?

**Dataclasses** (de Python stdlib) son más simples y rápidos. **Pydantic** tiene validación automática pero es más lento.

Regla en nuestro proyecto:
- **Pydantic** para I/O: modelos que leemos de archivos o escribimos a archivos. Necesitamos validación.
- **Dataclasses** para el inner loop: el decoder genera tokens a 200ms por step. No podemos pagar el overhead de Pydantic ahí.

**Analogía**: Pydantic es como un inspector de calidad que revisa cada pieza que entra o sale de la fábrica. Dataclass es como un operario rápido que trabaja dentro de la línea de producción — no revisa, solo construye.

## ValidationError: cuando algo sale mal

Cuando los datos no cumplen las reglas, Pydantic lanza `ValidationError`:

```python
from pydantic import ValidationError

try:
    fn = FunctionDef(
        name="",           # ← problema: string vacío no es válido para nuestro uso
        description="Test",
        parameters={},
        returns={}
    )
except ValidationError as e:
    print(e)
    # 1 validation error for FunctionDef
    # name
    #   Input should be a valid string [type=string_type, ...]
```

**¿Por qué importa?** En nuestro pipeline, si el JSON de funciones tiene un error, queremos un mensaje claro tipo "la función X tiene un parámetro con tipo inválido", no un crash misterioso con un `KeyError`.

## Estrategia de "defense in depth"

Nuestro pipeline tiene **dos niveles de validación**:

1. **Input validation** (Pydantic en loaders): cuando leemos los JSONs de entrada, validamos que tengan la estructura correcta.
2. **Output validation** (Pydantic en validator): cuando el modelo genera un JSON, validamos que sea correcto.

¿Por qué dos? Porque el constrained decoder **garantiza** JSON válido por construcción (eso es su trabajo). Pero el validador post-generación es una **red de seguridad** — por si acaso. En ingeniería, esto se llama "defense in depth" o "defensa en profundidad".

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Pydantic** | Biblioteca Python para definir modelos de datos con validación automática |
| **BaseModel** | Clase base de Pydantic — todos los modelos heredan de ella |
| **Field** | Función para agregar metadata y restricciones a campos de un modelo |
| **ValidationError** | Excepción que lanza Pydantic cuando los datos no cumplen las reglas |
| **Dataclass** | Decorador de Python para crear clases de datos simples y rápidas (sin validación automática) |
| **Defense in depth** | Estrategia de seguridad: múltiples capas de protección, no depender de una sola |
| **Type hint** | Anotación de tipo en el código (`x: int`) — no afecta ejecución, ayuda a tools y humanos |
| **Inner loop** | El loop más crítico y ejecutado del programa — acá la performance importa |

## Checkpoint — ¿lo entendiste?

1. **¿Qué hace Pydantic que un dataclass no hace?**

<details><summary>Respuesta</summary>Pydantic valida automáticamente los datos contra los tipos y restricciones definidos. Un dataclass solo guarda los datos sin validar. Si le pasás un string donde se espera un int, Pydantic lanza error; un dataclass lo acepta silenciosamente.</details>

2. **¿Por qué usamos Pydantic para I/O pero dataclasses para el decoder?**

<details><summary>Respuesta</summary>Pydantic tiene overhead de ~200μs por instanciación. En el inner loop (200 tokens × 11 prompts), eso suma ~110ms innecesarios. El decoder necesita velocidad, no validación en cada token. Los loaders y validators necesitan validación pero no velocidad extrema.</details>

3. **¿Qué es `ValidationError` y cuándo lo ves?**

<details><summary>Respuesta</summary>Es una excepción que Pydantic lanza cuando los datos no cumplen las reglas del modelo. La ves cuando intentás crear un modelo con datos inválidos: tipo incorrecto, string vacío donde se espera contenido, etc.</details>

4. **¿Qué es `default_factory=dict`?**

<details><summary>Respuesta</summary>Es una forma de decirle a Pydantic "si no se pasa este argumento, creá un dict vacío". Evita el problema de mutable defaults (que todos los objetos compartan el mismo dict).</details>

5. **¿Por qué el campo `returns` de FunctionDef no aparece en el output?**

<details><summary>Respuesta</summary>`returns` es metadata de input — le dice al modelo qué retorna cada función para que pueda informar al prompt. El formato de output del subject es solo `{"name": ..., "parameters": {...}}`. No hay campo `returns` en la salida.</details>

---

# M3: Cargadores

## Qué vas a aprender acá

En este módulo construís los **cargadores** — los módulos que leen los archivos JSON de entrada y los convierten en objetos Python validados. Es como los operarios que descargan los materiales de construcción y los revisan antes de que lleguen a la línea de producción.

## Leer JSON: la materia prima

JSON (JavaScript Object Notation) es el formato de datos más popular del mundo. Se ve así:

```json
[
  {"name": "fn_add_numbers", "description": "Add two numbers"},
  {"name": "fn_greet", "description": "Generate a greeting"}
]
```

Python tiene `json` en su biblioteca estándar (stdlib) — no necesitás instalar nada:

```python
import json

with open("data/input/functions_definition.json") as f:
    data = json.load(f)  # Convierte JSON → lista de dicts de Python
```

**¿Qué es `json.load(f)`?** Lee todo el archivo y lo convierte en un objeto Python. Un JSON array se convierte en `list`, un object en `dict`, strings en `str`, numbers en `int` o `float`, etc.

## Cargador de funciones

```python
# src/loader/function_loader.py
from pathlib import Path
import json
from src.models.function_definition import FunctionDef

def load_functions(path: Path) -> list[FunctionDef]:
    """Carga y valida definiciones de funciones. Falla si hay nombres duplicados."""
    with open(path) as f:
        data = json.load(f)

    # Convertir cada item a un modelo Pydantic — valida automáticamente
    functions = [FunctionDef(**item) for item in data]

    # Chequear nombres duplicados
    names = [fn.name for fn in functions]
    dupes = [n for n in names if names.count(n) > 1]
    if dupes:
        raise ValueError(f"Funciones con nombres duplicados: {set(dupes)}")

    return functions
```

**¿Qué hace `FunctionDef(**item)`?** El `**` desempaqueta un dict en argumentos de palabra clave. Es como decir:

```python
# Estas dos líneas son equivalentes:
fn = FunctionDef(**{"name": "fn_add", "description": "Add", "parameters": {}, "returns": {}})
fn = FunctionDef(name="fn_add", description="Add", parameters={}, returns={})
```

Cuando hacés `FunctionDef(**item)`, Pydantic valida que el dict tenga la estructura correcta. Si falta un campo o el tipo es incorrecto, lanza `ValidationError`.

**¿Por qué chequeamos duplicados?** Porque nuestro decoder usa un **trie** (ver M7) para restringir los nombres de función. Si hay dos funciones con el mismo nombre, el trie no sabe cuál elegir. Además, es un error de datos que debemos detectar temprano.

## Cargador de prompts

```python
# src/loader/input_loader.py
from pathlib import Path
import json

def load_prompts(path: Path) -> list[str]:
    """Carga prompts desde un archivo JSON. Soporta lista de strings o listas de dicts."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Se esperaba una lista no vacía en {path}")

    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        else:
            raise ValueError(f"Formato de prompt inválido: {item}")

    return prompts
```

**¿Por qué dos formatos?** Porque el JSON de entrada puede ser una lista de strings (`["What is 2+3?", "Hello"]`) o una lista de objetos con key "prompt" (`[{"prompt": "What is 2+3?"}]`). El subject de 42 puede usar cualquiera de los dos. Nuestro cargador maneja ambos.

## Cargador de vocabulario

Este es el cargador más importante y complejo. El vocabulario es el diccionario que el modelo usa para convertir texto a tokens y viceversa.

```python
# src/loader/vocab_loader.py
import json
from dataclasses import dataclass, field
from llm_sdk.llm_sdk import Small_LLM_Model

@dataclass
class Vocab:
    """Vocabulario del modelo con índices pre-computados."""
    token2id: dict[str, int]           # token_text → token_id
    id2token: dict[int, str]           # token_id → token_text
    tokens_starting_with: dict[str, set[int]]  # primer_char → set de token_ids
    vocab_size: int

def load_vocab(model: Small_LLM_Model) -> Vocab:
    """Carga vocabulario del modelo y construye índices pre-computados."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        raw_vocab = json.load(f)

    token2id = raw_vocab
    id2token = {v: k for k, v in token2id.items()}

    # Pre-indexar por primer carácter decodificado
    tokens_starting_with: dict[str, set[int]] = {}
    for token_text, token_id in token2id.items():
        try:
            # Intentar decodificar el token a texto legible
            decoded = token_text.encode('utf-8').decode('utf-8')
            first_char = decoded[0] if decoded else ""
        except (UnicodeDecodeError, IndexError):
            # Tokens con bytes UTF-8 incompletos — los agrupamos bajo "<byte>"
            first_char = "<byte>"
        tokens_starting_with.setdefault(first_char, set()).add(token_id)

    return Vocab(
        token2id=token2id,
        id2token=id2token,
        tokens_starting_with=tokens_starting_with,
        vocab_size=len(token2id),
    )
```

**¿Qué es `tokens_starting_with`?** Es un índice que agrupa tokens por su primer carácter. En vez de escanear 151K tokens para encontrar los que empiezan con `{`, directamente hacés `tokens_starting_with["{"]` y te da un set de IDs. Es la diferencia entre buscar un libro en una biblioteca desorganizada (escanear todo) vs usar el catálogo (úsqueda instantánea).

**¿Qué son los tokens con bytes UTF-8 incompletos?** Algunos tokens de BPE representan **parciales** de caracteres Unicode. Por ejemplo, `é` en UTF-8 son dos bytes: `0xC3` y `0xA9`. Un token puede representar solo el primer byte — eso es un "byte incompleto". No podemos decodificarlo a un carácter legible, así que lo agrupamos bajo `"<byte>"` en el índice.

## Manejo de errores en cargadores

Nuestros cargadores siguen una regla clara: **fail fast en inputs**. Si el archivo de entrada es inválido, no tiene sentido continuar.

```python
# Errores que los cargadores pueden lanzar:
# FileNotFoundError → el archivo no existe
# json.JSONDecodeError → el JSON está malformado
# ValidationError (Pydantic) → la estructura no cumple el schema
# ValueError → reglas de negocio (nombres duplicados, lista vacía, etc.)
```

**¿Por qué fail fast?** Porque si el archivo de funciones tiene un error, TODO el pipeline va a fallar. No queremos procesar 11 prompts para descubrir en el último que el archivo de entrada estaba mal. Mejor fallar AHORA con un mensaje claro.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **JSON** | JavaScript Object Notation — formato de datos universal (`{"key": "value"}`) |
| **json.load()** | Función de Python que lee un archivo JSON y lo convierte a objetos Python |
| **Vocabulario** | Diccionario del modelo: mapea tokens (texto) a IDs (números) y viceversa |
| **Token** | Unidad mínima de texto que el modelo procesa. Un token puede ser una palabra, sub-palabra, o parte de un carácter |
| **Pre-indexación** | Construir índices auxiliares al inicio para hacer búsquedas rápidas después |
| **UTF-8** | Codificación de caracteres que convierte texto a bytes. Un carácter puede ocupar 1-4 bytes |
| **BPE** | Byte-Pair Encoding — algoritmo de tokenización que divide texto en sub-palabras (ver M5) |
| ****item** | Sintaxis de Python para desempaquetar un diccionario en argumentos de palabra clave |
| **Fail fast** | Estrategia de error: fallar temprano con un mensaje claro, no intentar continuar con datos inválidos |

## Checkpoint — ¿lo entendiste?

1. **¿Qué hace `tokens_starting_with` y por qué es importante?**

<details><summary>Respuesta</summary>Agrupa tokens por su primer carácter. En vez de escanear 151K tokens para encontrar los que empiezan con `{`, hacemos `tokens_starting_with["{"]` y obtenemos el set directamente. Reduce ~150ms por step a ~0.1ms.</details>

2. **¿Qué pasa si el JSON de funciones tiene un nombre duplicado?**

<details><summary>Respuesta</summary>El cargador lanza ValueError y el programa para. No tiene sentido continuar porque el decoder (trie) no maneja duplicados correctamente.</details>

3. **¿Por qué los tokens con bytes incompletos van a `"<byte>"`?**

<details><summary>Respuesta</summary>Porque no se pueden decodificar a un carácter legible. Son parciales de caracteres Unicode. Los agrupamos para no crashear el pre-indexado, y el token_filter los maneja como caso especial.</details>

4. **¿Qué es `json.load()` y qué retorna?**

<details><summary>Respuesta</summary>Lee un archivo JSON y lo convierte a objetos Python: JSON object → dict, array → list, string → str, number → int/float, true/false → bool, null → None.</details>

5. **¿Por qué fail fast en inputs?**

<details><summary>Respuesta</summary>Porque si los datos de entrada son inválidos, TODO el pipeline va a fallar. Es mejor fallar ahora con un mensaje claro que procesar todo para descubrir el error al final.</details>

---

# M4: El prompt

## Qué vas a aprender acá

En este módulo entendés **qué es un prompt** y cómo diseñamos uno que le diga al modelo exactamente qué hacer. Es como redactar las instrucciones de un manual — si las dejás ambiguas, el modelo hace cualquiera.

## ¿Qué es un prompt?

Un **prompt** es el texto que le pasás a un modelo de LLM para que genere una respuesta. Es la "pregunta" o "instrucción".

**Analogía**: el prompt es como el pedido que le hacés a un cocinero. Si le decís "hacé algo rico", te puede hacer cualquier cosa. Si le decís "hacé una milanesa con puré, sincebolla, punto crocante", te va a dar exactamente lo que pediste.

## System prompt vs user prompt

Los prompts suelen tener dos partes:

1. **System prompt**: las instrucciones generales — quién es el asistente, qué debe hacer, qué formato usar
2. **User prompt**: la pregunta específica del usuario

```
[System]
Sos un asistente que llama funciones. Dada una pregunta, outputá UN JSON.

[User]
¿Cuánto es 2 + 3?
```

En nuestro caso, no usamos el chat template del modelo (con tokens especiales `<|system|>`, `<|user|>`, etc.). Usamos un **string plano** que combina ambas partes. Es más portable y testeable.

## Nuestro prompt template

```python
# src/prompt/prompt_builder.py
from src.models.function_definition import FunctionDef

SYSTEM_PROMPT = """You are a function calling assistant. Given the user's query, you must output a JSON object that calls the most appropriate function.

Available functions:

{function_list}

Output ONLY a JSON object with "name" and "parameters" fields. No explanation."""

def build_function_list(functions: list[FunctionDef]) -> str:
    """Arma el listado numerado de funciones disponibles."""
    parts = []
    for i, fn in enumerate(functions, 1):
        params = ", ".join(
            f"{pname} ({pinfo['type']})"
            for pname, pinfo in fn.parameters.items()
        )
        parts.append(f"{i}. {fn.name}: {fn.description}\n   Parameters: {params}")
    return "\n\n".join(parts)

def build_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Arma el prompt completo para un query del usuario."""
    function_list = build_function_list(functions)
    return SYSTEM_PROMPT.format(function_list=function_list) + f"\nUser query: {user_prompt}"
```

## Ejemplo de prompt generado

Si las funciones son las 5 del proyecto y el usuario pregunta "What is 2+3?", el prompt se ve así:

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
User query: What is 2+3?
```

**¿Por qué "Output ONLY a JSON object..."?** Porque el modelo es pequeño (0.6B parámetros) y puede generar texto explicativo antes del JSON. Ese texto rompería nuestro decoder (que espera empezar directamente con `{`). La instrucción "No explanation" reduce la probabilidad de que el modelo genere texto extra.

## ¿Por qué este formato y no otro?

Probamos (o podemos probar) varias variantes:

| Variante | Pros | Contras |
|----------|------|---------|
| **String plano (nuestro)** | Simple, portable, testeable | Modelo puede ignorar la instrucción |
| **Chat template** (`<\|system\|>...`) | El modelo lo reconoce mejor | Requiere tokens especiales del modelo, menos portable |
| **Few-shot** (con examples) | Modelo entiende mejor el formato | Más tokens, más lento, modelo puede copiar los examples |
| **XML format** | Estructura clara | Modelo pequeño puede no entender XML |

Para un modelo de 0.6B, el string plano es la mejor opción: simple, directo, y el modelo Qwen3 lo maneja bien.

## La importancia del formato de output

Nuestro output debe ser **exactamente**:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

No:
- `{"function": "fn_add_numbers", ...}` (key incorrecta)
- `{"name": "fn_add_numbers", "params": {...}}` (key incorrecta)
- `[{"name": "fn_add_numbers", ...}]` (array wrapper — el subject dice object, no array)

El prompt le dice al modelo el formato exacto. El constrained decoder se encarga de que el output **sea** ese formato por construcción.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Prompt** | Texto que le pasás a un LLM para que genere una respuesta |
| **System prompt** | Instrucciones generales del asistente (quién es, qué hace, formato) |
| **User prompt** | La pregunta o instrucción específica del usuario |
| **Template** | Plantilla con placeholders que se rellenan con datos reales |
| **Few-shot** | Técnica de prompt: incluir examples de input/output para que el modelo entienda el patrón |
| **Chat template** | Formato especial con tokens como `<\|system\|>`, `<\|user\|>` que algunos modelos reconocen |
| **String plano** | Texto sin formato especial — solo un string normal |
| **Token** | Unidad mínima de texto que el modelo procesa (ver M5 para detalles) |

## Checkpoint — ¿lo entendiste?

1. **¿Cuál es la diferencia entre system prompt y user prompt?**

<details><summary>Respuesta</summary>El system prompt son las instrucciones generales (quién es el asistente, qué formato usar). El user prompt es la pregunta específica. En nuestro caso, los combinamos en un solo string plano.</details>

2. **¿Por qué usamos string plano en vez de chat template?**

<details><summary>Respuesta</summary>Porque es más portable (funciona con cualquier modelo), más testeable (podemos verificar el prompt con asserts simples), y el modelo Qwen3 entiende bien instrucciones en texto plano.</details>

3. **¿Qué pasaría si el prompt no dice "Output ONLY a JSON object"?**

<details><summary>Respuesta</summary>El modelo podría generar texto explicativo antes del JSON ("La función que necesitas es fn_add_numbers..."). Eso rompería nuestro decoder, que espera empezar directamente con `{`.</details>

4. **¿Por qué el prompt enumera las funciones con números?**

<details><summary>Respuesta</summary>La numeración ayuda al modelo a distinguir las funciones y facilita la lectura. Es un patrón común en prompts de function calling.</details>

5. **¿Qué pasa si el usuario pasa un prompt que no tiene respuesta con las funciones disponibles?**

<details><summary>Respuesta</summary>El modelo igualmente intentará elegir la función "más cercana". El constrained decoder forzará un JSON válido, pero la accuracy puede ser baja. Nuestro validador post-generación detectará si el nombre no existe.</details>

---

# M5: BPE y tokenización

## Qué vas a aprender acá

En este módulo entendés **qué es BPE** (Byte-Pair Encoding) y cómo el modelo convierte texto a tokens. Este es un concepto que el usuario NO conoce — se explica desde cero. Es como aprender a leer el alfabeto del modelo.

## ¿Qué es un token?

Ya sabés que un token es la unidad más pequeña de texto que el modelo procesa. Pero **¿qué es exactamente?**

Un token **NO** es una palabra. **NO** es un carácter. Es algo intermedio definido por el tokenizador del modelo.

**Analogía**: imaginá que el texto es una oración y los tokens son como las sílabas. "计算机" (computadora en chino) puede ser un token. "unbelievable" puede ser tres tokens: "un", "believ", "able". No hay una regla fija — depende de cómo fue entrenado el tokenizador.

Para el modelo Qwen3-0.6B, el vocabulario tiene **~151,643 tokens**. Cada token es un string que mapea a un número (el token ID).

## ¿Qué es BPE?

**BPE = Byte-Pair Encoding**. Es el algoritmo que se usa para crear el vocabulario del modelo. Funciona así:

1. **Empezás con bytes**: cada carácter se convierte a sus bytes en UTF-8. "H" = `[72]`, "é" = `[195, 169]`
2. **Contás pares**: mirás qué par de bytes aparece más seguido en todo el corpus de entrenamiento
3. **Fusionás el par más común**: si "e" y "s" aparecen mucho juntos, los fusionás en un nuevo token "es"
4. **Repetís**: seguís fusionando pares hasta llegar al tamaño deseado del vocabulario (~151K para Qwen)

**Analogía**: es como si tuvieras un alfabeto de letras sueltas y fueras juntando las más frecuentes en "silabas". Primero "th" se vuelve un token, luego "the", luego "Ġthe" (con el espacio). Cada paso reduce la cantidad de tokens necesarios para representar el texto.

## ¿Por qué "Ġthe" es un token?

En el vocabulario de BPE, los espacios al inicio de palabras se representan con `Ġ` (que es el byte `0x20` — espacio — pero mostrado como carácter visible). Entonces:

- `"the"` (sin espacio al inicio) es un token
- `"Ġthe"` (con espacio al inicio) es **otro token** diferente

¿Por qué? Porque en inglés, "the" al inicio de una oración es diferente de "the" después de un espacio. El tokenizador los trata como tokens separados para ser más eficiente.

```python
# Ejemplo de tokens del vocabulario Qwen3:
# "Ġthe" → token con el espacio incluido
# "Ġa" → " a"
# "Ġis" → " is"
# "hello" → "hello" (sin espacio)
# "world" → "world"
# "Ġworld" → " world"
```

## Vocab.json: el diccionario del modelo

El archivo `vocab.json` es un JSON que mapea cada token (string) a su ID (número):

```json
{
  "!": 0,
  "\"": 1,
  "#": 2,
  "$": 3,
  ...
  "Ġthe": 326,
  "Ġa": 264,
  "Ġis": 318,
  ...
  "fn_add_numbers": 148234,
  ...
}
```

Nuestro `vocab_loader.py` lee este archivo y construye:
- `token2id`: `{"Ġthe": 326, ...}` — buscar ID por token
- `id2token`: `{326: "Ġthe", ...}` — buscar token por ID
- `tokens_starting_with`: `{"Ġ": {326, 264, 318, ...}, "f": {148234, ...}, ...}` — pre-indexado por primer carácter

## Tokens multi-carácter: el problema central

Un token de BPE puede representar **múltiples caracteres**:

```python
# Ejemplos de tokens y sus caracteres:
token = "Ġthe"      # 4 caracteres: ' ', 't', 'h', 'e'
token = "Ġfunction"  # 9 caracteres: ' ', 'f', 'u', 'n', 'c', 't', 'i', 'o', 'n'
token = "hello"      # 5 caracteres
token = "Ġ"          # 1 carácter (espacio)
token = "a"          # 1 carácter
```

**¿Por qué importa?** Porque nuestro constrained decoder necesita validar que cada token generado **mantiene el JSON válido**. Si el token es "Ġthe" (4 caracteres), necesitamos procesar los 4 caracteres a través de la state machine, no solo el primero.

**Analogía**: es como si te dieran un paquete de ladrillos (token) en vez de un ladrillo a la vez (carácter). No podés revisar el paquete mirando solo la caja — necesitás abrirlo y revisar cada ladrillo adentro.

## UTF-8 bytes: el caso especial

UTF-8 es la codificación de texto más usada. Cada carácter se convierte a 1-4 bytes:

- ASCII (`a`, `z`, `0`, `9`): 1 byte
- Caracteres acentuados (`é`, `ñ`, `ü`): 2 bytes
- Caracteres CJK (chino, japonés): 3 bytes
- Emojis: 4 bytes

**El problema con BPE byte-level**: el tokenizador puede crear tokens que representan **bytes parciales** de un carácter UTF-8. Por ejemplo, `é` son dos bytes: `0xC3` y `0xA9`. Un token puede representar solo `0xC3` — eso es un "byte incompleto" que no se puede decodificar a un carácter legible.

```python
# Tokens problemáticos:
token = "Ã©"  # bytes 0xC3 0xA9 — parcial de 'é'
# No se puede decodificar limpiamente a texto legible
```

Nuestro `vocab_loader.py` maneja esto: si un token no se puede decodificar a UTF-8 limpio, lo agrupa bajo `"<byte>"` en el índice `tokens_starting_with`.

## ¿Cómo se usa BPE en nuestro proyecto?

1. **Tokenización**: `model.encode(prompt)` convierte el texto a una lista de token IDs usando BPE
2. **Generación**: en cada step, el modelo genera logits para ~151K tokens. Elegimos uno.
3. **Detokenización**: `model.decode(token_id)` convierte el token ID de vuelta a texto
4. **Simulación**: nuestro decoder toma el texto del token y lo procesa char-by-char a través de la state machine

```python
# Flujo completo:
prompt = "What is 2+3?"
input_ids = model.encode(prompt)  # BPE tokeniza: [5765, 318, 220, 17, 10, 18, 30]
#                                        (tokens aproximados)

# En cada step del loop:
logits = model.get_logits_from_input_ids(input_ids)  # ~151K floats
best_id = argmax(allowed)  # Elegimos el mejor token permitido
token_text = vocab.id2token[best_id]  # "Ġthe" por ejemplo
# Procesamos "Ġthe" char por char a través de la state machine
```

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **BPE** | Byte-Pair Encoding — algoritmo que crea el vocabulario fusionando pares de bytes frecuentes |
| **Token** | Unidad mínima de texto: puede ser un carácter, una palabra, o una sub-palabra |
| **Token ID** | Número que identifica un token en el vocabulario (ej: 326 para "Ġthe") |
| **Vocabulario** | Diccionario completo de tokens del modelo (~151K para Qwen3) |
| **Tokenizador** | Algoritmo que convierte texto a tokens (y viceversa) |
| **Ġ** | Representación visible del byte de espacio (0x20) en tokens BPE |
| **Byte-level BPE** | BPE que opera sobre bytes en vez de caracteres — maneja cualquier Unicode |
| **UTF-8** | Codificación de texto: 1-4 bytes por carácter |
| **Byte incompleto** | Token que representa bytes parciales de un carácter Unicode — no se puede decodificar limpiamente |
| **Sub-palabra** | Token que es parte de una palabra: "unbeliev" en "unbelievable" |

## Checkpoint — ¿lo entendiste?

1. **¿Qué es BPE en una frase?**

<details><summary>Respuesta</summary>Un algoritmo que crea tokens fusionando los pares de bytes más frecuentes en el corpus de entrenamiento, hasta llegar a ~151K tokens.</details>

2. **¿Por qué "Ġthe" y "the" son tokens diferentes?**

<details><summary>Respuesta</summary>Porque el tokenizador BPE trata el espacio como un carácter más. "Ġthe" es " the" (con espacio), que aparece frecuentemente al inicio de palabras. "the" sin espacio es otro patrón con diferente frecuencia.</details>

3. **¿Qué problema causan los tokens multi-carácter en constrained decoding?**

<details><summary>Respuesta</summary>Un token puede contener múltiples caracteres, y solo el primero determina si es candidato. Pero el token completo puede romper el JSON si los caracteres restantes son inválidos. Por eso procesamos TODOS los caracteres del token a través de la state machine.</details>

4. **¿Qué es un "byte incompleto" y cómo lo manejamos?</details>

<details><summary>Respuesta</summary>Es un token que representa bytes parciales de un carácter Unicode (ej: solo el primer byte de `é`). No se puede decodificar a texto legible. Los agrupamos bajo `"<byte>"` en el índice y los manejamos como caso especial en el token filter.</details>

5. **¿Por qué el vocabulario tiene ~151K tokens en vez de, digamos, 10K?**

<details><summary>Respuesta</summary>Porque más tokens = menos sub-palabras = texto representado con menos tokens = más contexto cabe en la ventana del modelo. 151K es un equilibrio entre cobertura y eficiencia para Qwen3.</details>

---

# M6: Máquina de estados

## Qué vas a aprender acá

En este módulo construís la **state machine** del decoder JSON — la estructura que sabe en qué parte del JSON estamos y qué caracteres son válidos接下来. Es como un GPS que te dice "estás en la calle X, solo podés girar a la derecha o seguir derecho".

## ¿Qué es una máquina de estados?

Una **máquina de estados** (o autómata) es un modelo computacional que tiene:
- **Estados**: situaciones en las que puede estar
- **Transiciones**: de un estado a otro, basadas en un input
- **Un estado actual**: ¿dónde estoy ahora?

**Analogía**: imaginá un semáforo. Tiene 3 estados: ROJO, AMARILLO, VERDE. Las transiciones son:
- ROJO → VERDE (después de 30s)
- VERDE → AMARILLO (después de 25s)
- AMARILLO → ROJO (después de 5s)

El semáforo nunca va de ROJO a AMARILLO directamente. Las reglas de transición definen qué caminos son posibles.

## La state machine del decoder JSON

Nuestro decoder JSON tiene **15 estados**. Cada uno representa una posición dentro del JSON que estamos generando:

```
ROOT                  → Empezamos, esperando '{'
OBJECT_OPEN           → '{' leído, dentro del objeto raíz
IN_OBJECT             → Dentro del objeto, esperando key o '}'
KEY_START             → '"' leído, empezando a leer nombre de key
IN_KEY                → Leyendo caracteres del nombre de key
KEY_END               → '"' leído al final del nombre de key
COLON                 → ':' leído, esperando value
VALUE_START           → Primer carácter del value
IN_STRING_VALUE       → Dentro de un value string
IN_NUMBER_VALUE       → Dentro de un number
IN_BOOL_VALUE         → Dentro de un boolean (true/false)
IN_NULL_VALUE         → Dentro de null
ESCAPE_IN_STRING      → '\' leído dentro de string
VALUE_END             → Value terminado, esperando ',' o '}'
COMPLETE              → '}' de cierre leído — ¡LISTO!
```

## Diagrama de transiciones

```
ROOT ──────────────{──────────────→ OBJECT_OPEN

OBJECT_OPEN ───────"──────────────→ KEY_START

IN_OBJECT ─────────"──────────────→ KEY_START
IN_OBJECT ─────────}──────────────→ COMPLETE

KEY_START ─────────char──────────→ IN_KEY
IN_KEY ────────────"──────────────→ KEY_END
IN_KEY ────────────char──────────→ IN_KEY

KEY_END ───────────:──────────────→ COLON

COLON ─────────────"──────────────→ IN_STRING_VALUE
COLON ─────────────- or digit────→ IN_NUMBER_VALUE
COLON ─────────────t──────────────→ IN_BOOL_VALUE
COLON ─────────────f──────────────→ IN_BOOL_VALUE
COLON ─────────────n──────────────→ IN_NULL_VALUE
COLON ─────────────{──────────────→ PARAMS_OBJECT

IN_STRING_VALUE ───"──────────────→ VALUE_END
IN_STRING_VALUE ───\──────────────→ ESCAPE_IN_STRING
IN_STRING_VALUE ───char──────────→ IN_STRING_VALUE

IN_NUMBER_VALUE ───digit or .────→ IN_NUMBER_VALUE
IN_NUMBER_VALUE ───e or E────────→ IN_NUMBER_VALUE
IN_NUMBER_VALUE ───, or } or ws──→ VALUE_END

VALUE_END ─────────,──────────────→ IN_OBJECT
VALUE_END ─────────}──────────────→ COMPLETE
```

## Implementación en Python

Usamos un `enum` para los estados y un `dataclass` para el estado actual:

```python
# src/decoder/state.py
from enum import Enum, auto
from dataclasses import dataclass, field
import copy

class DecoderPhase(Enum):
    """Estados de la state machine del decoder JSON."""
    ROOT = auto()
    OBJECT_OPEN = auto()
    IN_OBJECT = auto()
    KEY_START = auto()
    IN_KEY = auto()
    KEY_END = auto()
    COLON = auto()
    VALUE_START = auto()
    IN_STRING_VALUE = auto()
    IN_NUMBER_VALUE = auto()
    IN_BOOL_VALUE = auto()
    IN_NULL_VALUE = auto()
    ESCAPE_IN_STRING = auto()
    VALUE_END = auto()
    COMPLETE = auto()
    PARAMS_OBJECT = auto()

@dataclass(slots=True)
class DecoderState:
    """Estado actual del decoder JSON."""
    phase: DecoderPhase = DecoderPhase.ROOT
    current_key: str = ""           # Key que estamos leyendo
    keys_enclosed: set[str] = field(default_factory=set)  # Keys ya escritas
    depth: int = 0                  # Nivel de anidamiento
    number_has_digit: bool = False  # Para validar numbers
    number_has_dot: bool = False    # Para validar decimals
    bool_buffer: str = ""           # Buffer para "true"/"false"/"null"

    def _advance_char(self, char: str) -> bool:
        """Procesa un carácter a través de la state machine.
        Retorna True si la transición es válida, False si no."""
        # ... implementación de transiciones ...
        pass

    def simulate(self, token_text: str) -> tuple[bool, "DecoderState"]:
        """Simula procesar un token completo.
        Retorna (es_válido, nuevo_estado)."""
        state = copy.copy(self)  # Copia superficial (~50ns, no deepcopy ~5μs)
        for char in token_text:
            if not state._advance_char(char):
                return False, self  # Si falla, retorno estado original
        return True, state

    def update_from_text(self, token_text: str) -> None:
        """Avanza el estado real con el token generado."""
        for char in token_text:
            self._advance_char(char)

    def expected_first_chars(self) -> set[str]:
        """Retorna los caracteres válidos para el próximo token."""
        if self.phase == DecoderPhase.ROOT:
            return {"{"}
        if self.phase == DecoderPhase.KEY_START:
            return {'"'}
        # ... más casos ...
        return set()
```

**¿Por qué `@dataclass(slots=True)`?** `__slots__` le dice a Python "este objeto solo tiene estos atributos específicos, no necesitás un `__dict__`". Esto ahorra memoria y hace que la creación de objetos sea más rápida (~30% más rápido en benchmarks). En el inner loop, cada milisegundo cuenta.

**¿Por qué `copy.copy()` y no `copy.deepcopy()`?** Porque `simulate` necesita una copia del estado para explorar sin afectar el estado real. `copy.copy()` es una copia superficial (~50nanosegundos). `copy.deepcopy()` es profunda (~5microsegundos — 100x más lento). Funciona porque los campos son tipos inmutables (str, int, bool) o estructuras simples.

## Ejemplo: procesando un JSON paso a paso

Veamos cómo la state machine procesa `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`:

```
Char: {  → ROOT → OBJECT_OPEN     ✓
Char: "  → OBJECT_OPEN → KEY_START ✓
Char: n  → KEY_START → IN_KEY      ✓  (current_key = "n")
Char: a  → IN_KEY → IN_KEY         ✓  (current_key = "na")
Char: m  → IN_KEY → IN_KEY         ✓  (current_key = "nam")
Char: e  → IN_KEY → IN_KEY         ✓  (current_key = "name")
Char: "  → IN_KEY → KEY_END        ✓  (keys_enclosed = {"name"})
Char: :  → KEY_END → COLON         ✓
Char: "  → COLON → IN_STRING_VALUE ✓
Char: f  → IN_STRING_VALUE → IN_STRING_VALUE ✓
... (más caracteres del nombre)
Char: "  → IN_STRING_VALUE → VALUE_END ✓
Char: ,  → VALUE_END → IN_OBJECT   ✓
Char: "  → IN_OBJECT → KEY_START   ✓
Char: p  → KEY_START → IN_KEY      ✓  (current_key = "p")
...
```

Cada carácter avanza la state machine un paso. Si algún carácter no es válido en el estado actual (por ejemplo, un número en IN_KEY), `_advance_char` retorna False.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Máquina de estados** | Modelo computacional con estados, transiciones, y un estado actual |
| **State machine** | Término en inglés para máquina de estados |
| **Estado** | Situación en la que se encuentra la machine (ej: "leyendo key", "dentro de string") |
| **Transición** | Cambio de un estado a otro, disparado por un input (carácter) |
| **Enum** | Tipo de dato que solo puede tener valores predefinidos (como DecoderPhase.ROOT) |
| **Dataclass** | Decorador de Python para crear clases de datos con `__init__`, `__repr__`, etc. automáticos |
| **`__slots__`** | Lista de atributos permitidos en un dataclass — ahorra memoria y tiempo |
| **copy.copy()** | Copia superficial de un objeto (~50ns) — rápido pero comparte objetos internos |
| **simulate()** | Método que explora qué pasaría si procesáramos un token sin afectar el estado real |

## Checkpoint — ¿lo entendiste?

1. **¿Cuáles son los 3 estados más importantes del decoder?**

<details><summary>Respuesta</summary>ROOT (empezamos, esperando '{'), IN_OBJECT (dentro del objeto, esperando key o '}'), COMPLETE (JSON terminado). Son los estados que definen el inicio, el cuerpo, y el fin del JSON.</details>

2. **¿Por qué `simulate()` hace `copy.copy()` en vez de modificar el estado directamente?**

<details><summary>Respuesta</summary>Porque `simulate` es una exploración — necesita ver qué pasaría sin afectar el estado real. Si modificara el estado directamente, después de simular 151K tokens el estado estaría corrompido.</details>

3. **¿Qué pasa si `_advance_char` recibe un carácter inválido?**

<details><summary>Respuesta</summary>Retorna False, y `simulate()` retorna `(False, estado_original)`. El token que causó el problema no es válido, así que se descarta del set de allowed tokens.</details>

4. **¿Qué es `DecoderPhase.COMPLETE` y cuándo se alcanza?**

<details><summary>Respuesta</summary>Es el estado final — se alcanza cuando se lee el '}' de cierre del objeto raíz. Cuando el decoder llega a COMPLETE, para de generar tokens.</details>

5. **¿Por qué `__slots__` hace los objetos más rápidos?**

<details><summary>Respuesta</summary>Porque Python no necesita crear un `__dict__` (diccionario dinámico) para cada objeto. Los atributos se almacenan en un array fijo, que es más rápido de acceder y consume menos memoria.</details>

---

# M7: El trie

## Qué vas a aprender acá

En este módulo conocés el **trie** (prefix tree), la estructura de datos que usamos para restringir la selección de nombres de función. Es como un árbol de decisiones que te dice "si escribiste 'fn_a', el siguiente carácter solo puede ser 'd' (para 'fn_add_numbers')" o "si escribiste 'fn_g', solo puede ser 'r' (para 'fn_greet')".

## ¿Qué es un trie?

Un **trie** (pronunciado "try") es un árbol donde cada nodo representa un carácter. Para encontrar si una palabra existe en el trie, recorrés el árbol carácter por carácter.

**Analogía**: imaginá unorganizador de archivos con carpetas anidadas:

```
carpeta_raíz/
├── f/
│   └── n/
│       └── _/
│           ├── a/
│           │   └── d/
│           │       └── d/
│           │           └── _/
│           │               └── n/
│           │                   └── u/
│           │                       └── m/
│           │                           └── b/
│           │                               └── e/
│           │                                   └── r/
│           │                                       └── s/ ← ¡fn_add_numbers!
│           └── g/
│               └── r/
│                   └── e/
│                       └── e/
│                           └── t/ ← ¡fn_greet!
└── (más funciones...)
```

Para buscar "fn_add_numbers", entrás a `f/`, luego `n/`, luego `_/`, luego `a/`, y así sucesivamente. Si en algún punto la ruta no existe, la palabra no está en el trie.

## Implementación

```python
# src/decoder/trie.py
from dataclasses import dataclass, field

@dataclass(slots=True)
class TrieNode:
    """Nodo del trie."""
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    function_name: str | None = None  # Non-None solo en nodos terminales
    is_end: bool = False

def build_trie(function_names: list[str]) -> TrieNode:
    """Construye un trie desde una lista de nombres de función."""
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

**¿Por qué `function_name` y `is_end`?** `is_end` indica "este nodo es el final de un nombre completo". `function_name` guarda el nombre completo para referencia. Un nodo puede tener `is_end=True` y `children` vacíos (es un nodo hoja que es un nombre completo) o `is_end=True` y `children` (es un nombre completo que también es prefix de otro nombre — como "fn_" siendo prefix de todas las funciones).

## Uso en constrained decoding

Cuando el decoder está en `IN_STRING_VALUE` para el campo "name", necesita saber qué caracteres son válidos:

```python
def valid_next_chars(trie: TrieNode, prefix: str) -> set[str]:
    """Retorna los caracteres válidos para continuar el prefix."""
    node = trie
    for char in prefix:
        if char not in node.children:
            return set()  # Prefix no existe en el trie
        node = node.children[char]

    # Los chars válidos son las keys de children
    valid = set(node.children.keys())

    # Si el nodo es terminal, '"' también es válido (cerrar el nombre)
    if node.is_end:
        valid.add('"')

    return valid

def is_complete_name(trie: TrieNode, prefix: str) -> bool:
    """¿El prefix es exactamente un nombre de función válido?"""
    node = trie
    for char in prefix:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end
```

### Ejemplo concreto

Dadas las funciones `fn_add_numbers`, `fn_greet`, `fn_reverse_string`:

```
prefix = "fn_a" → valid_next_chars = {"d"}  (solo "fn_add_numbers" continúa)
prefix = "fn_g" → valid_next_chars = {"r"}  (solo "fn_greet" continúa)
prefix = "fn_" → valid_next_chars = {"a", "g", "r", "g", "s"}  (todas las funciones)
prefix = "fn_add_numbers" → valid_next_chars = {'"'}  (solo cerrar, es completo)
```

## ¿Por qué no hardcodeamos los nombres?

**Zero hardcoding**: el trie se construye dinámicamente desde el JSON de entrada. Si mañana se agregan funciones, el trie se adapta automáticamente. No hay que cambiar ni una línea de código.

```python
# El trie se construye una vez al inicio:
trie = build_trie([fn.name for fn in functions])

# Y se usa en cada step del decoder:
valid_chars = valid_next_chars(trie, current_prefix)
```

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Trie** | Árbol de prefixos donde cada nodo representa un carácter. Se usa para buscar y validar strings |
| **Prefix tree** | Otra nombre para trie — "árbol de prefijos" |
| **Nodo** | Cada punto del trie. Tiene children (caracteres siguientes) y optionally un nombre completo |
| **Prefix** | String parcial que representa el camino recorrido desde la raíz del trie |
| **Hardcoding** | Poner valores fijos directamente en el código (malo porque no es flexible) |
| **Zero hardcoding** | No poner ningún valor fijo — todo viene de datos de entrada |

## Checkpoint — ¿lo entendiste?

1. **¿Qué es un trie en una frase?**

<details><summary>Respuesta</summary>Un árbol donde cada nodo es un carácter, usado para verificar si un string (o su prefix) coincide con nombres conocidos.</details>

2. **¿Qué retorna `valid_next_chars(trie, "fn_a")` con las 5 funciones del proyecto?**

<details><summary>Respuesta</summary>`{"d"}` — porque solo "fn_add_numbers" continúa después de "fn_a". Las demás funciones empiezan con "fn_g" o "fn_r" o "fn_s".</details>

3. **¿Por qué `is_end` es importante?**

<details><summary>Respuesta</summary>Porque indica que el nodo actual es un nombre completo. Cuando `is_end=True`, además de los children, `"` también es válido (para cerrar el string del nombre).</details>

4. **¿Qué pasa si el trie no tiene un prefix?**

<details><summary>Respuesta</summary>Si el prefix no existe en el trie (ej: "xyz"), `valid_next_chars` retorna un set vacío. Eso significa que el modelo está generando un nombre inválido — el decoder entra en repair mode o skip.</details>

5. **¿Por qué el trie se construye una vez al inicio?**

<details><summary>Respuesta</summary>Porque los nombres de función no cambian durante la ejecución. Construirlo una vez y reusarlo es más eficiente que reconstruirlo en cada step del loop.</details>

---

# M8: El filtro de tokens

## Qué vas a aprender acá

En este módulo está **el corazón del constrained decoding**: el filtro que decide qué tokens puede elegir el modelo en cada step. Es como un guardia de seguridad en la puerta — solo deja pasar a los que cumplen las reglas.

## El problema fundamental

El modelo genera **logits** — una lista de ~151K números, uno por cada token del vocabulario. Cada logit indica cuánto "prefiere" el modelo ese token. Para elegir el siguiente token, normalmente harías `argmax` (elegir el más alto).

**Pero**: el modelo puede preferir un token que rompa el JSON. Si el modelo dice "el siguiente token es 'ñ'", pero estamos esperando un '{', ese token es inválido.

**Solución**: filtrar los tokens ANTES de elegir. Solo dejamos pasar los que mantienen el JSON válido.

## Las 3 fases del filtrado

El filtro tiene 3 fases, como un proceso de selección de trabajadores:

### Fase 1: Pre-filtro por categoría de primer carácter

El primer carácter del token nos dice mucho. Si estamos en ROOT, solo necesitamos tokens que empiecen con `{`. Si estamos leyendo una key, solo tokens que empiecen con un carácter válido para key names.

```python
# En token_filter.py
current_char_categories = state.expected_first_chars()  # set de chars válidos

candidate_ids: set[int] = set()
for char in current_char_categories:
    candidate_ids.update(vocab.tokens_starting_with.get(char, set()))
```

**¿Por qué usar `tokens_starting_with`?** Porque es un índice pre-computado (ver M3). Sin él, tendríamos que escanear 151K tokens para encontrar los que empiezan con el carácter correcto. Con él, es una búsqueda O(1) en un dict.

### Fase 2: Decodificación parcial + validación char-by-char

Para cada token candidato, lo procesamos a través de la state machine para ver si es válido:

```python
# Para cada candidate_id:
for token_id in candidate_ids:
    token_text = vocab.id2token[token_id]
    valid, new_state = state.simulate(token_text)

    if valid:
        # El token mantiene el JSON válido
        allowed_ids.add(token_id)
```

**¿Por qué `simulate()` y no solo mirar el primer carácter?** Porque un token puede tener múltiples caracteres. El primero puede ser válido, pero el segundo puede romper todo. Por ejemplo:

```
Token: "a}" → primer char 'a' es válido (key name), pero '}' cierra el objeto
                prematuramente sin haber escrito el value. ¡Inválido!
```

`simulate()` procesa TODOS los caracteres del token a través de la state machine y retorna `(False, ...)` si alguno es inválido.

### Fase 3: Post-filtro de schema

Después de verificar que el token mantiene el JSON válido, verificamos que cumpla las reglas del schema:

```python
if schema.allows(token_text, new_state):
    allowed_ids.add(token_id)
```

El schema validator chequea:
- Si estamos en VALUE para "name": solo nombres de función conocidos (trie)
- Si estamos en KEY de parameters: solo keys que existen en el schema de la función seleccionada
- Si estamos en VALUE de un parameter: tipo correcto (string, number, etc.)
- Si estamos cerrando parameters con '}': todos los required keys ya presentes

## El problema de los tokens multi-char (revisited)

Recordá que un token BPE puede tener múltiples caracteres. El filtro de la Fase 1 solo mira el primer carácter. Pero el token completo puede ser inválido.

**Ejemplo**:

```
Estado actual: KEY_START (empezando a leer una key)
Token candidato: "ab" (primer char 'a' es válido para key name)
Simulación: 'a' → IN_KEY ✓, 'b' → IN_KEY ✓ → VÁLIDO ✓

Token candidato: "}" (primer char '}' NO es válido para KEY_START)
Pre-filtro: descartado en Fase 1 (no está en tokens_starting_with["}"])
```

Pero hay un caso más sutil:

```
Estado actual: IN_KEY (leyendo key name, ya tenemos "na")
Token candidato: "me" (primer char 'm' es válido)
Simulación: 'm' → IN_KEY ✓, 'e' → IN_KEY ✓ → VÁLIDO ✓
Token candidato: "me\" (primer char 'm' es válido)
Simulación: 'm' → IN_KEY ✓, 'e' → IN_KEY ✓, '"' → KEY_END ✓ → VÁLIDO ✓

Pero: "me\"" cierra la key. Si "name" no es una key válida en el schema,
       el post-filtro lo descarta.
```

## Argmax: elegir el mejor token

Una vez que tenemos el set de allowed tokens, elegimos el que tiene el logit más alto:

```python
# argmax sobre allowed tokens
best_id = max(allowed, key=lambda tid: logits[tid])
```

**¿Qué es argmax?** Es una función que retorna el índice del valor más alto. `argmax([1, 5, 3, 2])` retorna `1` (porque 5 es el mayor).

**¿Por qué no usamos softmax?** Softmax convierte logits a probabilidades que suman 1. Pero nosotros solo necesitamos el MÁS ALTO, no las probabilidades. Argmax es más rápido y produce el mismo resultado para nuestra elección.

**¿Por qué elegir el más alto y no un azar?** Porque queremos que el modelo elija el token que MÁS prefiere dentro de los permitidos. Eso maximiza la calidad del output.

## El filtro completo

```python
def compute_allowed_ids(
    state: DecoderState,
    schema: SchemaContext,
    vocab: Vocab,
    trie: TrieNode,
    logits: list[float],
) -> set[int]:
    """Computa el set de tokens permitidos para el próximo step."""
    allowed_ids: set[int] = set()

    # Fase 1: Pre-filtro por primer carácter
    expected_chars = state.expected_first_chars()
    candidate_ids: set[int] = set()
    for char in expected_chars:
        candidate_ids.update(vocab.tokens_starting_with.get(char, set()))

    # Fase 2: Validación char-by-char
    for token_id in candidate_ids:
        token_text = vocab.id2token[token_id]

        # Skip tokens con bytes UTF-8 incompletos
        if not _is_clean_utf8(token_text):
            continue

        valid, new_state = state.simulate(token_text)
        if not valid:
            continue

        # Fase 3: Schema constraints
        if schema.allows_token(token_text, new_state, trie):
            allowed_ids.add(token_id)

    return allowed_ids
```

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Logits** | Lista de ~151K números que indican la preferencia del modelo por cada token |
| **argmax** | Función que retorna el índice del valor más alto en una lista |
| **Softmax** | Función que convierte logits a probabilidades (no la usamos — argmax es suficiente) |
| **Masking** | Técnica de poner `-inf` a tokens no permitidos para que argmax los ignore |
| **Pre-filtro** | Primera pasada: filtrar tokens por categoría de primer carácter |
| **Post-filtro** | Última pasada: verificar schema constraints (nombres, tipos, required keys) |
| **Allowed set** | Set de token IDs que son válidos para el próximo step |
| **Candidate set** | Set amplio de tokens que pasan el pre-filtro (antes de validación char-by-char) |
| **UTF-8 limpio** | Token que se puede decodificar a texto sin bytes parciales |

## Checkpoint — ¿lo entendiste?

1. **¿Cuáles son las 3 fases del filtro de tokens?**

<details><summary>Respuesta</summary>Fase 1: Pre-filtro por primer carácter (usa tokens_starting_with). Fase 2: Validación char-by-char a través de la state machine (simulate). Fase 3: Schema constraints (trie, tipos, required keys).</details>

2. **¿Por qué necesitamos simulate() en vez de solo mirar el primer carácter?**

<details><summary>Respuesta</summary>Porque un token puede tener múltiples caracteres. El primero puede ser válido pero el segundo puede romper el JSON. simulate() procesa TODOS los caracteres.</details>

3. **¿Qué es argmax y por qué no usamos softmax?**

<details><summary>Respuesta</summary>argmax retorna el índice del valor más alto. No usamos softmax porque solo necesitamos elegir el mejor token, no calcular probabilidades. argmax es más rápido y produce el mismo resultado.</details>

4. **¿Qué pasa si el allowed set está vacío?**

<details><summary>Respuesta</summary>Significa que no hay tokens válidos para el próximo step. El decoder intenta "repair" (cerrar estructuras abiertas) o skip el prompt si no puede reparar.</details>

5. **¿Por qué skipeamos tokens con bytes UTF-8 incompletos?**

<details><summary>Respuesta</summary>Porque no se pueden decodificar a texto legible y causarían problemas al procesar char-by-char a través de la state machine.</details>

---

# M9: El bucle de generación

## Qué vas a aprender acá

En este módulo armás el **loop completo** que conecta el modelo con el decoder. Es como la cinta transportadora de una fábrica — cada step procesa un token, y el resultado alimenta el siguiente step.

## El loop paso a paso

```python
# src/decoder/constrained_generator.py
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder.state import DecoderState, DecoderPhase
from src.decoder.schema_validator import SchemaContext
from src.decoder.token_filter import compute_allowed_ids
from src.loader.vocab_loader import Vocab
from src.decoder.trie import TrieNode

MAX_TOKENS = 200  # Safety net — output esperado es ~30-60 tokens

def generate(
    model: Small_LLM_Model,
    prompt: str,
    vocab: Vocab,
    functions: list[FunctionDef],
    trie: TrieNode,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, bool]:
    """Genera output JSON constrained para un prompt.

    Retorna: (texto_generado, éxito)
    """
    # 1. Tokenizar el prompt
    input_ids = model.encode(prompt).tolist()

    # 2. Estado inicial
    state = DecoderState()
    schema = SchemaContext(functions)

    # 3. Loop de generación
    for step in range(max_tokens):
        # 3a. Obtener logits del modelo
        logits = model.get_logits_from_input_ids(input_ids)

        # 3b. Filtrar tokens permitidos
        allowed = compute_allowed_ids(state, schema, vocab, trie, logits)

        if not allowed:
            # Repair o break
            break

        # 3c. Elegir el mejor token permitido
        best_id = max(allowed, key=lambda tid: logits[tid])

        # 3d. Agregar a la secuencia
        input_ids.append(best_id)

        # 3e. Actualizar estado
        token_text = vocab.id2token[best_id]
        state.update_from_text(token_text)

        # 3f. Actualizar schema context
        schema.update(state)

        # 3g. Verificar si terminamos
        if state.phase == DecoderPhase.COMPLETE:
            break

    # 4. Extraer solo los tokens generados (no el prompt)
    prompt_length = len(model.encode(prompt).tolist())
    generated_ids = input_ids[prompt_length:]
    generated = model.decode(generated_ids)

    return generated, state.phase == DecoderPhase.COMPLETE
```

## Flujo visual de un step

```
input_ids = [5765, 318, 220, 17, 10, 18, 30, ...]  ← prompt tokenizado

         ┌──────────────────┐
         │ model.get_logits  │ ← El modelo "piensa" y produce 151K logits
         │ from_input_ids    │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ logits = [0.1,    │ ← Un número por cada token del vocabulario
         │  -0.3, 2.5, ...] │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ compute_allowed   │ ← Filtramos: solo tokens que mantienen JSON válido
         │ _ids(...)         │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ allowed = {1, 5,  │ ← Set de IDs permitidos (ej: tokens que empiezan con '{')
         │  42, 1234}        │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ best_id = argmax  │ ← Elegimos el token con logit más alto entre los permitidos
         │ (logits[allowed]) │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ input_ids.append  │ ← Agregamos el token a la secuencia
         │ (best_id)         │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ state.update      │ ← Actualizamos la state machine con el texto del token
         │ (token_text)      │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ state == COMPLETE?│ ← ¿El JSON está terminado?
         └────────┬─────────┘
              NO ↙    ↘ SÍ
         Volver al      PARAR
         paso 3a
```

## ¿Por qué no usamos KV-cache?

El **KV-cache** es una optimización que evita recalcular los logits de tokens anteriores. Sin KV-cache, cada step recalcula logits para TODA la secuencia (prompt + tokens generados).

**¿Por qué no lo usamos?** Porque el SDK no lo expone. `get_logits_from_input_ids` recibe la secuencia completa y retorna logits para el último token. No hay forma de pasar un cache.

**¿Cuánto nos cuesta?** Cada step toma ~150-200ms en CPU. Para 50 tokens × 11 prompts = ~110s. El límite es 5 min (300s), así que tenemos ~190s de margen para I/O, validación, etc.

## Terminación: ¿cuándo paramos?

Hay **dos** condiciones de terminación:

1. **State == COMPLETE**: el '}' de cierre del objeto raíz fue leído. El JSON está completo.
2. **Safety net**: `max_tokens = 200`. Si llegamos a 200 tokens sin completar, paramos (el output probablemente está roto).

**¿Por qué NO usamos EOS (End of Sequence)?** Porque el subject de 42 advierte explícitamente: "modelos pequeños como Qwen3-0.6B pueden emitir EOS prematuramente". El modelo puede pensar que terminó cuando en realidad falta una key del JSON. Usar solo constrained termination (COMPLETE) es más seguro.

```python
# En el loop:
if state.phase == DecoderPhase.COMPLETE:
    break  # JSON completo — paramos

if step >= max_tokens:
    break  # Safety net — algo salió mal
```

## Manejo de tokens de thinking

Qwen3 puede emitir `<|begin_of_thought|>...<|end_of_thought|>` antes de la respuesta. Estos tokens no son parte del JSON y deben ser skipeados.

```python
# Durante la generación, si el decoder encuentra tokens de thinking:
# El decoder espera '{' como primer token del JSON.
# Si encuentra '<' primero, entra en un estado especial de "skip thinking"
# hasta encontrar '{'.
```

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **KV-cache** | Optimización que almacena keys/values calculados para evitar recalcularlos (no lo usamos) |
| **EOS** | End of Sequence — token que indica "el modelo terminó de generar" (no lo usamos por riesgo de EOS prematuro) |
| **Safety net** | Límite máximo de tokens para evitar loops infinitos |
| **Tokenizar** | Convertir texto a una lista de token IDs |
| **Detokenizar** | Convertir token IDs de vuelta a texto |
| **Step** | Una iteración del loop de generación — un token generado |
| **Complete** | Estado donde el JSON está terminado ('}' de cierre leído) |
| **Thinking tokens** | Tokens que el modelo genera antes de la respuesta real (ej: `<\|begin_of_thought\|>`) |

## Checkpoint — ¿lo entendiste?

1. **¿Cuáles son los 7 pasos de cada iteración del loop?**

<details><summary>Respuesta</summary>1) get_logits, 2) compute_allowed_ids, 3) check empty set, 4) argmax, 5) append, 6) update state, 7) check termination.</details>

2. **¿Por qué NO usamos KV-cache?**

<details><summary>Respuesta</summary>Porque el SDK no lo expone. `get_logits_from_input_ids` recibe la secuencia completa y no hay forma de pasar un cache previo.</details>

3. **¿Por qué NO usamos EOS para terminar?**

<details><summary>Respuesta</summary>Porque el subject advierte que modelos pequeños emiten EOS prematuramente. Podría pasar que el modelo piense que terminó cuando falta una key del JSON. Usar solo COMPLETE es más seguro.</details>

4. **¿Qué es `max_tokens = 200` y por qué es un safety net?**

<details><summary>Respuesta</summary>Es el límite máximo de tokens generados. El output esperado es ~30-60 tokens. Si llegamos a 200, algo salió mal (loop infinito, JSON incompleto). Paramos para no colgar.</details>

5. **¿Qué pasa con los tokens de thinking de Qwen3?**

<details><summary>Respuesta</summary>El decoder los skipea. Espera '{' como primer token del JSON. Si encuentra tokens de thinking primero, los ignora hasta encontrar '{'.</details>

---

# M10: Validación y pipeline

## Qué vas a aprender acá

En este módulo armás la **validación post-generación** y el **pipeline completo** que orquesta todo. Es como la inspección de calidad final antes de que el producto salga de la fábrica.

## Validación post-generación

Aunque nuestro constrained decoder **garantiza** JSON válido por construcción, tenemos un validador como red de seguridad:

```python
# src/validator/output_validator.py
import json
from pydantic import ValidationError
from src.models.output import FunctionCall
from src.models.function_definition import FunctionDef

def validate_output(
    raw_output: str,
    functions: list[FunctionDef],
) -> FunctionCall | str:
    """Valida el output generado contra las definiciones de funciones.

    Retorna FunctionCall en éxito, string de error en fallo.
    """
    # 1. Parsear JSON
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        return f"JSON inválido: {e}"

    # 2. Validar contra Pydantic model
    try:
        call = FunctionCall(**parsed)
    except ValidationError as e:
        return f"Error de validación de schema: {e}"

    # 3. Verificar que el nombre de función existe
    valid_names = {fn.name for fn in functions}
    if call.name not in valid_names:
        return f"Función desconocida: {call.name}"

    # 4. Verificar tipos de parámetros
    fn_def = next(f for f in functions if f.name == call.name)
    for pname, pvalue in call.parameters.items():
        if pname not in fn_def.parameters:
            return f"Parámetro desconocido: {pname}"
        expected = fn_def.parameters[pname]["type"]
        if not _type_matches(pvalue, expected):
            return f"Tipo incorrecto para {pname}: se esperaba {expected}, se obtuvo {type(pvalue).__name__}"

    # 5. Verificar que no hay parámetros extra
    extra = set(call.parameters.keys()) - set(fn_def.parameters.keys())
    if extra:
        return f"Parámetros inesperados: {extra}"

    return call

def _type_matches(value: object, expected_type: str) -> bool:
    """Verifica si un valor coincide con el tipo JSON esperado."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float))
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True  # null o tipos desconocidos pasan
```

## ¿Por qué dos niveles de validación?

El decoder garantiza **syntactic validity** (JSON parseable, keys conocidos). El validador garantiza **semantic validity** (nombre existe, tipos correctos, no hay extras).

**Analogía**: el decoder es como un molde que asegura que el ladrillo tenga la forma correcta. El validador es como un inspector que verifica que el material sea del tipo correcto y que no haya defects ocultos.

## El pipeline completo

```python
# src/pipeline.py
import json
import time
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import load_vocab
from src.decoder.trie import build_trie
from src.prompt.prompt_builder import build_prompt
from src.decoder.constrained_generator import generate
from src.validator.output_validator import validate_output
from src.models.output import FunctionCall

def run(args) -> int:
    """Pipeline completo: load → generate → validate → output.

    Retorna 0 en éxito, 1 en falla crítica.
    """
    # 1. Cargar datos
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    # 2. Cargar modelo y vocabulario
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    trie = build_trie([fn.name for fn in functions])

    # 3. Procesar cada prompt
    results = []
    success_count = 0
    start_time = time.time()

    for i, prompt_text in enumerate(prompts):
        # 3a. Construir prompt completo
        full_prompt = build_prompt(functions, prompt_text)

        # 3b. Generar con constrained decoding
        generated, is_complete = generate(model, full_prompt, vocab, functions, trie)

        # 3c. Validar output
        validation = validate_output(generated, functions)
        if isinstance(validation, FunctionCall):
            results.append(validation.model_dump())
            success_count += 1
            print(f"  [{i+1}/{len(prompts)}] OK: {validation.name}")
        else:
            results.append({"error": str(validation), "prompt": prompt_text})
            print(f"  [{i+1}/{len(prompts)}] FAIL: {validation}")

    # 4. Escribir output
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 5. Resumen
    elapsed = time.time() - start_time
    print(f"\nProcesados {len(prompts)} prompts, {success_count} exitosos en {elapsed:.1f}s")

    return 0 if success_count > 0 else 1
```

## Escribir el output

El output se escribe en `data/output/function_calls.json` con formato:

```json
[
  {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}},
  {"name": "fn_greet", "parameters": {"name": "Alice"}},
  ...
]
```

**¿Por qué `indent=2`?** Para que el JSON sea legible por humanos. Sin indentación, todo queda en una línea gigante.

**¿Por qué `mkdir(parents=True, exist_ok=True)`?** Para crear `data/output/` si no existe. `parents=True` crea también `data/` si falta. `exist_ok=True` no lanza error si ya existe.

## ¿Por qué el pipeline retorna exit code?

- **Exit 0**: éxito (al menos un prompt procesado correctamente)
- **Exit 1**: falla crítica (no se pudieron cargar los datos, o ningún prompt tuvo éxito)

Esto es importante para CI/CD y para que el subject de 42 lo verifique: `uv run python -m src && echo "OK" || echo "FAIL"`.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Pipeline** | Secuencia de pasos que transforman input en output — load → process → validate → write |
| **Syntactic validity** | JSON parseable con estructura correcta (keys, comas, llaves) |
| **Semantic validity** | Datos correctos dentro del JSON (nombres existen, tipos son correctos) |
| **Defense in depth** | Múltiples capas de validación, no depender de una sola |
| **Exit code** | Código que el programa retorna al SO: 0 = éxito, ≠ 0 = fallo |
| **model_dump()** | Método de Pydantic que convierte un modelo a dict (para serializar a JSON) |
| **model_validate()** | Método de Pydantic que valida un dict contra el modelo |

## Checkpoint — ¿lo entendiste?

1. **¿Cuáles son las 5 validaciones del output_validator?**

<details><summary>Respuesta</summary>1) JSON parseable, 2) Estructura Pydantic correcta, 3) Nombre de función existe, 4) Tipos de parámetros correctos, 5) No hay parámetros extra.</details>

2. **¿Por qué `model_dump()` y no `json.dumps(call)`?**

<details><summary>Respuesta</summary>Pydantic no es serializable directamente a JSON. `model_dump()` convierte el modelo a un dict de Python, que `json.dump()` sí puede serializar.</details>

3. **¿Qué pasa si el validador retorna un string en vez de FunctionCall?**

<details><summary>Respuesta</summary>Significa que hay un error. El pipeline guarda el error como `{"error": "...", "prompt": "..."}` en el output y continúa con el siguiente prompt.</details>

4. **¿Por qué `parents=True` en `mkdir`?**

<details><summary>Respuesta</summary>Para crear directorios padres si no existen. Si `data/output/` no existe, `parents=True` crea también `data/`. Sin eso, fallaría porque `data/` no existe.</details>

5. **¿Qué exit code retorna el pipeline y por qué?**

<details><summary>Respuesta</summary>0 si al menos un prompt fue exitoso (éxito). 1 si ningún prompt tuvo éxito o hubo falla crítica. Esto permite que CI/CD y scripts detecten si el pipeline funcionó.</details>

---

# M11: Hardening

## Qué vas a aprender acá

En este módulo **pulís y asegurás** el proyecto: error handling completo, tests, README, lint. Es como pintar el edificio, poner la señalización, y pasar la inspección final.

## Error handling: los caminos que pueden fallar

Un proyecto serio maneja TODOS los caminos de error, no solo el "happy path":

```python
# Ejemplo de error handling en __main__.py
def main() -> None:
    """Entry point con manejo de errores."""
    try:
        args = parse_args()
        exit_code = run(args)
        sys.exit(exit_code)
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON malformado: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Datos inválidos: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR inesperado: {e}", file=sys.stderr)
        sys.exit(1)
```

**Regla**: errores en inputs → exit 1 con mensaje claro. Errores per-prompt → log warning, skip, continuar con otros.

## README: el manual de uso

El README debe ser tan claro que un dev nuevo pueda correr el proyecto con solo eso:

```markdown
# call_me_maybe

LLM function calling with constrained decoding — 42 school project.

## Quick Start

```bash
make install    # Instala dependencias
make run        # Ejecuta el pipeline
```

## Architecture

- `src/models/` — Pydantic models for input/output
- `src/loader/` — File loaders with validation
- `src/prompt/` — Prompt builder
- `src/decoder/` — Constrained decoding engine
- `src/validator/` — Post-generation validation
- `src/pipeline.py` — Orchestration

## Output Format

```json
[{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}]
```
```

## Tests: la red de seguridad

Tests unitarios verifican que cada módulo funciona correctamente SIN necesitar el modelo:

```python
# tests/test_state.py
def test_root_transitions_to_object_open():
    """ROOT → OBJECT_OPEN con '{'"""
    state = DecoderState()
    valid = state._advance_char('{')
    assert valid
    assert state.phase == DecoderPhase.OBJECT_OPEN

def test_invalid_char_in_root():
    """ROOT con carácter inválido"""
    state = DecoderState()
    valid = state._advance_char('a')
    assert not valid

# tests/test_trie.py
def test_trie_prefix_matching():
    """Trie encuentra prefixes correctos"""
    trie = build_trie(["fn_add_numbers", "fn_greet"])
    chars = valid_next_chars(trie, "fn_a")
    assert chars == {"d"}  # Solo "fn_add_numbers" continúa

# tests/test_token_filter.py
def test_filter_root_state():
    """En ROOT, solo tokens que empiezan con '{'"""
    # ... mock vocab, state en ROOT ...
    allowed = compute_allowed_ids(state, schema, vocab, trie, logits)
    # Verificar que todos los allowed empiezan con '{'
```

**¿Por qué tests SIN el modelo?** Porque el modelo pesa ~1.2GB y tarda ~2s en cargar. Los tests deben ser rápidos (<1s total). Usamos **mocks** (vocabularios simulados) para testear la lógica sin el modelo real.

## Lint: estilo y tipos

**flake8** chequea estilo (lineas muy largas, imports sin usar, etc.). **mypy** chequea tipos (que los type hints sean correctos).

```bash
make lint  # Ejecuta flake8 + mypy
```

Si `make lint` pasa, el código cumple los estándares de calidad del proyecto.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Error handling** | Manejo de errores: detectar, reportar, y recuperarse de fallos |
| **Happy path** | El camino donde todo sale bien — sin errores |
| **Exit code** | Código retornado al SO (0 = éxito, ≠0 = fallo) |
| **README** | Archivo que describe el proyecto: qué es, cómo correrlo, cómo contribuir |
| **Test unitario** | Test que verifica un módulo aislado, sin dependencias externas |
| **Mock** | Objeto simulado que reemplaza una dependencia real para testing |
| **flake8** | Herramienta de linting para Python — chequea estilo del código |
| **mypy** | Herramienta de chequeo de tipos estático para Python |
| **Coverage** | Porcentaje del código que está cubierto por tests |

## Checkpoint — ¿lo entendiste?

1. **¿Cuál es la regla de exit codes en nuestro proyecto?**

<details><summary>Respuesta</summary>0 = éxito (al menos un prompt procesado). 1 = falla crítica (no se pudieron cargar datos, o ningún prompt tuvo éxito).</details>

2. **¿Por qué los tests usan mocks en vez del modelo real?**

<details><summary>Respuesta</summary>Porque el modelo pesa ~1.2GB y tarda ~2s en cargar. Los tests deben ser rápidos. Mocks permiten testear la lógica sin cargar el modelo.</details>

3. **¿Qué chequea flake8 y qué chequea mypy?**

<details><summary>Respuesta</summary>flake8: estilo del código (líneas largas, imports sin usar, etc.). mypy: que los type hints sean correctos y consistentes.</details>

4. **¿Qué es "defense in depth" y dónde lo aplicamos?**

<details><summary>Respuesta</summary>Estrategia de múltiples capas de validación. Lo aplicamos en: Pydantic en loaders (input), constrained decoder (generación), y Pydantic en validator (output).</details>

5. **¿Por qué `data/output/` está en .gitignore?**

<details><summary>Respuesta</summary>Porque se genera en runtime. No queremos subir resultados al repo — cada ejecución puede generar resultados diferentes.</details>

---

# M12: Performance

## Qué vas a aprender acá

En este módulo entendés **por qué importa el rendimiento** y qué optimizamos. No es un módulo de código — es un módulo de **razonamiento** sobre performance.

## ¿Por qué importa?

El subject pide que el pipeline completo corra en **menos de 5 minutos en CPU**. Hagamos las cuentas:

```
11 prompts × ~50 tokens generados × ~200ms por token = ~110s
Margen para 5 min (300s): 190s para I/O, validación, carga, etc.
```

Estamos holgados... PERO solo si optimizamos el punto crítico: el **scanneo de vocabulario**.

## El bottleneck: scanneo de 151K tokens

Sin optimización, cada step del loop escanea TODOS los ~151K tokens para filtrar por primer carácter:

```
~151K tokens × ~50 steps × 11 prompts = ~83M comparaciones en CPython
```

CPython tarda ~1μs por comparación → ~83 segundos SÓLO para el filtrado. Y eso es sin contar el modelo (~110s) ni la validación.

**Solución**: pre-indexación. `tokens_starting_with[char]` es un dict que retorna el set de tokens en O(1). En vez de 83M comparaciones, tenemos ~11 × 50 × ~5K (tokens por categoría) = ~2.75M operaciones, que CPython hace en ~0.5s.

## Qué optimizamos

| Optimización | Impacto | Por qué |
|-------------|---------|---------|
| `tokens_starting_with` pre-indexed | CRÍTICO | Sin esto, ~83s de scanneo. Con esto, ~0.5s |
| `__slots__` en dataclasses del decoder | MEDIO | ~30% más rápido en creación de objetos |
| `copy.copy()` vs `copy.deepcopy()` | BAJO-MEDIO | ~50ns vs ~5μs por simulate call |
| Token simulation eficiente | BAJO | Tokens BPE son cortos (~3-5 chars avg) |

## Qué NO optimizamos

| Qué | Por qué no |
|-----|-----------|
| KV-cache | SDK no lo expone |
| Batching | SDK es single-sequence |
| GPU optimizations | El subject dice CPU |
| Cuantización | Podría degradar accuracy |
| Flash attention | No aplica a CPU |
| ONNX runtime | No es parte del scope |

## Slots vs Pydantic en el inner loop

**Pydantic** tiene overhead de ~200μs por instanciación (validación, coerción, serialización). En el inner loop:

```
200μs × 50 tokens × 11 prompts = 110ms extra
```

Es aceptable pero innecesario. El decoder no necesita validación en cada token — solo necesita guardar el estado. **Dataclass con `__slots__`** es ~30ns por instanciación, 6000x más rápido.

**Analogía**: Pydantic es un inspector de calidad que revisa CADA pieza que entra a la línea de producción. Útil para piezas que vienen de afuera (input/output), pero innecesario para piezas que se fabrican internamente (tokens en el loop).

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Bottleneck** | Punto más lento del sistema — el que limita la velocidad total |
| **Pre-indexación** | Construir índices auxiliares al inicio para hacer búsquedas más rápidas después |
| **CPython** | La implementación estándar de Python — es lenta en loops pero rápida en I/O |
| **KV-cache** | Optimización que almacena keys/values previos para no recalcularlos |
| **Batching** | Procesar múltiples inputs de una vez en vez de uno por uno |
| **`__slots__`** | Lista de atributos fija en un dataclass — ahorra memoria y tiempo |

## Checkpoint — ¿lo entendiste?

1. **¿Cuál es el bottleneck principal y cómo lo resolvemos?**

<details><summary>Respuesta</summary>El scanneo de 151K tokens en cada step. Lo resolvemos con pre-indexación: `tokens_starting_with[char]` retorna el set de tokens en O(1) en vez de escanear todo.</details>

2. **¿Por qué usamos dataclasses en vez de Pydantic en el decoder?**

<details><summary>Respuesta</summary>Pydantic tiene overhead de ~200μs por instanciación. En el inner loop (50 tokens × 11 prompts), eso suma ~110ms innecesarios. Dataclass con `__slots__` es ~30ns, 6000x más rápido.</details>

3. **¿Cuánto margen tenemos para el límite de 5 minutos?**

<details><summary>Respuesta</summary>~190 segundos. El modelo toma ~110s, el filtrado optimizado ~0.5s. Nos quedan ~189s para I/O, validación, carga, etc.</details>

4. **¿Por qué NO optimizamos KV-cache o batching?**

<details><summary>Respuesta</summary>Porque el SDK no lo expone. No tenemos acceso a KV-cache y el SDK es single-sequence (no soporta batching). Sería necesario modificar el SDK, que no podemos hacer.</details>

5. **¿Qué pasaría sin la pre-indexación?**

<details><summary>Respuesta</summary>~83M comparaciones en CPython = ~83s SÓLO para el filtrado. Sumado a los ~110s del modelo, serían ~193s. Quedaría poco margen para I/O y validación, y podríamos no cumplir 5 min.</details>

---

# M13: Lo que viene después (Bonus)

## Qué vas a aprender acá

Este módulo es **opcional** — no forma parte del MVP. Es una lista de mejoras que podés hacer DESPUÉS de que el proyecto funcione. Los bonus del subject.

## B1: lint-strict

Agregar un target `lint-strict` en el Makefile que ejecuta flake8 con reglas extendidas + mypy --strict.

## B2: Multi-modelo

Abstraer el modelo detrás de una interfaz `LLMProvider` que el CLI selecciona con `--model`.

## B3: Recode tokenizer

Usar `tokenizer.json` en vez de `vocab.json` — más metadata, más robustez.

## B4: Error recovery avanzado

En vez de skip on failure, intentar regenerar con temperatura modificada o prompt expandido.

## B5: Performance optimizations

Batch processing, KV-cache (si el SDK lo expone), profiling con cProfile, optimización de hot paths.

## B6: Test suite comprehensiva

Property-based testing con Hypothesis, coverage 90%+, tests de edge cases (strings vacíos, unicode, null).

## B7: Visualización de generación

Flag `--visualize` que muestra step-by-step con colores ANSI: verde=allowed, rojo=blocked, amarillo=current state.

## B8: Nested arguments complejos

Soporte para parameters que son objetos anidados. Extensión de la state machine y schema recursivo.

## B9: Public encode/decode API

`src/api.py` con funciones públicas `encode()`, `decode()`, `generate()` — facade sobre el pipeline para uso como librería.

## Glosario del módulo

| Término | Definición |
|---------|-----------|
| **Bonus** | Mejoras opcionales que no forman parte del MVP |
| **Property-based testing** | Testing donde se generan inputs aleatorios y se verifican propiedades generales |
| **Hypothesis** | Biblioteca de Python para property-based testing |
| **ANSI colors** | Códigos de escape para colorear texto en terminales |
| **Facade** | Patrón de diseño que simplifica una interfaz compleja detrás de una simple |

## Checkpoint — ¿lo entendiste?

1. **¿Los bonus son obligatorios?**

<details><summary>Respuesta</summary>No. Son mejoras opcionales para después del MVP funcional. El subject los lista como "BONUS" y no como "MUST".</details>

2. **¿Cuál es el bonus más impactante?**

<details><summary>Respuesta</summary>Depende de tus prioridades. B5 (performance) es útil si estás cerca del límite de 5 min. B6 (tests comprehensivos) es útil si querés alta coverage. B7 (visualización) es genial para aprender y demostrar.</details>

3. **¿Cuándo deberías implementar los bonus?**

<details><summary>Respuesta</summary>Después de que el MVP funcione y pase la evaluación. Los bonus son puntos extra, no requisitos. Primero asegurá lo básico.</details>

---

# Apéndice: Cómo usar este documento con NotebookLM

1. **Subí este archivo completo** a Google NotebookLM como fuente
2. **Creá un notebook** y seleccioná este documento como fuente
3. **Preguntá libremente**: NotebookLM va a buscar en el módulo relevante
4. **Cada módulo es autocontenido**: si preguntás sobre "trie", va a encontrar M7 sin importar que no hayas preguntado sobre los módulos anteriores

**Tips para preguntas efectivas**:
- "Explikame el filtro de tokens como si fuera un principiante" → va a M8
- "¿Cómo funciona la state machine?" → va a M6
- "¿Por qué usamos Pydantic en I/O pero no en el decoder?" → va a M2 y M12
- "¿Qué es BPE?" → va a M5
- "¿Cómo se conecta todo?" → va a M0 y M9

---

> **Fin del Plan Didáctico.** Este documento es la fuente definitiva para entender call_me_maybe desde los cimientos hasta el polishing final. Cada módulo es una pieza independiente que se puede estudiar por separado. Dale que lo construís. 🚀
