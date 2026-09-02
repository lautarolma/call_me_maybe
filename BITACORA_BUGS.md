# 🐛 BITÁCORA DE BUGS — call_me_maybe

> Registro cronológico de bugs encontrados, su causa raíz y resolución.
> **Regla**: cada entrada documenta QUÉ pasó, POR QUÉ pasó, y QUÉ aprendimos.

---

## BUG-001: `ParameterDef` definida pero nunca vinculada a `FunctionDef`

- **Fecha de detección**: 2026-08-23
- **Severidad**: MEDIA (no rompía el pipeline, pero dejaba una capa completa de validación sin efecto)
- **Estado**: ✅ RESUELTO
- **Archivos afectados**:
  - `src/models/function_definition.py` (corregido)
  - `tests/test_models.py` (actualizado)

### Síntoma

La clase `ParameterDef` existía en `function_definition.py`, pero `FunctionDef` no la usaba:

```python
# ANTES — ParameterDef era código muerto
class ParameterDef(BaseModel):
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type: 'string', 'number', 'boolean', 'null'")

class FunctionDef(BaseModel):
    parameters: dict[str, dict[str, str]] = Field(...)  # ← dict crudo, SIN validación
```

Consecuencia práctica: cualquier valor en el campo `type` de un parámetro pasaba
la validación silenciosamente:

```json
{"a": {"type": "chuchu"}}   // ✅ pasaba (NO debería)
{"a": {"type": 123}}        // ✅ pasaba (NO debería)
```

### Análisis de causa raíz

**Dato clave verificado**: el patrón incompleto está en la SPEC MISMA, no solo
en el código. `PLAN_IMPLEMENTACION.md` (Task 1.6) define exactamente ese par de
clases — `ParameterDef` por un lado y `FunctionDef.parameters:
dict[str, dict[str, str]]` por otro. El ejecutor copió la spec fielmente.

#### Atribución de autoría

| Documento | Subagente | Motor | Rol |
|-----------|-----------|-------|-----|
| `PLAN_IMPLEMENTACION.md` | lama-design | **big-pickle** | Diseño arquitectónico |
| `CRONOGRAMA_TRABAJO.md` | lama-tasks | deepseek-v4-flash-free | Planificación temporal |
| `PLAN_DIDACTICO.md` | lama-onboard | mimo-v2.5-free | Material didáctico |
| Código `src/` | Sin atribución registrada | Desconocido* | Ejecución del plan |

\* No hay git history ni metadatos que identifiquen qué motor escribió el código.
Lo único verificable es que el código coincide línea-por-línea con Task 1.6 del
plan de diseño, incluyendo el defecto.

**Nota de atribución** (fuente: criterio y memoria del operador del proyecto,
2026-08-23): el diseñador fue el subagente **lama-design** del pipeline lama,
configurado para tirar del motor big-pickle en tareas de diseño. La asignación
de motor por rol es una decisión de configuración de los agentes tomada al
armar el setup, no un dato registrable en los artefactos.

#### Precisión sobre la responsabilidad

- El error se atribuye al **pipeline agente+motor** (lama-design sobre
  big-pickle), no al modelo "a secas". El subagente produjo una spec con
  estructura muerta; eso es un fallo del output del agente en esa tarea.
- **Es un solo incidente**: NO hay evidencia suficiente para concluir que
  big-pickle tiende sistemáticamente a generar clases sin vincular en diseños.
  Lo documentado acá es UN caso, útil como señal a monitorear, no como
  veredicto.
- El hallazgo **sistémico y corregible** es otro: el flujo
  diseño → ejecución → entrega **no tenía ningún paso de verificación** que
  detectara entidades definidas pero no referenciadas (dead code estructural).
  Ese hueco de proceso es la causa raíz operativa; cualquier motor puede
  colarse por él.

**Conclusión**: la desconexión se originó en la fase de DISEÑO (subagente
lama-design / big-pickle), que definió la estructura correcta (`ParameterDef`)
pero nunca especificó su vinculación con `FunctionDef`. La implementación
replicó fielmente ese hueco porque el proceso no contemplaba detectarlo.

### Resolución

1. Se cambió `FunctionDef.parameters` a `dict[str, ParameterDef]`.
2. Se agregó `ParameterType = Literal["string", "number", "boolean", "null"]`
   para validar valores permitidos, no solo tipos.
3. `ParameterDef.name` ahora tiene default `""` y se sincroniza con la key del
   dict vía `field_validator(mode="after")` — el JSON de entrada no incluye el
   nombre dentro del objeto del parámetro (viene como key).
4. Tests nuevos: tipo inválido rechazado, tipo no-string rechazado, los 4 tipos
   documentados aceptados, sincronización de nombres.

```python
# DESPUÉS
ParameterType = Literal["string", "number", "boolean", "null"]

class ParameterDef(BaseModel):
    name: str = Field(default="", description="Parameter name")
    type: ParameterType = Field(...)

class FunctionDef(BaseModel):
    parameters: dict[str, ParameterDef] = Field(...)

    @field_validator("parameters", mode="after")
    @classmethod
    def _sync_parameter_names(cls, params): ...
```

### Verificación

- [x] `uv run pytest tests/test_models.py tests/test_loader.py -v` → 23 passed
- [x] `flake8` + `mypy` limpios
- [x] Carga end-to-end de las 5 funciones reales de `data/input/functions_definition.json`

### Lecciones aprendidas

1. **Una clase definida pero no referenciada es un smell**: si existe en el
   modelo, debe estar conectada al flujo de datos o eliminarse.
2. **El diseño también puede tener bugs**: revisar las specs antes de asumir
   que el ejecutor fue el culpable. Acá el plan contenía el defecto.
3. **Validar VALORES, no solo tipos**: `dict[str, dict[str, str]]` garantizaba
   strings, pero aceptaba cualquier string. `Literal` cierra el set de valores.
4. Cuando un modelo Pydantic recibe datos cuya identidad viene de la KEY del
   dict (y no del value), documentar explícitamente cómo se sincroniza.
5. **El pipeline necesita verificación de coherencia spec↔código**: ningún paso
   del flujo lama detectó dead code estructural. Acción preventiva sugerida:
   agregar al verify (sdd-verify / revisión post-implementación) un chequeo de
   "clases/funciones exportadas sin referencias".
6. **Atribuir con precisión**: el responsable es el pipeline agente+motor, y un
   incidente aislado no es una tendencia. Documentar como señal monitoreable,
   no como veredicto sobre el motor.

---

## Procedimiento 2026-09-02 — Config de linters (Makefile / pyproject.toml)

**Nota enunciativa**: el diseño original (lama-design) definió la config de
linters de forma mínima y no alineada al subject: el target `lint` del Makefile
solo incluía `mypy src/ --ignore-missing-imports` (1 de los 5 flags
obligatorios) y no existía `lint-strict`. Además, `ignore_missing_imports`
vivía duplicado entre `pyproject.toml` y el Makefile.

**Cambios efectuados** (mínimos, sin tocar lógica de negocio):
- `Makefile`: `lint` ahora ejecuta `flake8 .` y `mypy .` con los 5 flags
  obligatorios del subject (`--warn-return-any --warn-unused-ignores
  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`);
  se agregó el target `lint-strict` (`mypy . --strict`); se eliminó el
  `--max-line-length=120` redundante (ya vive en `.flake8`).
- `pyproject.toml`: se agregó `exclude = "(llm_sdk/|tests/)"` en `[tool.mypy]`
  — necesario para que `mypy .` no chequeé el SDK provisto ni los tests (no
  evaluados por el subject), análogo al `extend-exclude` de `.flake8`.
- `tests/test_loader.py`: se removió un `# type: ignore[arg-type]` unused
  (mypy lo resolvía como `Any` por el override de `llm_sdk`, por lo que era
  código muerto).

**Coherencia de diseño**: flake8 no lee `pyproject.toml` (solo `.flake8` /
`setup.cfg` / `tox.ini`), mientras que mypy sí — por eso la asimetría de
config (`.flake8` + `[tool.mypy]` en `pyproject.toml`) es la correcta y no una
inconsistencia.

**Verificación**: `make lint` y `make lint-strict` pasan limpios (11 archivos,
solo `src/`).

---

<!-- Próximos bugs se agregan acá abajo con el mismo formato -->
