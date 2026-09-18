---
name: avance-mvp
description: >
  Alerta de endline y orden de ataque del proyecto call_me_maybe frente al deadline
  del MVP (21/09/2026) y el cronograma general de 3 proyectos (03/10/2026). Evalúa
  avance vs plan y emite una alerta COMPACTA (≤7 líneas) para no ensuciar la lectura.
  Trigger: al iniciar una jornada o sesión, al cerrar o abrir una task del plan de
  ejecución, en la frontera post-Task 4.3 (checkpoint de reevaluación por accuracy),
  o cuando el usuario diga "alerta endline", "orden de ataque", "¿cómo voy?",
  "¿cuánto me falta?", "avance", "deadline", "endline", "faltan X días".
license: Apache-2.0
metadata:
  author: gentleman-programming
  version: "1.0"
---

# avance-mvp

## When to Use

- Al **iniciar una jornada**: antes de cualquier trabajo, emitir la alerta.
- Al **cerrar o abrir una task** del plan de ejecución: emitir la alerta con el
  nuevo estado.
- En la **frontera post-Task 4.3**: ejecutar el CHECKPOINT DE RE-EVALUACIÓN
  (plan A/B/C abajo) y decidir la estrategia contra el deadline el MISMO día.
- Cuando el usuario pregunte por avance, retraso, días restantes o deadline.

## Critical Patterns

1. **El endline del MVP de call_me_maybe es el 21/09/2026** (hoy 18/09 → restan
   3 días calendario, incluido el buffer de fin de semana). El cronograma general
   de 3 proyectos cierra el 03/10/2026, con Flying y Codection aún sin iniciar.
2. **Salida SIEMPRE compacta** (≤7 líneas) — jamás informes largos ni tablas
   extensas en la alerta diaria. La suciedad visual entorpece; los datos finos
   viven en `docs/tracking/CRONOGRAMA_GENERAL.md` y `docs/design/CRONOGRAMA_TRABAJO.md`.
3. **Orden de ataque vigente (inamovible hasta aviso)**:
   ```
   call_me_maybe: 4.2 smoke real Qwen 0.6B → 4.3 accuracy ≥90% → 5.1 validador
                  → 5.2-5.3 pipeline E2E → 6.4 DoD → [MVP VERDE]
   luego: Flying → Codection (alcance a renegociar)
   ```
4. **MVP = Phases 1-6 SOLO.** Prohibido hasta MVP verde: Phase 7/BONUS,
   refactors fuera de plan, documentación teórica extra, features no pactadas.
5. **Puntos de foco por fase**:
   - Phase 4: thinking tokens de Qwen3 (`<|begin_of_thought|>`), timing
     ~200ms/step, RAM/CPU del primer contacto real; accuracy ≥90% (≥10/11)
     y <5 min total en Task 4.3 (la más incierta del proyecto).
   - Phase 5: validador sintáctico/semántico (5.1), pipeline + formato exacto
     del subject (5.2-5.3).
   - Phase 6: 6.4 DoD primero (lint + E2E + KPIs); 6.1/6.2/6.3 recortables.
6. **Checkpoint post-Task 4.3 (plan A/B/C — decidir el MISMO día)**:
   - **A** (accuracy ≥90% y <5 min): seguir el orden → objetivo 21-22/09.
   - **B** (80-89% o timing >5 min): 1 iteración acotada de prompts (≤0.5
     jornada) o recorte de KPI; decidir con el usuario ese mismo día.
   - **C** (<80% o bloqueante): escalar al usuario el mismo día; renegociar
     alcance/deadline. Está PROHIBIDO arrastrar el problema a la jornada
     siguiente.

## Output Format (obligatorio)

```text
🛎 ENDLINE call_me_maybe — 21/09 | restan {N} días hábiles
Estado: Phase {x}: {task actual} | suite {N} green | didáctico M{m}
FOCO: {próximas 1-2 tareas en 1 línea}
MVP: Phases 1-6 SOLO (7/refactor/doc-extra = prohibido hasta verde)
⚠ {riesgo o decisión pendiente — 1 línea, solo si aplica}
```

Máximo 7 líneas. Sin markdown pesado, sin tablas, sin historial.

## Commands

- Al iniciar sesión / cerrar-abrir task: **"alerta endline"** → ejecutar la
  evaluación y emitir la alerta en el formato de arriba.
- Post-Task 4.3: **"checkpoint 4.3"** → medir accuracy + timing, aplicar
  plan A/B/C y dejar por escrito la decisión en PROGRESS_TRACKER.md.
- Consulta de estado: **"¿cómo voy?" / "¿cuánto me falta?"** → misma alerta
  compacta + 1 línea extra con la recomendación del día.

## Resources

- **Plan maestro**: [docs/tracking/CRONOGRAMA_GENERAL.md](../../docs/tracking/CRONOGRAMA_GENERAL.md)
- **Cronograma de diseño (21 jornadas)**: [docs/design/CRONOGRAMA_TRABAJO.md](../../docs/design/CRONOGRAMA_TRABAJO.md)
- **Premisa en la guía**: [docs/tracking/GUIA_RAPIDA.md](../../docs/tracking/GUIA_RAPIDA.md)
- **Registro diario**: [docs/tracking/PROGRESS_TRACKER.md](../../docs/tracking/PROGRESS_TRACKER.md)
- **Tareas por hacer**: [docs/design/PLAN_IMPLEMENTACION.md](../../docs/design/PLAN_IMPLEMENTACION.md)