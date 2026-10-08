"""Metrics and timing facilities for the pipeline and benchmarks.

Centralizes how the pipeline measures and reports per-prompt tracking and
how benchmarks capture per-phase generation metrics.

- ``measure_time`` is re-exported from ``src.utils.timer`` so callers depend
  on this module (the metrics interface) instead of the timer implementation.
- ``PhaseMetrics`` / ``MetricsRun`` provide a typed, serializable breakdown of
  a generation run: total forwards, skip-if-single count and elapsed time,
  bucketed by :class:`~src.decoder.state.DecoderPhase`. The benchmark writer
  is responsible for discarding the warm-up pass (see the benchmark scripts);
  ``MetricsRun.write_json`` reports with ``warm_up_discarded: true``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterator

from src.decoder.state import DecoderPhase
from src.utils.timer import measure_time

__all__ = [
    "measure_time",
    "track_prompt",
    "report_prompt_metrics",
    "PhaseMetrics",
    "MetricsRun",
]


@dataclass
class PhaseMetrics:
    """Per-phase counters for one :class:`DecoderPhase` bucket.

    Attributes:
        total_forwards: Exact count of model forward calls made while the
            generation was in this phase.
        skips_if_single: Count of tokens skipped by the M5 skip-if-single
            logic while in this phase (forward call avoided).
        elapsed_time_ms: Exact wall-clock time spent in this phase.
    """

    total_forwards: int = 0
    skips_if_single: int = 0
    elapsed_time_ms: float = 0.0


@dataclass
class MetricsRun:
    """Accumulator of per-phase generation metrics for a benchmark run.

    A single instance can be shared across prompts; counters accumulate per
    phase. Use :meth:`to_dict` / :meth:`write_json` for the structured report.
    """

    phases: dict[str, PhaseMetrics] = field(default_factory=dict)

    def _bucket(self, phase: DecoderPhase) -> PhaseMetrics:
        """Return (and lazily create) the metrics bucket for a phase."""
        key = phase.name
        bucket = self.phases.get(key)
        if bucket is None:
            bucket = PhaseMetrics()
            self.phases[key] = bucket
        return bucket

    def add_forward(self, phase: DecoderPhase) -> None:
        """Record one model forward call made while in ``phase``."""
        self._bucket(phase).total_forwards += 1

    def add_skips(self, phase: DecoderPhase, count: int = 1) -> None:
        """Record ``count`` skip-if-single tokens skipped while in ``phase``."""
        self._bucket(phase).skips_if_single += count

    def add_elapsed(self, phase: DecoderPhase, elapsed_ms: float) -> None:
        """Accumulate ``elapsed_ms`` wall-clock time attributed to ``phase``."""
        self._bucket(phase).elapsed_time_ms += elapsed_ms

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Serialize the per-phase breakdown as a JSON-ready mapping."""
        return {
            name: {
                "total_forwards": bucket.total_forwards,
                "skips_if_single": bucket.skips_if_single,
                "elapsed_time_ms": round(bucket.elapsed_time_ms, 3),
            }
            for name, bucket in sorted(self.phases.items())
        }

    def totals(self) -> dict[str, int | float]:
        """Aggregate counters across all phases (JSON-ready)."""
        return {
            "total_forwards": sum(b.total_forwards for b in self.phases.values()),
            "skips_if_single": sum(b.skips_if_single for b in self.phases.values()),
            "elapsed_time_ms": round(
                sum(b.elapsed_time_ms for b in self.phases.values()), 3
            ),
        }

    def report(self) -> dict[str, object]:
        """Full structured report payload.

        Includes the ``warm_up_discarded`` flag: the benchmark protocol
        always runs one discarded warm-up pass before measuring (see the
        criterion in the benchmark scripts).
        """
        return {
            "warm_up_discarded": True,
            "totals": self.totals(),
            "phases": self.to_dict(),
        }

    def write_json(self, path: Path) -> None:
        """Write the structured report to ``path`` (creates parents)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2, sort_keys=True))


@contextmanager
def track_prompt(index: int) -> Iterator[None]:
    """Measure and report wall-clock time of a single prompt generation.

    Args:
        index: Zero-based index of the prompt being generated.

    Prints:
        ``[Started] Prompt <index>`` before the block and
        ``[Timming] Prompt <index>: <elapsed> ms`` after it, keeping the
        timing concerns out of the pipeline loop.
    """
    print(f"[Started] Prompt {index}")
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        print(f"[Timming] Prompt {index}: {elapsed_ms} ms")


def report_prompt_metrics(
    runs: Sequence[MetricsRun], out_path: Path | None = None
) -> None:
    """Print and persist the forward count PER PROMPT.

    WHY ONE `MetricsRun` PER PROMPT instead of a single accumulator: the
    forward count is the ONLY decoding magnitude that does NOT depend on the
    hardware. There is no KV-cache, so a forward means "re-feed the whole
    sequence", and *how many* times that must happen is decided by the
    grammar (state + oracle + filter), not by the CPU. Comparing that number
    between two runs is what separates the two hypotheses that share the same
    symptom:

        forwards EQUAL    -> the code did the same work and the extra time
                             is ENVIRONMENT (compilation, thermal, noise).
        forwards DIFFERENT -> the code changed the work; optimize.

    The `s/fwd` column is the counterpart: if it rises while the forward
    count stays the same, the machine was slower per operation.

    WHY IT DOES NOT REUSE `MetricsRun.report()`: that method hardcodes
    `warm_up_discarded: True`, which describes the benchmark protocol (one
    warm-up pass is discarded). The pipeline discards none, so writing that
    `true` into the JSON would be false data in the evidence.

    Args:
        runs: One `MetricsRun` per prompt, in the order they ran.
        out_path: If given, writes the same breakdown as JSON.
    """
    print()
    print("=== Per-prompt decode metrics ===")
    print(f"  {'#':>3}  {'forwards':>8}  {'skips':>6}  {'elapsed(s)':>10}  {'s/fwd':>6}")

    per_prompt: list[dict[str, object]] = []
    for index, prompt_run in enumerate(runs):
        totals = prompt_run.totals()
        forwards = int(totals["total_forwards"])
        elapsed_s = float(totals["elapsed_time_ms"]) / 1000.0
        # With no forwards the quotient does not exist (all oracle/M5): show
        # n/a instead of dividing by zero and killing the run.
        s_per_fwd = f"{elapsed_s / forwards:>6.2f}" if forwards else f"{'n/a':>6}"
        print(
            f"  {index:>3}  {forwards:>8}  {int(totals['skips_if_single']):>6}"
            f"  {elapsed_s:>10.2f}  {s_per_fwd}"
        )
        per_prompt.append(
            {
                "index": index,
                "total_forwards": forwards,
                "skips_if_single": int(totals["skips_if_single"]),
                "elapsed_time_ms": float(totals["elapsed_time_ms"]),
                "phases": prompt_run.to_dict(),
            }
        )

    total_forwards = sum(int(r.totals()["total_forwards"]) for r in runs)
    total_ms = sum(float(r.totals()["elapsed_time_ms"]) for r in runs)
    total_s_per_fwd = f"{total_ms / total_forwards / 1000.0:>6.2f}" if total_forwards else "    n/a"
    print(f"  {'ALL':>3}  {total_forwards:>8}  {'':>6}  {total_ms / 1000.0:>10.2f}  {total_s_per_fwd}")

    if out_path is not None:
        payload = {
            "per_prompt": per_prompt,
            "totals": {
                "total_forwards": total_forwards,
                "elapsed_time_ms": round(total_ms, 3),
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"  wrote decode metrics -> {out_path}")
