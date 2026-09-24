"""Unit tests for src/utils/metrics.py per-phase metrics (Task 5 pre-work)."""

from __future__ import annotations

import json
from pathlib import Path

from src.decoder.state import DecoderPhase
from src.utils.metrics import MetricsRun, PhaseMetrics


def test_phase_metrics_accumulate_and_serialize() -> None:
    run = MetricsRun()
    run.add_forward(DecoderPhase.IN_KEY)
    run.add_forward(DecoderPhase.IN_KEY)
    run.add_skips(DecoderPhase.IN_KEY, 2)
    run.add_elapsed(DecoderPhase.IN_KEY, 1.5)
    run.add_forward(DecoderPhase.IN_STRING_VALUE)

    report = run.report()
    assert report["warm_up_discarded"] is True

    phases = report["phases"]
    assert phases["IN_KEY"]["total_forwards"] == 2
    assert phases["IN_KEY"]["skips_if_single"] == 2
    assert phases["IN_KEY"]["elapsed_time_ms"] == 1.5
    assert phases["IN_STRING_VALUE"]["total_forwards"] == 1

    totals = report["totals"]
    assert totals["total_forwards"] == 3
    assert totals["skips_if_single"] == 2


def test_metrics_write_json(tmp_path: Path) -> None:
    run = MetricsRun()
    run.add_forward(DecoderPhase.ROOT)
    run.add_elapsed(DecoderPhase.ROOT, 42.0)

    path = tmp_path / "nested" / "metrics_run.json"
    run.write_json(path)

    payload = json.loads(path.read_text())
    assert payload["warm_up_discarded"] is True
    assert payload["phases"]["ROOT"]["elapsed_time_ms"] == 42.0
    assert payload["phases"]["ROOT"]["total_forwards"] == 1


def test_phase_metrics_defaults() -> None:
    pm = PhaseMetrics()
    assert (pm.total_forwards, pm.skips_if_single, pm.elapsed_time_ms) == (0, 0, 0.0)
