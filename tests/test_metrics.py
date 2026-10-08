"""Unit tests for src/utils/metrics.py per-phase metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.decoder.state import DecoderPhase
from src.utils.metrics import MetricsRun, PhaseMetrics, report_prompt_metrics


def test_phase_metrics_accumulate_and_serialize() -> None:
    run = MetricsRun()
    run.add_forward(DecoderPhase.IN_KEY)
    run.add_forward(DecoderPhase.IN_KEY)
    run.add_skips(DecoderPhase.IN_KEY, 2)
    run.add_elapsed(DecoderPhase.IN_KEY, 1.5)
    run.add_forward(DecoderPhase.IN_STRING_VALUE)

    report = run.report()
    assert report["warm_up_discarded"] is True

    phases = cast(dict[str, Any], report["phases"])
    assert phases["IN_KEY"]["total_forwards"] == 2
    assert phases["IN_KEY"]["skips_if_single"] == 2
    assert phases["IN_KEY"]["elapsed_time_ms"] == 1.5
    assert phases["IN_STRING_VALUE"]["total_forwards"] == 1

    totals = cast(dict[str, Any], report["totals"])
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


def _run(forward: int, elapsed_ms: float) -> MetricsRun:
    run = MetricsRun()
    for _ in range(forward):
        run.add_forward(DecoderPhase.IN_STRING_VALUE)
    run.add_elapsed(DecoderPhase.IN_STRING_VALUE, elapsed_ms)
    return run


def test_report_prompt_metrics_persists_one_entry_per_prompt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "out" / "decode_metrics.json"

    report_prompt_metrics([_run(3, 3000.0), _run(2, 1000.0)], path)

    payload = json.loads(path.read_text())
    assert [p["index"] for p in payload["per_prompt"]] == [0, 1]
    assert [p["total_forwards"] for p in payload["per_prompt"]] == [3, 2]
    assert payload["totals"] == {"total_forwards": 5, "elapsed_time_ms": 4000.0}
    # The table goes to stdout: it is the evidence pasted into the report.
    assert "  3" in capsys.readouterr().out


def test_report_prompt_metrics_survives_a_prompt_without_forwards(
    tmp_path: Path,
) -> None:
    """A prompt solved 100% by the oracle has 0 forwards: s/fwd does not exist.

    Dividing by zero right there would bring the whole run down, which is
    exactly what the forwards count comes to document.
    """
    path = tmp_path / "decode_metrics.json"

    report_prompt_metrics([_run(0, 0.0), _run(4, 2000.0)], path)

    payload = json.loads(path.read_text())
    assert payload["per_prompt"][0]["total_forwards"] == 0
    assert payload["totals"]["total_forwards"] == 4
