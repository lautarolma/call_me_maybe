#!/usr/bin/env python3
# mypy: ignore-errors
"""Machine fingerprint + end-to-end latency benchmark for call_me_maybe.

PORTABLE MEASUREMENT TOOL — TEMPORARY. It lives at the REPO ROOT as
``bench_machine.py`` so a plain ``git clone`` ships it, and DELETE the file once
the campus measurement campaign is over (it is not meant to be part of the
submission). It can also be copied to any folder inside the project tree (it
auto-detects the repo root by walking up until it finds ``pyproject.toml``);
pass ``--repo-root`` to override when it lives outside the repo.

Run it from the repo root::

    uv run python bench_machine.py --label campus-corriente
    uv run python bench_machine.py --devices cpu gpu --label campus-uno
    uv run python bench_machine.py --dry-run

What it does:

1. Prints a hardware + software fingerprint: CPU model, logical/available
   cores, RAM, torch build, CUDA/MPS availability, and the device/dtype/thread
   count the SDK would auto-select on this machine.
2. Runs the full pipeline as a SUBPROCESS, streaming its output live, and
   measures the WALL-CLOCK time of the whole process.
3. Optionally runs several passes (``--devices cpu gpu``) on the SAME machine:
   the CPU pass hides the GPUs with ``CUDA_VISIBLE_DEVICES=""`` and the GPU pass
   leaves them visible. The SDK auto-selects the device, so ``src/`` is never
   touched.
4. Reports BOTH the process wall time and the internal generation time, plus
   the hardware-independent forward count read from ``decode_metrics.json``.
5. Verdicts PASS/FAIL against the subject KPI (suite of 11 prompts < 5 min).
6. Writes a per-machine JSON report so runs from different machines can be
   diffed afterwards.

GPU PREREQUISITE: the shipped environment pins ``torch==...+cpu`` (see the
``pytorch-cpu`` index in ``pyproject.toml``), and a ``+cpu`` wheel can NEVER
run CUDA. The ``gpu`` pass therefore requires a one-time environment change:
a CUDA torch build plus ``accelerate`` (which the SDK needs for
``device_map="auto"``). Use ``--setup-gpu`` to apply it automatically, or run
the two ``uv pip install`` commands it prints. Revert afterwards with
``uv sync --reinstall`` (restores the pinned CPU build from ``uv.lock``).

Why as a subprocess instead of importing the pipeline: ``CUDA_VISIBLE_DEVICES``
must be set BEFORE torch is imported, and a fresh process yields a clean,
uncontaminated measurement. Threads are NEVER capped: each pass runs at the
machine's full capacity (torch defaults to ``nproc``), so the measured latency
reflects what the evaluator would see.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

KPI_SECONDS_DEFAULT = 300.0  # subject KPI: suite of 11 prompts < 5 minutes
GEN_TIME_RE = re.compile(r"\[Timing\] Complete run: ([\d.]+) ms")

# Default CUDA wheel index for --setup-gpu. MUST match the driver's CUDA
# version; override with --gpu-index-url when the campus machine needs another
# (e.g. cu121, cu126). See https://pytorch.org/get-started/locally/.
GPU_INDEX_URL_DEFAULT = "https://download.pytorch.org/whl/cu124"

# Probe run as a child process to read the REAL device/dtype the SDK would
# pick under a given environment (env vars affect torch only at import time,
# so this cannot be answered in-process after torch is loaded).
PROBE_CODE = """
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({"torch_available": False, "error": str(exc)}))
    raise SystemExit(0)
try:
    import accelerate
    accelerate_available = True
except Exception:
    accelerate_available = False
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(json.dumps({
    "torch_available": True,
    "torch_version": torch.__version__,
    "cuda_is_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "mps_is_available": bool(torch.backends.mps.is_available()),
    "would_select_device": device,
    "would_select_dtype": "float16" if device in ("cuda", "mps") else "float32",
    "accelerate_available": accelerate_available,
    "num_threads": int(torch.get_num_threads()),
    "num_interop_threads": int(torch.get_num_interop_threads()),
}))
"""


def detect_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory holding ``pyproject.toml``.

    Makes the tool location-independent: it works whether it sits in ``docs/``,
    in the repo root, or in any other folder inside the project. Falls back to
    ``start`` when no project root is found (then use ``--repo-root``).
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _read_mem_total_gib() -> float | None:
    """Best-effort total system RAM in GiB (Linux ``/proc/meminfo``)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return round(kib / (1024.0 * 1024.0), 2)
    except OSError:
        pass
    return None


def _read_cpu_model() -> str:
    """Best-effort CPU model string (Linux ``/proc/cpuinfo``)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _hf_model_cached() -> bool:
    """True if the Qwen3-0.6B weights are already in the HF hub cache."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    try:
        return any(p.name.startswith("models--Qwen--Qwen3") for p in hub.iterdir())
    except OSError:
        return False


def torch_fingerprint() -> dict[str, object]:
    """Describe the installed torch build and the device the SDK would pick.

    Mirrors the auto-selection in ``llm_sdk/llm_sdk/__init__.py``
    (mps > cuda > cpu, then float16 on GPU/MPS and float32 on CPU) so the
    report states, up front, whether this machine can run on GPU at all.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"torch_available": False, "error": str(exc)}

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "mps_is_available": bool(torch.backends.mps.is_available()),
        "would_select_device": device,
        "would_select_dtype": "float16" if device in ("cuda", "mps") else "float32",
    }


def collect_fingerprint() -> dict[str, object]:
    """Gather the machine + runtime fingerprint for the report."""
    try:
        available_cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # not Linux
        available_cores = os.cpu_count()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": _read_cpu_model(),
        "logical_cores": os.cpu_count(),
        "available_cores": available_cores,
        "mem_total_gib": _read_mem_total_gib(),
        "hf_model_cached": _hf_model_cached(),
        **torch_fingerprint(),
    }


def print_fingerprint(fp: dict[str, object]) -> None:
    """Pretty-print the fingerprint to the console."""
    print("=== Machine fingerprint ===")
    for key in (
        "hostname",
        "platform",
        "python_version",
        "cpu_model",
        "logical_cores",
        "available_cores",
        "mem_total_gib",
        "hf_model_cached",
        "torch_version",
        "cuda_is_available",
        "cuda_version",
        "mps_is_available",
        "would_select_device",
        "would_select_dtype",
        "error",
    ):
        if key in fp:
            print(f"  {key:<20}: {fp[key]}")
    print()


def probe_environment(env: dict[str, str], repo_root: Path) -> dict[str, object]:
    """Ask a child process which device/dtype the SDK would select with ``env``."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", PROBE_CODE],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": str(exc)}
    for line in reversed(out.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"probe_error": out.stderr.strip() or "no JSON from probe"}


def build_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    """Build the pipeline subprocess command using the current interpreter."""
    return [
        sys.executable,
        "-m",
        "src",
        "--functions_definition",
        str(args.functions),
        "--input",
        str(args.input),
        "--output",
        str(out_json),
    ]


def build_env(device: str) -> dict[str, str]:
    """Return the child environment for a pass: full threads, device isolated.

    Threads are deliberately NOT capped, so each pass runs at the machine's
    full capacity (torch defaults to ``nproc``) — exactly what the evaluator
    would see. For the CPU pass, ``CUDA_VISIBLE_DEVICES=""`` hides the GPUs so
    the SDK auto-selects ``cpu``; the GPU pass leaves them visible.
    """
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_pipeline(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout: float,
    repo_root: Path,
) -> tuple[int, float, list[str]]:
    """Run the pipeline, stream + capture its output, and time the wall clock.

    Returns ``(exit_code, wall_seconds, lines)``. The child's stdout and stderr
    are merged so the warnings the pipeline prints to stderr end up in the same
    log and console stream.
    """
    from time import perf_counter

    lines: list[str] = []
    start = perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    reader.join(timeout=5)
    wall_s = perf_counter() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(lines), encoding="utf-8")

    if timed_out:
        print(f"\n[bench] TIMEOUT after {timeout:.0f}s — process killed.", file=sys.stderr)
        return 124, wall_s, lines
    return proc.returncode, wall_s, lines


def parse_generation_ms(lines: list[str]) -> float | None:
    """Extract the internal generation time printed by ``measure_time``."""
    for line in lines:
        match = GEN_TIME_RE.search(line)
        if match:
            return float(match.group(1))
    return None


def read_forwards(out_json: Path) -> int | None:
    """Read the hardware-independent forward total from ``decode_metrics.json``."""
    metrics_path = out_json.parent / "decode_metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return int(payload["totals"]["total_forwards"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def setup_gpu(index_url: str, repo_root: Path) -> int:
    """Install a CUDA torch build + accelerate into the current environment.

    One-time, opt-in step: it MUTATES the project venv. Revert afterwards with
    ``uv sync --reinstall`` (restores the CPU build pinned in ``uv.lock``).
    """
    commands = [
        [
            "uv", "pip", "install", "--python", sys.executable,
            "--index-url", index_url, "torch", "--reinstall-package", "torch",
        ],
        ["uv", "pip", "install", "--python", sys.executable, "accelerate"],
    ]
    for cmd in commands:
        print(f"[bench] $ {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, cwd=repo_root)
        except FileNotFoundError:
            print(
                "[bench] ERROR: `uv` not found on PATH. Run this step manually.",
                file=sys.stderr,
            )
            return 127
        if proc.returncode != 0:
            print(f"[bench] ERROR: command failed with code {proc.returncode}.", file=sys.stderr)
            return proc.returncode
    print(
        "\n[bench] GPU environment installed. To revert to the CPU build "
        "(needed for the submission): uv sync --reinstall\n"
    )
    return 0


def run_pass(
    args: argparse.Namespace, device: str, base_dir: Path, repo_root: Path
) -> dict[str, object]:
    """Run one measurement pass (``cpu`` or ``gpu``) and return its report."""
    print(f"\n########## PASS: {device.upper()} ##########")
    env = build_env(device)
    probe = probe_environment(env, repo_root)
    print("=== Device probe (what the SDK will actually pick) ===")
    for key in (
        "torch_version",
        "cuda_is_available",
        "cuda_device_count",
        "would_select_device",
        "would_select_dtype",
        "accelerate_available",
        "num_threads",
        "num_interop_threads",
        "probe_error",
    ):
        if key in probe:
            print(f"  {key:<22}: {probe[key]}")
    print()

    if device == "gpu" and not probe.get("cuda_is_available") and not probe.get("mps_is_available"):
        sys.stdout.flush()
        print(
            "[bench] SKIPPED: the GPU pass is impossible in this environment "
            "(no CUDA/MPS visible to torch). A '+cpu' torch wheel can never "
            "run CUDA. Install a CUDA build + accelerate first:\n"
            f"  uv pip install --python {sys.executable} "
            f"--index-url {args.gpu_index_url} torch --reinstall-package torch\n"
            f"  uv pip install --python {sys.executable} accelerate\n"
            "  (or re-run with --setup-gpu)",
            file=sys.stderr,
        )
        return {"device_requested": device, "skipped": True, "probe": probe}

    if device == "gpu" and not probe.get("accelerate_available"):
        sys.stdout.flush()
        print(
            "[bench] WARNING: `accelerate` is missing; the SDK uses "
            "device_map='auto' on cuda and will crash at model load. Install "
            "it: uv pip install accelerate",
            file=sys.stderr,
        )

    pass_dir = base_dir / device
    out_json = pass_dir / "function_calling_results.json"
    cmd = build_command(args, out_json)

    print("=== Run ===")
    print(f"  command : {' '.join(cmd)}")
    print(f"  cwd     : {repo_root}")
    print("  threads : full capacity (torch default = nproc)")
    print(f"  KPI     : {args.kpi:.0f} s")
    print(f"  output  : {out_json}")
    print()

    if args.dry_run:
        print("[bench] --dry-run: not executing the pipeline.")
        return {
            "device_requested": device,
            "dry_run": True,
            "probe": probe,
            "command": cmd,
        }

    code, wall_s, lines = run_pipeline(cmd, env, pass_dir / "run.log", args.timeout, repo_root)
    gen_ms = parse_generation_ms(lines)
    forwards = read_forwards(out_json)

    return {
        "device_requested": device,
        "device_used": probe.get("would_select_device"),
        "dtype_used": probe.get("would_select_dtype"),
        "num_threads": probe.get("num_threads"),
        "exit_code": code,
        "wall_seconds": round(wall_s, 3),
        "generation_seconds": round(gen_ms / 1000.0, 3) if gen_ms is not None else None,
        "total_forwards": forwards,
        "seconds_per_forward": round(wall_s / forwards, 4) if forwards else None,
        "kpi_pass": bool(code == 0 and wall_s <= args.kpi),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Path defaults are resolved against the detected repo root inside
    :func:`main` (so ``--repo-root`` can override it), hence they are ``None``
    here.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root (default: auto-detected by walking up for pyproject.toml).",
    )
    parser.add_argument(
        "--functions",
        type=Path,
        default=None,
        help="Path to the functions definition JSON (default: <root>/data/input/...).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the input prompts JSON (default: <root>/data/input/...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for per-run outputs and reports (default: <root>/data/output/bench).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "gpu"),
        default=["cpu"],
        help="Passes to run on this machine (default: cpu).",
    )
    parser.add_argument(
        "--kpi",
        type=float,
        default=KPI_SECONDS_DEFAULT,
        help=f"Latency KPI in seconds for the suite (default {KPI_SECONDS_DEFAULT:.0f}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Kill a pass if it exceeds this many seconds (default 3600).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Tag for the report/run id (e.g. 'campus-corriente'). Default: hostname.",
    )
    parser.add_argument(
        "--setup-gpu",
        action="store_true",
        help="Install a CUDA torch build + accelerate before measuring (mutates the venv).",
    )
    parser.add_argument(
        "--gpu-index-url",
        type=str,
        default=GPU_INDEX_URL_DEFAULT,
        help=f"CUDA wheel index for --setup-gpu (default {GPU_INDEX_URL_DEFAULT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fingerprint, probe and command, then exit without running.",
    )
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the repo root and fill in the path defaults. Returns (root, pyproject)."""
    if args.repo_root is not None:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = detect_repo_root(Path(__file__).resolve().parent)
    args.repo_root = repo_root
    args.functions = args.functions or repo_root / "data/input/functions_definition.json"
    args.input = args.input or repo_root / "data/input/function_calling_tests.json"
    args.output_dir = args.output_dir or repo_root / "data/output/bench"
    return repo_root, repo_root / "pyproject.toml"


def main(argv: list[str] | None = None) -> int:
    """Entry point: fingerprint, run each requested pass, report verdicts."""
    args = parse_args(argv)
    repo_root, pyproject = _resolve_paths(args)

    if not pyproject.is_file():
        print(
            f"[bench] ERROR: pyproject.toml not found at {pyproject}. Point "
            "--repo-root at the project root.",
            file=sys.stderr,
        )
        return 2

    fp = collect_fingerprint()
    print_fingerprint(fp)
    print(f"=== Repo ===\n  root : {repo_root}\n")

    if not fp.get("hf_model_cached", False):
        print(
            "[bench] WARNING: Qwen3-0.6B weights are NOT in the HF cache. The "
            "first run will DOWNLOAD them, inflating wall time. Warm the cache "
            "once before measuring.\n"
        )

    if args.setup_gpu:
        rc = setup_gpu(args.gpu_index_url, repo_root)
        if rc != 0:
            return rc
        fp = collect_fingerprint()  # refresh after the environment change
        print_fingerprint(fp)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or str(fp["hostname"])
    base_dir = args.output_dir / f"{label}_{stamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    passes = [run_pass(args, device, base_dir, repo_root) for device in args.devices]

    report: dict[str, object] = {
        "run_id": f"{label}_{stamp}",
        "label": label,
        "timestamp_utc": stamp,
        "fingerprint": fp,
        "kpi_seconds": args.kpi,
        "passes": passes,
    }
    report_path = base_dir / "bench_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"  {'device':<8} {'used':<6} {'threads':>7} {'wall(s)':>9} {'gen(s)':>8} {'forwards':>9} {'KPI':>5}")
    for entry in passes:
        if entry.get("skipped"):
            print(f"  {entry['device_requested']:<8} {'-':<6} {'SKIPPED':>7}")
            continue
        if entry.get("dry_run"):
            used = str(entry.get("probe", {}).get("would_select_device", "?"))
            nthreads = str(entry.get("probe", {}).get("num_threads", "?"))
            print(f"  {entry['device_requested']:<8} {used:<6} {nthreads:>7} {'dry-run':>9}")
            continue
        verdict = "PASS" if entry["kpi_pass"] else "FAIL"
        gen = entry["generation_seconds"]
        print(
            f"  {entry['device_requested']:<8} {str(entry['device_used']):<6} "
            f"{str(entry['num_threads']):>7} "
            f"{entry['wall_seconds']:>9.2f} "
            f"{(f'{gen:.2f}' if gen is not None else 'n/a'):>8} "
            f"{str(entry['total_forwards'] if entry['total_forwards'] is not None else 'n/a'):>9} "
            f"{verdict:>5}"
        )
    print(f"\n  report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
