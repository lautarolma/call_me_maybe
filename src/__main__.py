"""Entry point for call_me_maybe.

Usage:
    uv run python -m src [--functions_definition PATH] [--input PATH] [--output PATH]

Exit code is 0 on success and 1 on any error; errors are printed to stderr
without an unhandled traceback.

HOW THIS FILE RUNS:
- `python -m src` tells the interpreter: "find the `src` package in sys.path
  and run its `__main__.py` as a script". Python imports it with
  `__name__ == "__main__"`, which is why the block below runs.
- This pattern separates the entry point (this file, which orchestrates and
  handles errors) from the logic (cli.py and pipeline.py), leaving both
  testable without executing anything on import.
"""

from __future__ import annotations

# `sys` gives access to interpreter things: `sys.argv` (process args),
# `sys.stderr` (error stream, NOT buffered the same way as stdout),
# `sys.exit()` (terminates the process with a code).
import sys

from src.cli import parse_args
from src.pipeline import run


def main() -> int:
    """Parse arguments and run the pipeline."""
    # With no argument, parse_args reads sys.argv[1:] automatically.
    args = parse_args()
    # POSIX convention: a process returns an int to the OS; 0 = success,
    # any other value = failure. Returning the int (instead of calling
    # sys.exit() in here) keeps `main` pure and testable.
    return run(args)


if __name__ == "__main__":
    try:
        # `main()` returns the exit code; sys.exit(int) propagates it to the
        # OS. If the int is 0, the shell reads it as success ($? == 0).
        sys.exit(main())
    except Exception as exc:
        # Last-resort catch-all: it catches ANY unhandled exception
        # (FileNotFoundError, pydantic's ValidationError, Hub network
        # errors...) so the end user sees a clean message on stderr instead
        # of a full traceback.
        #
        # Tradeoff accepted here: we lose the stack trace (useful for debug)
        # in exchange for a clean UX. In development it is worth commenting
        # this except out temporarily to see the full traceback.
        #
        # Detail: it prints to STDERR, not STDOUT. The streams exist for
        # that: stdout is for DATA (redirectable to files/pipes), stderr for
        # DIAGNOSTICS. So `python -m src > output.txt` never pollutes the
        # data with error messages.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
