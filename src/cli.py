"""Command-line interface parsing for call_me_maybe."""

# `from __future__ import annotations` (PEP 563) postpones type hint
# evaluation: instead of being evaluated at import time, they are stored as
# strings inside `__annotations__`. This allows modern syntax such as
# `list[str] | None` (PEP 604) even on Python versions that do not support it
# at runtime (the `|` operator between types exists since 3.10).
# It is "free" and makes the code more portable.
from __future__ import annotations

import argparse
from pathlib import Path

# Defaults are evaluated ONCE, at module import time. They are `Path` objects
# RELATIVE to the current working directory (CWD), not to the file location:
# if you run the program from another folder, these paths point elsewhere.
# `Path` is preferable to raw strings because it offers portable operations
# (`/` to join, `.exists()`, `.read_text()`...) and abstracts the Windows
# (`\`) vs Unix (`/`) differences.
DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")

# OUTPUT FILE NAME — why `function_calling_results.json`.
#
# The subject contradicts itself:
#   · the CLI example           -> `data/output/function_calls.json`
#   · the format/validation spec -> `data/output/function_calling_results.json`
#   · the testing section        -> `output/function_calling_results.json`
# And the PEER REVIEW SHEET settles it unambiguously: "Check that the output
# file is created (default: data/output/function_calling_results.json, or the
# path provided with --output)".
#
# So: 2 of 3 mentions in the subject plus the official sheet point to
# `function_calling_results.json`. That is the default. The CLI example
# mention forces nothing: it is an example of how to pass `--output`, and if
# the evaluator uses it, the program writes wherever told (`args.output`).
#
# If `--output` is passed, that path wins: the default only applies when the
# flag is absent.
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``None`` means use ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``functions_definition``, ``input`` and
        ``output`` path attributes.

    HOW IT WORKS (internals):
    - `ArgumentParser` builds an internal registry of declared arguments.
      Each `add_argument` adds an entry with: flag name, type, default and
      help.
    - `parse_args(None)` reads `sys.argv[1:]` (the process's real args,
      excluding the script name). If you pass a list, it uses that.
    - The parser tokenizes argv: it matches each `--flag value`, IGNORES the
      double dashes and maps `--functions_definition` to the
      `functions_definition` attribute (dashes become underscores).
    - The `type=Path` parameter is NOT just typing: it is a CALLABLE that
      runs over the raw argv string. Namely, it internally does
      `Path(argv_value)`. If the callable raises, argparse aborts with a
      friendly error and exit code 2 (it never reaches our code).
    - If a required argument is missing or an unknown one is given, argparse
      prints the error + usage to stderr and calls `sys.exit(2)` on its own.
      That is why there is no manual validation of any of that here.
    - It returns an `argparse.Namespace`: a bag object (like a dict but with
      attribute access). `args.input` is `args.__dict__["input"]`.
    """
    # `description` appears in the help text header (`-h/--help`), which
    # argparse generates automatically from all the add_argument calls.
    parser = argparse.ArgumentParser(
        description="LLM function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        # The default is assigned as-is when the flag is NOT present in
        # argv. Note: it is the SAME Path object for the whole process
        # lifetime; since Path is immutable, sharing it is safe.
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output JSON file",
    )
    return parser.parse_args(argv)
