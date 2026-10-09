.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ src/*/__pycache__
	rm -rf tests/__pycache__

# Subject IV.2: mandatory target, exact flag list. strict=true in
# pyproject.toml layers the subject's recommended --strict on top of
# every mypy run, including this one.
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Subject IV.2: optional target, exact commands.
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest tests/ -v
