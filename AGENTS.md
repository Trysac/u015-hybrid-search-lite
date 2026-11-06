# AGENTS.md

Instructions for coding agents (and humans) working on `trysac-hybrid-search-lite`.

## Commands

| Task | Command |
|---|---|
| Install dev environment | `uv sync` |
| Run every gate (must pass before a PR) | `uv run python scripts/check.py` |
| Tests only | `uv run pytest --cov=hybrid_search_lite` |
| Lint / auto-fix | `uv run ruff check .` / `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Types | `uv run mypy` |
| Build | `uv build` |

The gate sequence is: ruff lint, ruff format check, mypy strict, pytest with branch coverage >= 80 %,
documentation manifest (`docs-manifest.json`), gitleaks. It stops at the first failure.

## Folder map

| Path | Content |
|---|---|
| `src/hybrid_search_lite/` | The package. Public API is whatever `__init__.py` exports in `__all__`. |
| `tests/` | pytest tests, one file per public module, exercising the public API. |
| `scripts/check.py` | The single definition of the quality gates. |
| `docs/` | Architecture, decisions (ADRs), testing, security, project record, stack. |
| `fabrica.json` | Install and check commands used by the factory and CI. |
| `docs-manifest.json` | Documentation files the docs gate requires. |
| `.github/` | CI wrapper and Dependabot configuration. |

## Conventions

- Python >= 3.10, fully typed (`mypy --strict`), `py.typed` ships with the package.
- No runtime dependencies unless an ADR in `docs/decisions/` justifies one.
- Write the failing test first; test through the public API, not private helpers.
- Every behavior change updates `README.md` usage, `CHANGELOG.md` (Unreleased) and, if it is a
  design choice, a new ADR.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Limits

- Never push to `main`; work on a branch and open a pull request.
- Do NOT edit `.github/`, `fabrica.json`, `docs-manifest.json` or `scripts/`. They are the quality
  contract; changes to them are made by a human.
- Never lower the coverage threshold, disable a lint rule or add `# type: ignore` to make a gate pass.
- Never commit secrets, tokens or `.env` files; gitleaks blocks them.
