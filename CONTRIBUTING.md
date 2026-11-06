# Contributing

Thanks for helping improve `trysac-hybrid-search-lite`.

## Workflow

1. Branch from `main`; never push to `main` directly.
2. `uv sync` to install the dev environment.
3. Write a failing test, then the change.
4. `uv run python scripts/check.py` must pass.
5. Update `CHANGELOG.md` under **Unreleased** and the docs the change affects.
6. Open a pull request describing the problem and the solution.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat: add strict mode to normalize
fix: reject tab-only input
docs: document error messages
```

`feat` bumps the minor version, `fix` the patch version, and a `BREAKING CHANGE:` footer the major
version.

## Code of conduct

Be respectful and constructive. Maintainers may close contributions that ignore this guide.
