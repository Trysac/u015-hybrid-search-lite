# Testing

## Running

```sh
uv run pytest --cov=hybrid_search_lite   # tests with coverage report
uv run python scripts/check.py      # every gate, as CI and the factory run it
```

## Policy

- Branch coverage of `src/hybrid_search_lite` must stay at or above 80 %; the gate fails below it.
- Tests exercise the public API (`import hybrid_search_lite`) and assert every observable effect:
  return values, raised exceptions and their messages.
- Each bug fix starts with a failing test that reproduces it.
- Tests are deterministic: no network, no wall clock, no dependence on test order.

## Layout

`tests/test_<module>.py` mirrors each public module. Shared fixtures go in `tests/conftest.py`.
