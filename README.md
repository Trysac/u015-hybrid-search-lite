# hybrid-search-lite

[![PyPI version](https://img.shields.io/pypi/v/trysac-hybrid-search-lite)](https://pypi.org/project/trysac-hybrid-search-lite/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-hybrid-search-lite)](LICENSE)
[![CI](https://github.com/Trysac/u015-hybrid-search-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u015-hybrid-search-lite/actions/workflows/ci.yml)

In-memory hybrid search fusing BM25 and cosine vector rankings with reciprocal rank fusion, metadata filters and pickle-free SQLite persistence

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-hybrid-search-lite
# or
uv add trysac-hybrid-search-lite
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import hybrid_search_lite

# normalize() trims surrounding whitespace and lower-cases the text.
print(hybrid_search_lite.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    hybrid_search_lite.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
