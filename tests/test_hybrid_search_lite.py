"""Behavior tests for the public API of hybrid_search_lite."""

from __future__ import annotations

import pytest

import hybrid_search_lite


def test_normalize_trims_and_lowercases() -> None:
    assert hybrid_search_lite.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hybrid_search_lite.normalize(value)


def test_version_is_semver() -> None:
    parts = hybrid_search_lite.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
