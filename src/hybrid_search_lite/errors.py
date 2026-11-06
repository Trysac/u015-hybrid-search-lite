"""Typed errors for hybrid_search_lite."""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "dimension_mismatch",
    "index_mismatch",
    "index_full",
    "input_too_large",
    "corrupt_file",
    "invalid_argument",
]

_MAX_MESSAGE_CHARS = 200


class SearchError(ValueError):
    """Base error of the library.

    Attributes:
        code: Stable machine-readable error code.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        if len(message) > _MAX_MESSAGE_CHARS:
            limit = _MAX_MESSAGE_CHARS - 3
            message = message[:limit] + "..."
        super().__init__(message)
        self.code: ErrorCode = code
