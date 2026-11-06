"""Dependency-free hybrid search: BM25 and cosine fused with reciprocal rank fusion."""

from __future__ import annotations

from .errors import SearchError
from .index import (
    Document,
    EmbedderPort,
    HybridIndex,
    SearchHit,
    default_tokenizer,
    reciprocal_rank_fusion,
)

__all__ = [
    "Document",
    "EmbedderPort",
    "HybridIndex",
    "SearchError",
    "SearchHit",
    "__version__",
    "default_tokenizer",
    "reciprocal_rank_fusion",
]

__version__ = "0.1.0"
