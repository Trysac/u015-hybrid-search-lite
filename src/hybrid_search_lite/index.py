"""In-memory hybrid search: BM25, cosine vectors and reciprocal rank fusion."""

from __future__ import annotations

import json
import logging
import math
import operator
import os
import re
import sqlite3
import sys
import unicodedata
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from .errors import SearchError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MetadataValue = str | int | float | bool
SearchMode = Literal["keyword", "vector", "hybrid"]

_MODES = ("keyword", "vector", "hybrid")
_MAX_ID_CHARS = 256
_MAX_TEXT_CHARS = 1_000_000
_MAX_METADATA_ITEMS = 100
_MAX_METADATA_STRING = 10_000
_MAX_DIMENSION = 65_536
_MAX_COMPONENT = 1e150
_EMBED_BATCH = 64
_MAX_RANKINGS = 64
_MAX_RANKING_LENGTH = 100_000
_MAX_FILTER_KEYS = 100
_MAX_FILTER_VALUES = 1_000
_MAX_META_ROWS = 32
_SCHEMA = "1"
_TOKEN_RE = re.compile(r"[^\W_]+")

_Filter = tuple[str, tuple[object, ...]]


def _invalid(message: str) -> SearchError:
    return SearchError("invalid_argument", message)


def _corrupt(message: str) -> SearchError:
    return SearchError("corrupt_file", message)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _check_id(doc_id: object) -> None:
    if not isinstance(doc_id, str) or not 1 <= len(doc_id) <= _MAX_ID_CHARS:
        raise _invalid("id must be a string of 1 to 256 characters")


def _check_int_min(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _invalid(f"{name} must be an integer >= {minimum}")


def _check_bm25(k1: object, b: object) -> None:
    if not _is_finite_number(k1) or not _is_finite_number(b):
        raise _invalid("k1 and b must be finite numbers")
    if k1 < 0 or not 0 <= b <= 1:  # type: ignore[operator]
        raise _invalid("k1 must be >= 0 and b must be within [0, 1]")


def _check_rrf_k(k: object) -> None:
    if not _is_finite_number(k) or k < 0:  # type: ignore[operator]
        raise _invalid("rrf_k must be a finite number >= 0")


def _check_weights(weights: object, expected: int) -> list[float]:
    if isinstance(weights, str | bytes) or not isinstance(weights, Sequence):
        raise _invalid("weights must be a sequence of numbers")
    if len(weights) != expected:
        raise _invalid(f"weights must contain exactly {expected} values")
    checked: list[float] = []
    for weight in weights:
        if not _is_finite_number(weight) or weight < 0:
            raise _invalid("weights must be finite numbers >= 0")
        checked.append(float(weight))
    return checked


def _rank(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def default_tokenizer(text: str) -> list[str]:
    """Split ``text`` into lower-cased alphanumeric tokens.

    Applies NFKC normalization and case folding, then splits on every character
    that is not a letter or digit. No stemming is performed.

    Raises:
        SearchError: ``invalid_argument`` for non-strings, ``input_too_large``
            for text longer than 1,000,000 characters.
    """
    if not isinstance(text, str):
        raise _invalid("text must be a string")
    if len(text) > _MAX_TEXT_CHARS:
        raise SearchError("input_too_large", "text exceeds 1000000 characters")
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: float = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with reciprocal rank fusion (Cormack et al., SIGIR 2009).

    ``score(d) = sum(weight_i / (k + rank_i(d)))`` with 1-based ranks. An id that
    repeats inside one ranking only counts at its first position. Results are
    sorted by descending score, ties by id ascending.

    Raises:
        SearchError: ``invalid_argument`` for a bad ``k``, ``weights`` or id, and
            ``input_too_large`` for more than 64 rankings or more than 100,000
            entries in one ranking.
    """
    _check_rrf_k(k)
    if isinstance(rankings, str) or not isinstance(rankings, Sequence):
        raise _invalid("rankings must be a sequence of sequences")
    if len(rankings) > _MAX_RANKINGS:
        raise SearchError("input_too_large", "too many rankings")
    used = [1.0] * len(rankings) if weights is None else _check_weights(weights, len(rankings))
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, used, strict=True):
        if isinstance(ranking, str) or not isinstance(ranking, Sequence):
            raise _invalid("each ranking must be a sequence of ids")
        if len(ranking) > _MAX_RANKING_LENGTH:
            raise SearchError("input_too_large", "ranking is too long")
        seen: set[str] = set()
        for position, doc_id in enumerate(ranking, start=1):
            if not isinstance(doc_id, str):
                raise _invalid("ranking entries must be strings")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)
    return _rank(scores)


class EmbedderPort(Protocol):
    """Port to an embedding model.

    Implementations expose a stable ``model_id``, the vector ``dimension`` and an
    ``embed`` method returning one vector of ``dimension`` floats per input text.
    """

    model_id: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text."""


@dataclass(frozen=True)
class Document:
    """An indexable document.

    Attributes:
        id: Identifier of 1 to 256 characters.
        text: Content of at most 1,000,000 characters.
        metadata: Flat mapping of string keys to str, int, float (finite) or bool.
    """

    id: str
    text: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_id(self.id)
        if not isinstance(self.text, str):
            raise _invalid("text must be a string")
        if len(self.text) > _MAX_TEXT_CHARS:
            raise SearchError("input_too_large", "text exceeds 1000000 characters")
        metadata = self.metadata
        if not isinstance(metadata, Mapping):
            raise _invalid("metadata must be a mapping")
        if len(metadata) > _MAX_METADATA_ITEMS:
            raise SearchError("input_too_large", "metadata has too many entries")
        checked: dict[str, MetadataValue] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not 1 <= len(key) <= _MAX_ID_CHARS:
                raise _invalid("metadata keys must be strings of 1 to 256 characters")
            if isinstance(value, str):
                if len(value) > _MAX_METADATA_STRING:
                    raise SearchError("input_too_large", "metadata string is too long")
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise _invalid("metadata floats must be finite")
            elif not isinstance(value, int):
                raise _invalid("metadata values must be str, int, float or bool")
            checked[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(checked))


@dataclass(frozen=True)
class SearchHit:
    """One search result.

    ``score`` is the BM25 score in keyword mode, the cosine in vector mode and the
    fused RRF score in hybrid mode. Ranks are 1-based and ``None`` when the
    retriever did not return the document.
    """

    doc_id: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None
    document: Document


@dataclass(frozen=True, slots=True)
class _Entry:
    document: Document
    terms: dict[str, int]
    length: int
    vector: array[float]
    norm: float


def _same_value(actual: object, wanted: object) -> bool:
    if isinstance(actual, bool) != isinstance(wanted, bool):
        return False
    return bool(actual == wanted)


def _compile_filters(filters: Mapping[str, object] | None) -> list[_Filter]:
    if filters is None:
        return []
    if not isinstance(filters, Mapping):
        raise _invalid("filters must be a mapping")
    if len(filters) > _MAX_FILTER_KEYS:
        raise SearchError("input_too_large", "too many filter keys")
    compiled: list[_Filter] = []
    for key, value in filters.items():
        if not isinstance(key, str):
            raise _invalid("filter keys must be strings")
        options: tuple[object, ...]
        if isinstance(value, list | tuple | set | frozenset):
            if len(value) > _MAX_FILTER_VALUES:
                raise SearchError("input_too_large", "too many filter values")
            options = tuple(value)
        else:
            options = (value,)
        for option in options:
            if not isinstance(option, str | int | float):
                raise _invalid("filter values must be str, int, float or bool")
        compiled.append((key, options))
    return compiled


def _filter_matches(metadata: Mapping[str, MetadataValue], filters: Sequence[_Filter]) -> bool:
    for key, options in filters:
        if key not in metadata:
            return False
        actual = metadata[key]
        if not any(_same_value(actual, option) for option in options):
            return False
    return True


def _cosine(query: array[float], query_norm: float, entry: _Entry) -> float:
    denominator = query_norm * entry.norm
    if denominator == 0.0:
        return 0.0
    dot = float(sum(map(operator.mul, query, entry.vector)))
    return float(max(-1.0, min(1.0, dot / denominator)))


def _pack(vector: array[float]) -> bytes:
    if sys.byteorder == "big":
        swapped = array("d", vector)
        swapped.byteswap()
        return swapped.tobytes()
    return vector.tobytes()


def _unpack(blob: bytes) -> array[float]:
    vector: array[float] = array("d")
    vector.frombytes(blob)
    if sys.byteorder == "big":
        vector.byteswap()
    return vector


class HybridIndex:
    """In-memory hybrid index fusing BM25 and cosine rankings with RRF.

    Vector search is brute force and is documented for up to about 50,000
    documents. BM25 statistics (document count, document frequency, average
    length) always describe the whole corpus, also when filters are used.

    Mutations (``add``, ``remove``) are not thread-safe: serialize them and do
    not search concurrently with a mutation.
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        *,
        tokenizer: Callable[[str], list[str]] = default_tokenizer,
        stopwords: frozenset[str] = frozenset(),
        k1: float = 1.5,
        b: float = 0.75,
        max_documents: int = 100_000,
        max_query_chars: int = 10_000,
    ) -> None:
        model_id = getattr(embedder, "model_id", None)
        dimension = getattr(embedder, "dimension", None)
        if not isinstance(model_id, str) or not model_id:
            raise _invalid("embedder.model_id must be a non-empty string")
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise _invalid("embedder.dimension must be an integer")
        if not 1 <= dimension <= _MAX_DIMENSION:
            raise _invalid("embedder.dimension must be within 1 and 65536")
        _check_bm25(k1, b)
        _check_int_min(max_documents, "max_documents", 1)
        _check_int_min(max_query_chars, "max_query_chars", 1)
        self._embedder = embedder
        self._model_id = model_id
        self._dimension = dimension
        self._tokenizer = tokenizer
        self._stopwords = frozenset(stopwords)
        self._k1 = float(k1)
        self._b = float(b)
        self._max_documents = max_documents
        self._max_query_chars = max_query_chars
        self._entries: dict[str, _Entry] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in self._tokenizer(text) if token not in self._stopwords]

    def _to_vector(self, raw: Sequence[float]) -> array[float]:
        if isinstance(raw, str | bytes | bytearray):
            raise _invalid("embedding must be a sequence of real numbers")
        try:
            vector = array("d", raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid("embedding must be a sequence of real numbers") from exc
        if len(vector) != self._dimension:
            raise SearchError(
                "dimension_mismatch",
                f"embedding has {len(vector)} components, expected {self._dimension}",
            )
        if not all(abs(value) <= _MAX_COMPONENT for value in vector):
            raise _invalid("embedding components must be finite and at most 1e150")
        return vector

    def _embed_checked(self, texts: Sequence[str]) -> list[array[float]]:
        vectors: list[array[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = list(texts[start : start + _EMBED_BATCH])
            result = self._embedder.embed(batch)
            try:
                rows = list(result)
            except TypeError as exc:
                raise _invalid("embedder must return a list of vectors") from exc
            if len(rows) != len(batch):
                raise _invalid("embedder returned the wrong number of vectors")
            vectors.extend(self._to_vector(row) for row in rows)
        return vectors

    def _make_entry(self, document: Document, vector: array[float]) -> _Entry:
        tokens = self._tokenize(document.text)
        terms = dict(Counter(tokens))
        return _Entry(document, terms, len(tokens), vector, math.hypot(*vector))

    def _discard(self, doc_id: str) -> bool:
        entry = self._entries.pop(doc_id, None)
        if entry is None:
            return False
        for term in entry.terms:
            posting = self._postings[term]
            del posting[doc_id]
            if not posting:
                del self._postings[term]
        self._total_length -= entry.length
        return True

    def _insert(self, entry: _Entry) -> None:
        doc_id = entry.document.id
        self._discard(doc_id)
        self._entries[doc_id] = entry
        for term, count in entry.terms.items():
            self._postings.setdefault(term, {})[doc_id] = count
        self._total_length += entry.length

    def add(self, docs: Iterable[Document]) -> None:
        """Add documents; an existing id is replaced and the last duplicate wins.

        The call is all-or-nothing: nothing changes when it raises.

        Raises:
            SearchError: ``index_full`` when ``max_documents`` would be exceeded,
                ``dimension_mismatch`` for an embedding of the wrong size and
                ``invalid_argument`` for anything that is not a Document.
        """
        items = list(islice(docs, self._max_documents + 1))
        if len(items) > self._max_documents:
            raise SearchError("index_full", "too many documents for max_documents")
        unique: dict[str, Document] = {}
        for item in items:
            if not isinstance(item, Document):
                raise _invalid("add() expects Document instances")
            unique[item.id] = item
        new_ids = sum(1 for doc_id in unique if doc_id not in self._entries)
        if len(self._entries) + new_ids > self._max_documents:
            raise SearchError("index_full", "index would exceed max_documents")
        documents = list(unique.values())
        vectors = self._embed_checked([document.text for document in documents])
        entries = [self._make_entry(d, v) for d, v in zip(documents, vectors, strict=True)]
        for entry in entries:
            self._insert(entry)
        logger.debug("added %d documents, index size %d", len(entries), len(self._entries))

    def remove(self, doc_id: str) -> bool:
        """Remove a document; return ``True`` if it existed."""
        _check_id(doc_id)
        return self._discard(doc_id)

    def _allowed_ids(self, compiled: Sequence[_Filter]) -> set[str] | None:
        if not compiled:
            return None
        allowed: set[str] = set()
        for doc_id, entry in self._entries.items():
            if _filter_matches(entry.document.metadata, compiled):
                allowed.add(doc_id)
        return allowed

    def _keyword_ranking(self, query: str, allowed: set[str] | None) -> list[tuple[str, float]]:
        total = len(self._entries)
        avgdl = self._total_length / total
        scores: dict[str, float] = {}
        for term in dict.fromkeys(self._tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                length = self._entries[doc_id].length
                norm = 1.0 - self._b + self._b * length / avgdl
                weight = tf * (self._k1 + 1.0) / (tf + self._k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * weight
        return _rank(scores)

    def _vector_ranking(self, query: str, allowed: set[str] | None) -> list[tuple[str, float]]:
        query_vector = self._embed_checked([query])[0]
        query_norm = math.hypot(*query_vector)
        scores: dict[str, float] = {}
        for doc_id, entry in self._entries.items():
            if allowed is not None and doc_id not in allowed:
                continue
            scores[doc_id] = _cosine(query_vector, query_norm, entry)
        return _rank(scores)

    def _hit(
        self,
        doc_id: str,
        score: float,
        keyword_rank: int | None,
        vector_rank: int | None,
    ) -> SearchHit:
        document = self._entries[doc_id].document
        return SearchHit(doc_id, score, keyword_rank, vector_rank, document)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        filters: Mapping[str, object] | None = None,
        rrf_k: float = 60,
        weights: Sequence[float] = (1.0, 1.0),
        candidates: int = 50,
    ) -> list[SearchHit]:
        """Search the index.

        Args:
            query: Query text of at most ``max_query_chars`` characters.
            k: Number of hits to return (>= 1).
            mode: ``keyword`` (BM25), ``vector`` (cosine) or ``hybrid`` (RRF).
            filters: Metadata filters applied before ranking. A scalar means
                equality, a list means membership, an unknown key never matches.
            rrf_k: RRF constant (hybrid mode).
            weights: ``(keyword_weight, vector_weight)`` for RRF.
            candidates: Top results taken from each retriever before fusion.

        Ties always sort by document id ascending.

        Raises:
            SearchError: ``invalid_argument``, ``input_too_large`` or
                ``dimension_mismatch`` (query embedding of the wrong size).
        """
        if mode not in _MODES:
            raise _invalid("mode must be keyword, vector or hybrid")
        _check_int_min(k, "k", 1)
        _check_int_min(candidates, "candidates", 1)
        _check_rrf_k(rrf_k)
        weight_list = _check_weights(weights, 2)
        if not isinstance(query, str):
            raise _invalid("query must be a string")
        if len(query) > self._max_query_chars:
            raise SearchError("input_too_large", "query exceeds max_query_chars")
        allowed = self._allowed_ids(_compile_filters(filters))
        if not self._entries or (allowed is not None and not allowed):
            return []
        logger.debug("search mode=%s k=%d", mode, k)
        if mode == "keyword":
            ranked = self._keyword_ranking(query, allowed)[:k]
            return [self._hit(d, s, r, None) for r, (d, s) in enumerate(ranked, start=1)]
        if mode == "vector":
            ranked = self._vector_ranking(query, allowed)[:k]
            return [self._hit(d, s, None, r) for r, (d, s) in enumerate(ranked, start=1)]
        keyword = self._keyword_ranking(query, allowed)[:candidates]
        vector = self._vector_ranking(query, allowed)[:candidates]
        keyword_rank = {d: r for r, (d, _) in enumerate(keyword, start=1)}
        vector_rank = {d: r for r, (d, _) in enumerate(vector, start=1)}
        rankings = [list(keyword_rank), list(vector_rank)]
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weight_list)[:k]
        return [self._hit(d, s, keyword_rank.get(d), vector_rank.get(d)) for d, s in fused]

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the index to a SQLite file (no pickle; vectors are float64 BLOBs).

        The file is written to a temporary sibling and atomically renamed.

        Raises:
            SearchError: ``invalid_argument`` when the file cannot be written.
        """
        target = Path(path)
        temporary = target.with_name(target.name + ".tmp")
        meta = [
            ("schema", _SCHEMA),
            ("model_id", self._model_id),
            ("dimension", str(self._dimension)),
            ("k1", repr(self._k1)),
            ("b", repr(self._b)),
        ]
        rows = []
        for doc_id in sorted(self._entries):
            entry = self._entries[doc_id]
            metadata_json = json.dumps(dict(entry.document.metadata), sort_keys=True)
            rows.append((doc_id, entry.document.text, metadata_json, _pack(entry.vector)))
        try:
            temporary.unlink(missing_ok=True)
            with closing(sqlite3.connect(temporary)) as conn:
                conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute(
                    "CREATE TABLE docs (id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                    "metadata_json TEXT NOT NULL, vector BLOB NOT NULL)"
                )
                conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)
                conn.executemany(
                    "INSERT INTO docs (id, text, metadata_json, vector) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            os.replace(temporary, target)
        except (sqlite3.Error, OSError, UnicodeError) as exc:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise _invalid("could not write the index file") from exc

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        embedder: EmbedderPort,
        *,
        tokenizer: Callable[[str], list[str]] = default_tokenizer,
        stopwords: frozenset[str] = frozenset(),
        max_documents: int = 100_000,
        max_query_chars: int = 10_000,
    ) -> HybridIndex:
        """Load an index written by :meth:`save`.

        The tokenizer and stopwords are not stored in the file: pass the same ones
        that were used when the index was built. Every row is validated.

        Raises:
            SearchError: ``corrupt_file`` for malformed content or an unknown
                schema, ``index_mismatch`` when the embedder's ``model_id`` or
                ``dimension`` differs from the stored ones, ``index_full`` when
                the file holds more than ``max_documents`` documents and
                ``invalid_argument`` when the file does not exist.
        """
        file = Path(path)
        if not file.is_file():
            raise _invalid("index file not found")
        uri = f"{file.resolve().as_uri()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                return cls._read(
                    conn, embedder, tokenizer, stopwords, max_documents, max_query_chars
                )
        except sqlite3.Error as exc:
            raise _corrupt("index file is not a valid SQLite index") from exc

    @classmethod
    def _read(
        cls,
        conn: sqlite3.Connection,
        embedder: EmbedderPort,
        tokenizer: Callable[[str], list[str]],
        stopwords: frozenset[str],
        max_documents: int,
        max_query_chars: int,
    ) -> HybridIndex:
        meta_rows = conn.execute(
            "SELECT key, value FROM meta LIMIT ?", (_MAX_META_ROWS + 1,)
        ).fetchall()
        if len(meta_rows) > _MAX_META_ROWS:
            raise _corrupt("meta table is too large")
        meta: dict[str, str] = {}
        for key, value in meta_rows:
            if not isinstance(key, str) or not isinstance(value, str):
                raise _corrupt("meta table holds a non-text value")
            meta[key] = value
        if meta.get("schema") != _SCHEMA:
            raise _corrupt("unsupported schema version")
        try:
            model_id = meta["model_id"]
            dimension = int(meta["dimension"])
            k1 = float(meta["k1"])
            b = float(meta["b"])
            _check_bm25(k1, b)
            if not 1 <= dimension <= _MAX_DIMENSION:
                raise _invalid("dimension out of range")
        except (KeyError, ValueError) as exc:
            raise _corrupt("meta table is incomplete or malformed") from exc
        index = cls(
            embedder,
            tokenizer=tokenizer,
            stopwords=stopwords,
            k1=k1,
            b=b,
            max_documents=max_documents,
            max_query_chars=max_query_chars,
        )
        if model_id != index._model_id:
            raise SearchError("index_mismatch", "index was built with a different model_id")
        if dimension != index._dimension:
            raise SearchError("index_mismatch", "index was built with a different dimension")
        cursor = conn.execute("SELECT id, text, metadata_json, vector FROM docs ORDER BY id")
        for count, (doc_id, text, metadata_json, blob) in enumerate(cursor, start=1):
            if count > max_documents:
                raise SearchError("index_full", "index file exceeds max_documents")
            if not (
                isinstance(doc_id, str)
                and isinstance(text, str)
                and isinstance(metadata_json, str)
                and isinstance(blob, bytes)
            ):
                raise _corrupt("docs row has an unexpected column type")
            if len(blob) != 8 * dimension:
                raise _corrupt("vector BLOB has the wrong length")
            vector = _unpack(blob)
            if not all(abs(value) <= _MAX_COMPONENT for value in vector):
                raise _corrupt("vector holds a non-finite or oversized component")
            try:
                parsed = json.loads(metadata_json)
            except (ValueError, RecursionError) as exc:
                raise _corrupt("metadata is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise _corrupt("metadata is not a JSON object")
            try:
                document = Document(doc_id, text, parsed)
            except SearchError as exc:
                raise _corrupt("document row is invalid") from exc
            index._insert(index._make_entry(document, vector))
        return index
