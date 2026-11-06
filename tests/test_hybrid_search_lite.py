"""Tests for hybrid_search_lite."""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import hybrid_search_lite
from hybrid_search_lite import (
    Document,
    HybridIndex,
    SearchError,
    SearchHit,
    default_tokenizer,
    reciprocal_rank_fusion,
)

VOCAB = ("cat", "dog", "fish", "bird")
MODES = ("keyword", "vector", "hybrid")


class FakeEmbedder:
    """Bag-of-words embedder over a four word vocabulary."""

    def __init__(self, model_id: str = "fake-1", dimension: int = 4) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.wrong_dimension = False
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            tokens = default_tokenizer(text)
            row = [float(tokens.count(word)) for word in VOCAB]
            rows.append(row[:-1] if self.wrong_dimension else row)
        return rows


class ScriptedEmbedder:
    """Embedder returning a fixed, possibly malformed, result."""

    model_id = "scripted"
    dimension = 2

    def __init__(self, result: object) -> None:
        self.result = result

    def embed(self, texts: Sequence[str]) -> Any:
        return self.result


def basic_docs() -> list[Document]:
    return [Document("d1", "cat sat"), Document("d2", "cat cat dog"), Document("d3", "dog")]


def basic_index(**kwargs: Any) -> HybridIndex:
    index = HybridIndex(FakeEmbedder(), **kwargs)
    index.add(basic_docs())
    return index


def ids(hits: list[SearchHit]) -> list[str]:
    return [hit.doc_id for hit in hits]


def summarize(hits: list[SearchHit]) -> list[tuple[Any, ...]]:
    rows = []
    for hit in hits:
        meta = dict(hit.document.metadata)
        rows.append((hit.doc_id, hit.score, hit.keyword_rank, hit.vector_rank, meta))
    return rows


def metadata_index() -> HybridIndex:
    index = HybridIndex(FakeEmbedder())
    docs = [
        Document("m1", "cat fish", {"n": 1, "f": 0.1, "b": True}),
        Document("m2", "dog bird"),
        Document("m3", "cat dog bird", {"lang": "en"}),
    ]
    index.add(docs)
    return index


def run_sql(path: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def assert_fused(got: list[tuple[str, float]], expected: list[tuple[str, float]]) -> None:
    assert [doc_id for doc_id, _ in got] == [doc_id for doc_id, _ in expected]
    for (_, score), (_, wanted) in zip(got, expected, strict=True):
        assert abs(score - wanted) < 1e-12


def test_version() -> None:
    assert hybrid_search_lite.__version__ == "0.1.0"


# --- errors -----------------------------------------------------------------


def test_search_error_is_value_error_with_code_and_short_message() -> None:
    error = SearchError("invalid_argument", "x" * 500)
    assert isinstance(error, ValueError)
    assert error.code == "invalid_argument"
    assert len(str(error)) == 200
    assert str(SearchError("index_full", "short")) == "short"


# --- tokenizer --------------------------------------------------------------


def test_default_tokenizer_acceptance_literal() -> None:
    assert default_tokenizer("ABC-123, \ufb01le") == ["abc", "123", "file"]


def test_default_tokenizer_adversarial_neighbours() -> None:
    assert default_tokenizer("abc123def") == ["abc123def"]
    assert default_tokenizer("snake_case") == ["snake", "case"]
    assert default_tokenizer("cat,dog;fish") == ["cat", "dog", "fish"]
    assert default_tokenizer("\u00c9COLE") == ["\u00e9cole"]
    assert default_tokenizer("Stra\u00dfe") == ["strasse"]
    assert default_tokenizer("") == []
    assert default_tokenizer(" -- ") == []


def test_default_tokenizer_limits() -> None:
    with pytest.raises(SearchError) as too_big:
        default_tokenizer("a" * 1_000_001)
    assert too_big.value.code == "input_too_large"
    with pytest.raises(SearchError) as not_text:
        default_tokenizer(5)  # type: ignore[arg-type]
    assert not_text.value.code == "invalid_argument"


# --- documents --------------------------------------------------------------

DOC_CASES: list[tuple[dict[str, Any], str]] = [
    ({"id": "", "text": "x"}, "invalid_argument"),
    ({"id": "x" * 257, "text": "x"}, "invalid_argument"),
    ({"id": 5, "text": "x"}, "invalid_argument"),
    ({"id": "a", "text": 5}, "invalid_argument"),
    ({"id": "a", "text": "x" * 1_000_001}, "input_too_large"),
    ({"id": "a", "text": "x", "metadata": {"f": math.nan}}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": {"f": math.inf}}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": {"v": [1]}}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": {1: "x"}}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": {"": "x"}}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": [("a", 1)]}, "invalid_argument"),
    ({"id": "a", "text": "x", "metadata": {"s": "x" * 10_001}}, "input_too_large"),
    ({"id": "a", "text": "x", "metadata": {f"k{i}": i for i in range(101)}}, "input_too_large"),
]


@pytest.mark.parametrize(("kwargs", "code"), DOC_CASES)
def test_document_validation(kwargs: dict[str, Any], code: str) -> None:
    with pytest.raises(SearchError) as info:
        Document(**kwargs)
    assert info.value.code == code


def test_document_boundaries_and_immutability() -> None:
    assert Document("x" * 256, "y" * 1_000_000).id == "x" * 256
    source = {"n": 1}
    doc = Document("a", "t", source)
    source["n"] = 2
    assert doc.metadata["n"] == 1
    with pytest.raises(TypeError):
        doc.metadata["x"] = 1  # type: ignore[index]
    assert Document("a", "t").metadata == {}


# --- keyword search ---------------------------------------------------------


def test_keyword_bm25_acceptance() -> None:
    index = basic_index()
    hits = index.search("cat", mode="keyword")
    assert ids(hits) == ["d2", "d1"]
    assert abs(hits[0].score - math.log(1.6) * 5 / 4.0625) < 1e-9
    assert abs(hits[1].score - math.log(1.6)) < 1e-9
    assert (hits[0].keyword_rank, hits[0].vector_rank) == (1, None)
    assert hits[1].keyword_rank == 2


def test_keyword_query_terms_are_deduplicated() -> None:
    index = basic_index()
    once = index.search("cat", mode="keyword")
    twice = index.search("cat cat", mode="keyword")
    assert [h.score for h in once] == [h.score for h in twice]


def test_keyword_multi_term_and_adjacent_tokens() -> None:
    index = HybridIndex(FakeEmbedder())
    index.add([Document("p1", "cat-dog"), Document("p2", "cat123"), Document("p3", "dog")])
    assert ids(index.search("cat", mode="keyword")) == ["p1"]
    assert ids(index.search("cat123", mode="keyword")) == ["p2"]
    assert ids(index.search("cat dog", mode="keyword"))[0] == "p1"


def test_keyword_ties_sort_by_id() -> None:
    index = HybridIndex(FakeEmbedder())
    index.add([Document("b", "cat"), Document("a", "cat")])
    assert ids(index.search("cat", mode="keyword")) == ["a", "b"]


def test_keyword_mode_does_not_embed_the_query() -> None:
    embedder = FakeEmbedder()
    index = HybridIndex(embedder)
    index.add(basic_docs())
    before = len(embedder.calls)
    index.search("cat", mode="keyword")
    assert len(embedder.calls) == before


def test_custom_tokenizer_and_parameters() -> None:
    index = HybridIndex(FakeEmbedder(), tokenizer=str.split, k1=0.0, b=0.0)
    index.add([Document("a", "Cat cat"), Document("b", "cat")])
    hits = index.search("cat", mode="keyword")
    assert ids(hits) == ["b"]
    assert hits[0].score == pytest.approx(math.log(1 + 1.5 / 1.5))


# --- vector search ----------------------------------------------------------


def test_vector_ranking_by_cosine_with_id_tiebreak() -> None:
    index = HybridIndex(FakeEmbedder())
    docs = [
        Document("c", "cat dog"),
        Document("b", "dog"),
        Document("a2", "cat"),
        Document("a1", "cat"),
    ]
    index.add(docs)
    hits = index.search("cat", mode="vector")
    assert ids(hits) == ["a1", "a2", "c", "b"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[2].score == pytest.approx(1 / math.sqrt(2))
    assert hits[3].score == pytest.approx(0.0)
    assert [h.vector_rank for h in hits] == [1, 2, 3, 4]
    assert all(h.keyword_rank is None for h in hits)


def test_vector_zero_query_and_zero_documents() -> None:
    index = HybridIndex(FakeEmbedder())
    index.add([Document("b", "cat"), Document("a", ""), Document("c", "dog")])
    hits = index.search("nothing known", mode="vector")
    assert ids(hits) == ["a", "b", "c"]
    assert all(h.score == 0.0 for h in hits)


# --- reciprocal rank fusion -------------------------------------------------


def test_rrf_acceptance_examples() -> None:
    got = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert_fused(got, [("b", 1 / 62 + 1 / 61), ("a", 1 / 61), ("c", 1 / 62)])
    weighted = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60, weights=[2.0, 1.0])
    assert_fused(weighted, [("b", 2 / 62 + 1 / 61), ("a", 2 / 61), ("c", 1 / 62)])


def test_rrf_edge_cases() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    tie = reciprocal_rank_fusion([["b"], ["a"]])
    assert [doc_id for doc_id, _ in tie] == ["a", "b"]
    repeated = reciprocal_rank_fusion([["a", "a", "b"]], k=0)
    assert_fused(repeated, [("a", 1.0), ("b", 1 / 3)])


RRF_BAD: list[dict[str, Any]] = [
    {"rankings": [["a"]], "k": -1},
    {"rankings": [["a"]], "k": math.inf},
    {"rankings": [["a"]], "weights": [1.0, 1.0]},
    {"rankings": [["a"]], "weights": [-1.0]},
    {"rankings": [["a"]], "weights": [math.nan]},
    {"rankings": [["a"]], "weights": "1"},
    {"rankings": "ab"},
    {"rankings": ["ab"]},
    {"rankings": [[1]]},
    {"rankings": [["a"], 5]},
    {"rankings": 5},
]


@pytest.mark.parametrize("kwargs", RRF_BAD)
def test_rrf_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SearchError) as info:
        reciprocal_rank_fusion(**kwargs)
    assert info.value.code == "invalid_argument"


def test_rrf_size_limits() -> None:
    with pytest.raises(SearchError) as many:
        reciprocal_rank_fusion([["a"]] * 65)
    assert many.value.code == "input_too_large"
    with pytest.raises(SearchError) as long:
        reciprocal_rank_fusion([["x"] * 100_001])
    assert long.value.code == "input_too_large"


# --- hybrid search ----------------------------------------------------------


def test_hybrid_fuses_both_rankings() -> None:
    index = basic_index()
    hits = index.search("cat")
    assert ids(hits) == ["d1", "d2", "d3"]
    by_id = {hit.doc_id: hit for hit in hits}
    assert (by_id["d1"].keyword_rank, by_id["d1"].vector_rank) == (2, 1)
    assert (by_id["d2"].keyword_rank, by_id["d2"].vector_rank) == (1, 2)
    assert (by_id["d3"].keyword_rank, by_id["d3"].vector_rank) == (None, 3)
    assert by_id["d1"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["d3"].score == pytest.approx(1 / 63)


def test_hybrid_weights_k_and_candidates() -> None:
    index = basic_index()
    assert ids(index.search("cat", weights=(5.0, 0.0))) == ["d2", "d1", "d3"]
    assert ids(index.search("cat", k=1)) == ["d1"]
    assert len(index.search("cat", k=100)) == 3
    narrow = index.search("cat", candidates=1)
    assert ids(narrow) == ["d1", "d2"]
    assert (narrow[1].keyword_rank, narrow[1].vector_rank) == (1, None)
    custom = index.search("cat", rrf_k=0)
    assert custom[0].score == pytest.approx(1.0 + 1 / 2)


def test_stopword_only_query() -> None:
    index = HybridIndex(FakeEmbedder(), stopwords=frozenset({"cat"}))
    index.add([Document("c1", "cat"), Document("c2", "cat dog"), Document("c3", "dog")])
    assert index.search("cat", mode="keyword") == []
    vector_ids = ids(index.search("cat", mode="vector"))
    assert vector_ids == ["c1", "c2", "c3"]
    hybrid = index.search("cat", mode="hybrid")
    assert ids(hybrid) == vector_ids
    assert all(hit.keyword_rank is None for hit in hybrid)
    assert ids(index.search("dog", mode="keyword")) == ["c2", "c3"]


# --- filters ----------------------------------------------------------------


def filter_index() -> HybridIndex:
    index = HybridIndex(FakeEmbedder())
    docs = [
        Document("e1", "cat one", {"lang": "en", "tag": "x", "n": 1, "flag": True}),
        Document("e2", "cat two", {"lang": "en", "tag": "y"}),
        Document("e3", "cat three", {"lang": "en", "tag": "z"}),
        Document("f1", "cat four", {"lang": "fr", "tag": "x"}),
        Document("g1", "cat five"),
    ]
    index.add(docs)
    return index


@pytest.mark.parametrize("mode", MODES)
def test_filters_scalar_and_membership(mode: Any) -> None:
    index = filter_index()
    hits = index.search("cat", mode=mode, filters={"lang": "en", "tag": ["x", "y"]})
    assert sorted(ids(hits)) == ["e1", "e2"]


@pytest.mark.parametrize("mode", MODES)
def test_filter_matching_nothing_returns_empty(mode: Any) -> None:
    index = filter_index()
    assert index.search("cat", mode=mode, filters={"lang": "de"}) == []
    assert index.search("cat", mode=mode, filters={"unknown": "en"}) == []
    assert index.search("cat", mode=mode, filters={"tag": []}) == []


def test_filter_value_types_are_strict() -> None:
    index = filter_index()
    assert ids(index.search("cat", filters={"n": True})) == []
    assert ids(index.search("cat", filters={"flag": 1})) == []
    assert ids(index.search("cat", filters={"n": 1})) == ["e1"]
    assert ids(index.search("cat", filters={"n": 1.0})) == ["e1"]
    assert ids(index.search("cat", filters={"n": [1, 2]})) == ["e1"]
    assert ids(index.search("cat", filters={"flag": True})) == ["e1"]
    assert ids(index.search("cat", filters={"lang": ("fr",)})) == ["f1"]
    assert len(index.search("cat", filters={})) == 5


def test_filter_validation() -> None:
    index = filter_index()
    for bad in ({"a": {}}, {1: "x"}, {"a": [None]}, {"a": None}):
        with pytest.raises(SearchError) as info:
            index.search("cat", filters=bad)  # type: ignore[arg-type]
        assert info.value.code == "invalid_argument"
    with pytest.raises(SearchError) as not_mapping:
        index.search("cat", filters=[("a", 1)])  # type: ignore[arg-type]
    assert not_mapping.value.code == "invalid_argument"
    with pytest.raises(SearchError) as many_values:
        index.search("cat", filters={"a": list(range(1001))})
    assert many_values.value.code == "input_too_large"
    with pytest.raises(SearchError) as many_keys:
        index.search("cat", filters={f"k{i}": 1 for i in range(101)})
    assert many_keys.value.code == "input_too_large"


# --- remove and replace -----------------------------------------------------


def test_removed_document_never_appears() -> None:
    index = basic_index()
    assert index.remove("d2") is True
    assert index.remove("d2") is False
    assert len(index) == 2
    for mode in MODES:
        assert "d2" not in ids(index.search("cat dog", mode=mode))
    assert ids(index.search("dog", mode="keyword")) == ["d3"]


def test_remove_validates_id() -> None:
    index = basic_index()
    with pytest.raises(SearchError) as info:
        index.remove("")
    assert info.value.code == "invalid_argument"


def test_adding_existing_id_replaces_document() -> None:
    index = basic_index()
    index.add([Document("d2", "fish")])
    assert len(index) == 3
    assert ids(index.search("cat", mode="keyword")) == ["d1"]
    assert ids(index.search("fish", mode="keyword")) == ["d2"]
    assert ids(index.search("dog", mode="keyword")) == ["d3"]


def test_duplicate_ids_in_one_call_last_wins() -> None:
    index = HybridIndex(FakeEmbedder())
    index.add([Document("x", "cat"), Document("x", "dog")])
    assert len(index) == 1
    assert ids(index.search("dog", mode="keyword")) == ["x"]
    assert index.search("cat", mode="keyword") == []


def test_add_accepts_generators_and_rejects_non_documents() -> None:
    index = HybridIndex(FakeEmbedder())
    index.add(Document(f"g{i}", "cat") for i in range(3))
    assert len(index) == 3
    with pytest.raises(SearchError) as info:
        index.add(["nope"])  # type: ignore[list-item]
    assert info.value.code == "invalid_argument"


def test_add_embeds_in_batches() -> None:
    embedder = FakeEmbedder()
    index = HybridIndex(embedder)
    index.add([Document(f"id{i}", "cat") for i in range(130)])
    assert [len(call) for call in embedder.calls] == [64, 64, 2]
    assert len(index) == 130


# --- edge cases -------------------------------------------------------------


def test_empty_index_returns_nothing_without_embedding() -> None:
    embedder = FakeEmbedder()
    index = HybridIndex(embedder)
    for mode in MODES:
        assert index.search("cat", mode=mode) == []
    assert embedder.calls == []


def test_empty_text_document() -> None:
    index = basic_index()
    index.add([Document("d4", "")])
    assert "d4" not in ids(index.search("cat", mode="keyword"))
    assert ids(index.search("cat", mode="vector")) == ["d1", "d2", "d3", "d4"]


def test_k_larger_than_corpus() -> None:
    index = basic_index()
    assert len(index.search("cat", mode="vector", k=1000)) == 3


# --- argument validation ----------------------------------------------------

SEARCH_BAD: list[dict[str, Any]] = [
    {"k": 0},
    {"k": True},
    {"k": 1.5},
    {"mode": "bogus"},
    {"candidates": 0},
    {"rrf_k": -1},
    {"rrf_k": True},
    {"weights": (1.0,)},
    {"weights": (1.0, -1.0)},
    {"weights": (1.0, math.nan)},
    {"weights": 5},
    {"weights": "ab"},
]


@pytest.mark.parametrize("kwargs", SEARCH_BAD)
def test_search_invalid_arguments(kwargs: dict[str, Any]) -> None:
    index = basic_index()
    with pytest.raises(SearchError) as info:
        index.search("cat", **kwargs)
    assert info.value.code == "invalid_argument"


def test_search_rejects_non_string_query() -> None:
    with pytest.raises(SearchError) as info:
        basic_index().search(5)  # type: ignore[arg-type]
    assert info.value.code == "invalid_argument"


CONSTRUCTOR_BAD: list[dict[str, Any]] = [
    {"k1": -1.0},
    {"k1": math.nan},
    {"b": 1.5},
    {"b": "x"},
    {"max_documents": 0},
    {"max_query_chars": 0},
]


@pytest.mark.parametrize("kwargs", CONSTRUCTOR_BAD)
def test_constructor_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SearchError) as info:
        HybridIndex(FakeEmbedder(), **kwargs)
    assert info.value.code == "invalid_argument"


def test_constructor_rejects_bad_embedders() -> None:
    for embedder in (FakeEmbedder(model_id=""), FakeEmbedder(dimension=0), object()):
        with pytest.raises(SearchError) as info:
            HybridIndex(embedder)  # type: ignore[arg-type]
        assert info.value.code == "invalid_argument"
    bad = FakeEmbedder()
    bad.dimension = True  # type: ignore[assignment]
    with pytest.raises(SearchError):
        HybridIndex(bad)


# --- limits and embedder faults ---------------------------------------------


def test_dimension_mismatch_on_add_leaves_index_untouched() -> None:
    embedder = FakeEmbedder()
    index = HybridIndex(embedder)
    embedder.wrong_dimension = True
    with pytest.raises(SearchError) as info:
        index.add([Document("a", "cat")])
    assert info.value.code == "dimension_mismatch"
    assert len(index) == 0


def test_dimension_mismatch_on_query() -> None:
    embedder = FakeEmbedder()
    index = HybridIndex(embedder)
    index.add(basic_docs())
    embedder.wrong_dimension = True
    with pytest.raises(SearchError) as info:
        index.search("cat", mode="vector")
    assert info.value.code == "dimension_mismatch"


def test_query_over_max_query_chars() -> None:
    index = basic_index(max_query_chars=5)
    assert ids(index.search("abcde", mode="keyword")) == []
    with pytest.raises(SearchError) as info:
        index.search("abcdef")
    assert info.value.code == "input_too_large"


def test_adding_beyond_max_documents() -> None:
    index = HybridIndex(FakeEmbedder(), max_documents=2)
    index.add([Document("a", "cat"), Document("b", "dog")])
    with pytest.raises(SearchError) as info:
        index.add([Document("c", "fish")])
    assert info.value.code == "index_full"
    index.add([Document("a", "fish")])
    assert len(index) == 2
    three = [Document("x", "cat"), Document("y", "cat"), Document("z", "cat")]
    with pytest.raises(SearchError) as many:
        HybridIndex(FakeEmbedder(), max_documents=2).add(three)
    assert many.value.code == "index_full"


BAD_EMBEDDINGS: list[object] = [
    None,
    [],
    [[1.0, 2.0], [3.0, 4.0]],
    [["a", "b"]],
    ["ab"],
    [[math.nan, 1.0]],
    [[math.inf, 1.0]],
    [[1e151, 1.0]],
    [[10**400, 1.0]],
    [5],
]


@pytest.mark.parametrize("result", BAD_EMBEDDINGS)
def test_malformed_embedder_output(result: object) -> None:
    index = HybridIndex(ScriptedEmbedder(result))  # type: ignore[arg-type]
    with pytest.raises(SearchError) as info:
        index.add([Document("a", "cat")])
    assert info.value.code == "invalid_argument"
    assert len(index) == 0


def test_embedder_accepts_integer_components() -> None:
    index = HybridIndex(ScriptedEmbedder([[1, 0]]))  # type: ignore[arg-type]
    index.add([Document("a", "cat")])
    assert ids(index.search("cat", mode="vector")) == ["a"]


# --- persistence ------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    index = metadata_index()
    path = tmp_path / "index.sqlite"
    index.save(path)
    assert not (tmp_path / "index.sqlite.tmp").exists()
    loaded = HybridIndex.load(path, FakeEmbedder())
    assert len(loaded) == 3
    for mode in MODES:
        for query in ("cat", "dog bird", "fish", "nothing"):
            assert summarize(index.search(query, mode=mode)) == summarize(
                loaded.search(query, mode=mode)
            )
    filters = {"lang": "en"}
    assert summarize(index.search("cat", filters=filters)) == summarize(
        loaded.search("cat", filters=filters)
    )


def test_metadata_types_survive_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(str(path))
    loaded = HybridIndex.load(path, FakeEmbedder())
    meta = loaded.search("fish", mode="keyword")[0].document.metadata
    assert dict(meta) == {"n": 1, "f": 0.1, "b": True}
    assert type(meta["b"]) is bool
    assert type(meta["n"]) is int
    assert type(meta["f"]) is float


def test_save_overwrites_and_empty_index_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    HybridIndex(FakeEmbedder()).save(path)
    loaded = HybridIndex.load(path, FakeEmbedder())
    assert len(loaded) == 0
    assert loaded.search("cat") == []


def test_load_keeps_k1_and_b(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    index = basic_index(k1=1.2, b=0.5)
    index.save(path)
    loaded = HybridIndex.load(path, FakeEmbedder())
    assert summarize(loaded.search("cat", mode="keyword")) == summarize(
        index.search("cat", mode="keyword")
    )


def test_load_passes_tokenizer_options(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    basic_index().save(path)
    loaded = HybridIndex.load(path, FakeEmbedder(), stopwords=frozenset({"cat"}))
    assert loaded.search("cat", mode="keyword") == []
    small = HybridIndex.load(path, FakeEmbedder(), max_query_chars=2)
    with pytest.raises(SearchError):
        small.search("cat")


def test_load_with_another_model_id(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    with pytest.raises(SearchError) as info:
        HybridIndex.load(path, FakeEmbedder(model_id="other"))
    assert info.value.code == "index_mismatch"
    with pytest.raises(SearchError) as dim:
        HybridIndex.load(path, FakeEmbedder(dimension=5))
    assert dim.value.code == "index_mismatch"


def test_load_schema_version_99(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    run_sql(path, "UPDATE meta SET value = '99' WHERE key = 'schema'")
    with pytest.raises(SearchError) as info:
        HybridIndex.load(path, FakeEmbedder())
    assert info.value.code == "corrupt_file"


CORRUPTIONS: list[tuple[str, tuple[object, ...]]] = [
    ("DELETE FROM meta WHERE key = 'model_id'", ()),
    ("DELETE FROM meta WHERE key = 'schema'", ()),
    ("UPDATE meta SET value = 'abc' WHERE key = 'k1'", ()),
    ("UPDATE meta SET value = 'nan' WHERE key = 'k1'", ()),
    ("UPDATE meta SET value = '2' WHERE key = 'b'", ()),
    ("UPDATE meta SET value = 'x' WHERE key = 'dimension'", ()),
    ("UPDATE meta SET value = '0' WHERE key = 'dimension'", ()),
    ("UPDATE docs SET vector = x'00'", ()),
    ("UPDATE docs SET vector = 'text'", ()),
    ("UPDATE docs SET metadata_json = '[1]'", ()),
    ("UPDATE docs SET metadata_json = '{bad'", ()),
    ("UPDATE docs SET metadata_json = ?", ('{"a": null}',)),
    ("UPDATE docs SET metadata_json = ?", ('{"a": 1e999}',)),
    ("UPDATE docs SET metadata_json = ?", ("__import__('os').system('true')",)),
    ("UPDATE docs SET metadata_json = ?", ("[" * 100_000,)),
    ("UPDATE docs SET id = '' WHERE id = 'm2'", ()),
    ("UPDATE docs SET vector = ? WHERE id = 'm2'", (struct.pack("<4d", math.nan, 0, 0, 0),)),
    ("DROP TABLE docs", ()),
]


@pytest.mark.parametrize(("sql", "params"), CORRUPTIONS)
def test_load_rejects_corrupt_files(
    tmp_path: Path, sql: str, params: tuple[object, ...]
) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    run_sql(path, sql, params)
    with pytest.raises(SearchError) as info:
        HybridIndex.load(path, FakeEmbedder())
    assert info.value.code == "corrupt_file"


def test_load_rejects_non_sqlite_and_empty_files(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.sqlite"
    garbage.write_bytes(b"this is not a sqlite database" * 50)
    with pytest.raises(SearchError) as info:
        HybridIndex.load(garbage, FakeEmbedder())
    assert info.value.code == "corrupt_file"
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")
    with pytest.raises(SearchError) as no_tables:
        HybridIndex.load(empty, FakeEmbedder())
    assert no_tables.value.code == "corrupt_file"


def test_load_rejects_oversized_meta_table(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    for i in range(40):
        run_sql(path, "INSERT INTO meta (key, value) VALUES (?, 'x')", (f"extra{i}",))
    with pytest.raises(SearchError) as info:
        HybridIndex.load(path, FakeEmbedder())
    assert info.value.code == "corrupt_file"


def test_load_rejects_non_text_meta_value(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    run_sql(path, "UPDATE meta SET value = 5 WHERE key = 'k1'")
    run_sql(path, "INSERT INTO meta (key, value) VALUES ('blob', x'00')")
    with pytest.raises(SearchError) as info:
        HybridIndex.load(path, FakeEmbedder())
    assert info.value.code == "corrupt_file"


def test_load_missing_file_and_max_documents(tmp_path: Path) -> None:
    with pytest.raises(SearchError) as missing:
        HybridIndex.load(tmp_path / "nope.sqlite", FakeEmbedder())
    assert missing.value.code == "invalid_argument"
    path = tmp_path / "index.sqlite"
    metadata_index().save(path)
    with pytest.raises(SearchError) as full:
        HybridIndex.load(path, FakeEmbedder(), max_documents=2)
    assert full.value.code == "index_full"


def test_save_to_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SearchError) as info:
        metadata_index().save(tmp_path / "missing" / "index.sqlite")
    assert info.value.code == "invalid_argument"


def test_round_trip_with_big_endian_byte_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = metadata_index()
    expected = summarize(index.search("cat", mode="vector"))
    monkeypatch.setattr(sys, "byteorder", "big")
    path = tmp_path / "index.sqlite"
    index.save(path)
    loaded = HybridIndex.load(path, FakeEmbedder())
    assert summarize(loaded.search("cat", mode="vector")) == expected


def test_vectors_are_stored_as_little_endian_float64(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    index = HybridIndex(FakeEmbedder())
    index.add([Document("a", "cat cat")])
    index.save(path)
    conn = sqlite3.connect(path)
    try:
        blob = conn.execute("SELECT vector FROM docs").fetchone()[0]
    finally:
        conn.close()
    if sys.byteorder == "little":
        assert struct.unpack("<4d", blob) == (2.0, 0.0, 0.0, 0.0)


# --- logging ----------------------------------------------------------------


def test_logging_never_contains_input_and_uses_null_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="hybrid_search_lite")
    index = HybridIndex(FakeEmbedder())
    index.add([Document("s1", "topsecretvalue cat")])
    index.search("topsecretvalue")
    assert "topsecretvalue" not in caplog.text
    handlers = logging.getLogger("hybrid_search_lite.index").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_search_hit_is_frozen() -> None:
    hit = basic_index().search("cat")[0]
    with pytest.raises(AttributeError):
        hit.score = 0.0  # type: ignore[misc]
