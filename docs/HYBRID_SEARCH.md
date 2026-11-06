# Hybrid search

`trysac-hybrid-search-lite` fuses a BM25 keyword ranking and a cosine vector ranking with
reciprocal rank fusion (RRF). It has no runtime dependencies (only `sqlite3`, `math` and `array`
from the standard library) and keeps everything in memory. Vector search is brute force and is
documented for up to about 50,000 documents.

## Scoring

- **BM25** (Robertson and Zaragoza, *The Probabilistic Relevance Framework*) with the non-negative
  Lucene idf `ln(1 + (N - n + 0.5) / (n + 0.5))` and term-frequency saturation
  `(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))`. Query terms are deduplicated. `N`, `n`
  and `avgdl` always describe the whole corpus, also when filters are active.
- **Vector**: cosine similarity between the query embedding and each document embedding.
- **RRF** (Cormack, Clarke and Buettcher, SIGIR 2009): `score(d) = sum(weight_i / (k + rank_i(d)))`
  with 1-based ranks over the top `candidates` of each retriever.
- Ties sort by document id ascending everywhere.

## Filters

Filters apply before ranking. A scalar means equality, a list means membership, an unknown key
never matches. `True` never equals `1`.

## File format

`HybridIndex.save` writes SQLite with `meta(key, value)` (schema `1`, `model_id`, `dimension`,
`k1`, `b`) and `docs(id PRIMARY KEY, text, metadata_json, vector BLOB)`. Vectors are little-endian
float64. Metadata is JSON, so `bool` and `int` stay distinct. Nothing is pickled or evaluated;
`load` validates every row and raises `SearchError` with code `corrupt_file` otherwise.
The tokenizer and stopwords are not stored: pass the same ones to `load`.

## Limits and errors

Every error is a `SearchError` (a `ValueError`) with a stable `code`: `dimension_mismatch`,
`index_mismatch`, `index_full`, `input_too_large`, `corrupt_file` or `invalid_argument`.
Ids are 1 to 256 characters, document text at most 1,000,000 characters, queries at most
`max_query_chars` (default 10,000), the index holds at most `max_documents` (default 100,000).
Embedding components must be finite with an absolute value of at most 1e150.

## Thread safety

`add` and `remove` are not thread-safe: serialize mutations and do not search during a mutation.
