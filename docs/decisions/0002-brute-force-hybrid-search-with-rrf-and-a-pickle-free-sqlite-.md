# Brute-force hybrid search with RRF and a pickle-free SQLite file format

Status: accepted (generated, pending human review)

## Context

Small RAG projects need exact-term and semantic retrieval without running a vector database. Pure vector search misses codes and names, keyword search misses paraphrases, and common persistence shortcuts such as pickle execute code on load.

## Decision

Implement BM25 and brute-force cosine search in one dependency-free in-memory index and fuse the two rankings with reciprocal rank fusion. Persist to SQLite with a documented schema (meta and docs tables, little-endian float64 BLOB vectors, JSON metadata) and validate every row on load instead of deserializing code. Bound all inputs (ids, text, queries, documents, filters, rankings) and report failures through one SearchError base class with stable codes.

## Consequences

The library has zero dependencies and exact, documented scoring that is easy to test, and index files are safe to open from untrusted sources. Brute-force cosine search is linear in the corpus, so it is documented for about 50,000 documents; approximate indexes, stemming and embedding providers are out of scope. Mutations are not thread-safe, and the tokenizer and stopwords are not persisted and must be supplied again on load.
