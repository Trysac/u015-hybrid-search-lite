# Project record

## Objective

In-memory hybrid search fusing BM25 and cosine vector rankings with reciprocal rank fusion, metadata filters and pickle-free SQLite persistence

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested.
- Out of scope: anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 292440 / 83256 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.5639 |

## Factory run u015 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2412 | 75796 | 0 | 4246 | 0.3818 |
| fix-1 | claude-sonnet-5-5 | no | 8 | 967 | 7513 | 57129 | 0.0342 |
| fix-2 | claude-sonnet-5-5 | no | 4 | 338 | 5674 | 24367 | 0.0200 |
| fix-3 | claude-sonnet-5-5 | no | 6 | 507 | 5994 | 39374 | 0.0240 |
| fix-4 | claude-sonnet-5-5 | no | 6 | 854 | 6947 | 41054 | 0.0300 |
| docs | claude-haiku-5-5 | no | 123834 | 4794 | 0 | 0 | 0.0739 |

Total: $0.5639. Prices: Anthropic list prices.
