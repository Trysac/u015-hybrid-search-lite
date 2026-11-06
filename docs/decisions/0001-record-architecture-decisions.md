# 1. Record architecture decisions

- Status: accepted
- Date: 2026

## Context

Decisions about this utility (dependencies, public API shape, supported Python versions) must stay
understandable for future maintainers and for the agents that change the code.

## Decision

We record every significant decision as an Architecture Decision Record in `docs/decisions/`, named
`NNNN-short-title.md`, with the sections Context, Decision and Consequences. ADRs are immutable once
accepted; a later ADR supersedes an earlier one explicitly.

## Consequences

- Reviewers can see why the code is the way it is without reading the whole history.
- Adding a runtime dependency or changing the public API requires a new ADR.
