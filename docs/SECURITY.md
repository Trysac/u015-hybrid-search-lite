# Security notes

## Threat model

`trysac-hybrid-search-lite` is a pure library without I/O. The main risks are:

- **Supply chain**: a compromised dependency or CI action. Mitigation: no runtime dependencies,
  Dependabot weekly updates, actions pinned to major versions, minimal workflow permissions.
- **Leaked secrets**: tokens committed by mistake. Mitigation: `gitleaks` runs in the local gates
  and in CI; `.env` files are git-ignored.
- **Untrusted input**: callers may pass hostile strings. Mitigation: validate input and fail with
  explicit exceptions; never `eval`, `exec` or deserialize untrusted data.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a
long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
