"""Run every quality gate of this repository, in order, stopping at the first failure.

Usage: ``uv run python scripts/check.py`` (the same command runs locally, in the factory and in CI).

Gates: lint, format, types, tests with branch coverage >= 80 %, documentation manifest, secrets,
known vulnerabilities in runtime dependencies.
Standard library only, no shell: it behaves the same on Windows and Linux.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_MANIFEST = ROOT / "docs-manifest.json"
ALLOW_MISSING_GITLEAKS = "FABRICA_ALLOW_MISSING_GITLEAKS"
ALLOW_OFFLINE_AUDIT = "FABRICA_ALLOW_OFFLINE_AUDIT"
GITLEAKS_HINT = (
    "gitleaks is not installed. Install it (https://github.com/gitleaks/gitleaks#installing), "
    "e.g. `winget install Gitleaks.Gitleaks`, `brew install gitleaks` or a release binary, "
    f"or set {ALLOW_MISSING_GITLEAKS}=1 to skip this gate locally."
)


def run(argv: list[str]) -> int:
    """Run one tool from the repository root and return its exit code."""
    return subprocess.run(argv, cwd=ROOT, check=False).returncode


def py_tool(module: str, *args: str) -> Callable[[], int]:
    """Run a Python tool through the current interpreter, so it works inside ``uv run``."""
    return lambda: run([sys.executable, "-m", module, *args])


def docs_errors(root: Path = ROOT) -> list[str]:
    """Return every violation of ``docs-manifest.json``; an empty list means the gate passes."""
    manifest = json.loads((root / "docs-manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for rel in manifest.get("required", []):
        path = root / rel
        if not path.is_file():
            errors.append(f"missing required file: {rel}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"required file is empty: {rel}")
    for pattern in manifest.get("required_globs", []):
        if not any(p.is_file() for p in root.glob(pattern)):
            errors.append(f"no file matches required pattern: {pattern}")
    for rel, limit in manifest.get("max_lines", {}).items():
        path = root / rel
        if path.is_file():
            count = len(path.read_text(encoding="utf-8").splitlines())
            if count > limit:
                errors.append(f"{rel} has {count} lines (max {limit})")
    for rel, needle in manifest.get("must_reference", {}).items():
        path = root / rel
        if path.is_file() and needle not in path.read_text(encoding="utf-8"):
            errors.append(f"{rel} must reference {needle}")
    return errors


def docs_gate() -> int:
    errors = docs_errors()
    for error in errors:
        print(f"docs: {error}", file=sys.stderr)
    if not errors:
        print("docs: every file in docs-manifest.json is present")
    return 1 if errors else 0


def secrets_gate() -> int:
    exe = shutil.which("gitleaks")
    if exe is None:
        if os.environ.get(ALLOW_MISSING_GITLEAKS) == "1":
            print("gitleaks: SKIPPED (not installed)")
            return 0
        print(f"gitleaks: FAIL - {GITLEAKS_HINT}", file=sys.stderr)
        return 1
    return run([exe, "dir", ".", "--redact", "--no-banner"])


def audit_gate() -> int:
    """Check runtime dependencies against the PyPI/OSV advisory databases with pip-audit."""
    if os.environ.get(ALLOW_OFFLINE_AUDIT) == "1":
        print("audit: SKIPPED (FABRICA_ALLOW_OFFLINE_AUDIT=1)")
        return 0
    uv = shutil.which("uv")
    if uv is None:
        print("audit: FAIL - uv is required to export the dependency list", file=sys.stderr)
        return 1
    export = subprocess.run(
        [
            uv,
            "export",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--format",
            "requirements-txt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if export.returncode != 0:
        print(export.stderr, file=sys.stderr)
        return export.returncode
    pins = [
        line
        for line in export.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not pins:
        print("audit: no runtime dependencies")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp) / "requirements.txt"
        reqs.write_text("\n".join(pins) + "\n", encoding="utf-8")
        return run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "-r",
                str(reqs),
                "--disable-pip",
                "--no-deps",
                "--strict",
            ]
        )


GATES: list[tuple[str, Callable[[], int]]] = [
    ("lint", py_tool("ruff", "check", ".")),
    ("format", py_tool("ruff", "format", "--check", ".")),
    ("types", py_tool("mypy", "--strict", "src")),
    (
        "tests",
        py_tool(
            "pytest",
            "--cov=hybrid_search_lite",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
        ),
    ),
    ("docs", docs_gate),
    ("secrets", secrets_gate),
    ("audit", audit_gate),
]


def main() -> int:
    for name, gate in GATES:
        print(f"==> {name}", flush=True)
        code = gate()
        if code != 0:
            print(f"==> {name}: FAILED (exit {code})", file=sys.stderr, flush=True)
            return code
        print(f"==> {name}: ok", flush=True)
    print("==> all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
