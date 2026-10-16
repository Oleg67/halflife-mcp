"""Memory files: Markdown with a YAML-like frontmatter block, written atomically.

    ---
    name: prefers-uv
    description: "user prefers uv over pip for python dependencies"
    metadata:
      tier: preference
      scope: global
      confidence: 0.7
      reinforced: 0
      last_reinforced: 2026-10-10
      created: 2026-10-10
    ---

    <body>

Only flat `key: value` pairs are used (nested under `metadata:` or not), so files stay
hand-editable and compatible with Claude Code's own memory files.
"""

from __future__ import annotations

import os
import re
import tempfile

_FM = re.compile(r"^---\s*\n(.*?)\n---[^\n]*\n?(.*)$", re.S)
_KV = re.compile(r"\s*([A-Za-z_][\w]*):\s*(.*)$")

# Secrets are refused, not redacted: a memory that needs one should not be stored.
_SECRET_WORDS = r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|credential|cookie)"
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{16,}"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),                                 # user:pass@host
    re.compile(r"\beyJ[\w-]{5,}\.[\w-]{5,}\.[\w-]{5,}"),                   # JWT
    re.compile(r"\b(?:sk|pk|rk)-[\w-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat|glpat|xox[abprs]|npm|pypi)[-_][\w-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[\w-]{30,}"),
    re.compile(r"(?i)\b[\w.-]*" + _SECRET_WORDS + r"[\w.-]*[\"']?\s*[:=]\s*[\"']?[^\s\"',;]{6,}"),
]


def find_secret(text: str) -> str | None:
    """The first secret-looking substring (truncated), or None."""
    for rx in SECRET_PATTERNS:
        m = rx.search(text)
        if m:
            return m.group(0)[:12] + "…"
    return None


def parse(text: str) -> tuple[dict[str, str], str]:
    """(metadata, body). Keys from `name:`, `description:` and `metadata:` are merged."""
    m = _FM.match(text)
    if not m:
        return {}, text.strip()
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        kv = _KV.match(line)
        if not kv:
            continue
        key, val = kv.group(1), kv.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if val:                                   # skip container lines like "metadata:"
            meta.setdefault(key, val)
    return meta, m.group(2).strip()


def _one_line(v: object) -> str:
    return " ".join(str(v).split())


def render(name: str, description: str, meta: dict[str, object], body: str) -> str:
    desc = _one_line(description).replace('"', "'")
    lines = ["---", f"name: {name}", f'description: "{desc}"', "metadata:"]
    lines += [f"  {k}: {_one_line(v)}" for k, v in meta.items() if v not in (None, "")]
    return "\n".join(lines + ["---", "", body.strip(), ""])


def write_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def slugify(s: str, n: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:n].rstrip("-") or "memory")


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
