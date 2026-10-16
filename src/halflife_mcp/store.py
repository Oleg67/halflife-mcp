"""The memory store: Markdown files (source of truth) + a SQLite index (SQL, vectors, graph).

Layout under the data home (HALFLIFE_HOME, else a per-OS default):

    memories/<scope>/<name>.md            one memory per file (memfile format)
    memories/<scope>/_graph/<name>.json   its entities + relations (optional)
    memories/<scope>/_archive/            faded and superseded memories (moved, not deleted)
    index.sqlite3                         derived index; delete it any time, it is rebuilt
    models/                               embedding model cache (only with the [vector] extra)

A memory id is "<scope dir>/<name>". Scope is "global" or a project: an absolute path
(normalized to its git root) or any short project name.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
from datetime import date, datetime

from . import memfile

KINDS = ("procedural", "preference", "fact", "episodic", "trace")
# file tier names (compatible with Claude Code memory_tools files) <-> API kinds
_TIER_OF_KIND = {"fact": "semantic"}
_KIND_OF_TIER = {"semantic": "fact", "long_term": "fact", "idea": "fact", "rule": "fact",
                 "working": "episodic"}
HALF_LIFE_DAYS = {"procedural": 365, "preference": 180, "fact": 180, "episodic": 30, "trace": 7}
DEFAULT_CONFIDENCE = {"procedural": 0.85, "preference": 0.7, "fact": 0.7, "episodic": 0.5,
                      "trace": 0.5}
PROVISIONAL_CONFIDENCE = 0.3
ARCHIVE_BELOW = 0.10          # decayed score below this -> moved to _archive (not traces)
TRACE_TTL_DAYS = 30           # traces are deleted after this
DUP_OVERLAP = 0.6             # same kind+scope, token Jaccard >= this -> reinforce, not a new file
MAX_DESCRIPTION = 200
MAX_BODY = 4000
ENTITY_TYPES = {"file", "repo", "tool", "library", "service", "person", "concept", "ticket",
                "command", "project", "organization"}
RELATIONS = {"uses", "depends_on", "part_of", "configures", "fixed", "caused", "contradicts",
             "supersedes", "related_to", "owns", "prefers"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, scope TEXT NOT NULL, scope_dir TEXT NOT NULL, kind TEXT NOT NULL,
  name TEXT NOT NULL, description TEXT, body TEXT, status TEXT NOT NULL,
  confidence REAL NOT NULL, reinforced INTEGER NOT NULL, last_reinforced TEXT,
  created TEXT, contradicts TEXT, source TEXT, path TEXT NOT NULL,
  mtime REAL NOT NULL, size INTEGER NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS memories_scope ON memories(scope_dir, kind, status);
CREATE TABLE IF NOT EXISTS vectors (
  id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
  model TEXT NOT NULL, vec BLOB NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, norm TEXT NOT NULL UNIQUE, type TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mentions (
  memory TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  entity INTEGER NOT NULL REFERENCES entities(id), PRIMARY KEY (memory, entity));
CREATE TABLE IF NOT EXISTS edges (
  src INTEGER NOT NULL REFERENCES entities(id), rel TEXT NOT NULL,
  dst INTEGER NOT NULL REFERENCES entities(id),
  memory TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (src, rel, dst, memory));
"""


# ---------------------------------------------------------------- locations
def data_home() -> str:
    env = os.environ.get("HALFLIFE_HOME")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "halflife")


def scope_key(scope: str | None) -> str:
    """'global', a project path normalized to its git root, or a project name."""
    s = (scope or "").strip()
    if not s or s.lower() == "global":
        return "global"
    if "/" in s or "\\" in s or s.startswith("~"):
        p = os.path.abspath(os.path.expanduser(s))
        cur = p
        while True:
            if os.path.exists(os.path.join(cur, ".git")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                return p
            cur = parent
    return s


def scope_dir_name(key: str) -> str:
    return "global" if key == "global" else (re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-") or "project")


def kind_of(meta: dict[str, str]) -> str:
    t = (meta.get("kind") or meta.get("tier") or "").strip().lower()
    t = _KIND_OF_TIER.get(t, t)
    if t in KINDS:
        return t
    return {"feedback": "procedural", "user": "preference", "project": "episodic"}.get(
        meta.get("type", ""), "fact")


def _float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def decayed(kind: str, confidence: float, last: str | None, today: date | None = None) -> float:
    today = today or date.today()
    anchor = _date(last) if last else None
    age = max(0, (today - anchor).days) if anchor else 0
    return confidence * 0.5 ** (age / HALF_LIFE_DAYS.get(kind, 180))


def clean_graph(entities, relations) -> dict:
    """Validated {entities, relations}; unknown types -> concept / related_to."""
    ents, seen = [], set()
    for e in entities or []:
        if isinstance(e, str):
            e = {"name": e}
        if not isinstance(e, dict):
            continue
        name = " ".join(str(e.get("name", "")).split())[:120]
        if len(name) < 2 or name.lower() in seen:
            continue
        seen.add(name.lower())
        t = str(e.get("type", "")).strip().lower()
        ents.append({"name": name, "type": t if t in ENTITY_TYPES else "concept"})
    rels = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        src, dst = (" ".join(str(r.get(k, "")).split())[:120] for k in ("src", "dst"))
        if len(src) < 2 or len(dst) < 2 or src.lower() == dst.lower():
            continue
        rel = str(r.get("rel", "")).strip().lower()
        rel = rel if rel in RELATIONS else "related_to"
        if {"src": src, "rel": rel, "dst": dst} not in rels:
            rels.append({"src": src, "rel": rel, "dst": dst})
        for name in (src, dst):
            if name.lower() not in seen:
                seen.add(name.lower())
                ents.append({"name": name, "type": "concept"})
    return {"entities": ents, "relations": rels}


class MemoryError_(ValueError):
    """A request the store refuses (bad id, secret in content, unknown kind...)."""


class Store:
    def __init__(self, home: str | None = None, embedder="auto"):
        self.home = home or data_home()
        self.root = os.path.join(self.home, "memories")
        os.makedirs(self.root, exist_ok=True)
        try:
            os.chmod(self.home, 0o700)
        except OSError:
            pass
        self._embedder = embedder
        self._lock = threading.RLock()
        self.db = sqlite3.connect(os.path.join(self.home, "index.sqlite3"),
                                  check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ embedding
    @property
    def embedder(self):
        if self._embedder == "auto":
            from .embed import get_embedder
            self._embedder = get_embedder(os.path.join(self.home, "models"))
        return self._embedder

    # ------------------------------------------------------------ files
    def _path(self, mem_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", mem_id or "") or ".." in mem_id:
            raise MemoryError_(f"bad memory id: {mem_id!r}")
        return os.path.join(self.root, mem_id + ".md")

    def _sidecar(self, scope_dir: str, name: str) -> str:
        return os.path.join(self.root, scope_dir, "_graph", name + ".json")

    def _read(self, path: str) -> dict:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        meta, body = memfile.parse(text)
        scope_dir = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[:-3]
        kind = kind_of(meta)
        status = meta.get("status", "active")
        graph = {}
        try:
            with open(self._sidecar(scope_dir, name), encoding="utf-8") as fh:
                graph = json.load(fh)
        except (OSError, ValueError):
            pass
        return {
            "id": f"{scope_dir}/{name}", "scope": meta.get("scope") or
            ("global" if scope_dir == "global" else scope_dir), "scope_dir": scope_dir,
            "kind": kind, "name": name, "description": meta.get("description", ""),
            "body": body, "status": status if status in ("active", "provisional", "superseded")
            else "active", "confidence": _float(meta.get("confidence"), DEFAULT_CONFIDENCE[kind]),
            "reinforced": _int(meta.get("reinforced")),
            "last_reinforced": meta.get("last_reinforced") or meta.get("created") or "",
            "created": meta.get("created", ""), "contradicts": meta.get("contradicts", ""),
            "source": meta.get("source", ""), "path": path,
            "graph": graph if isinstance(graph, dict) else {},
            "hash": hashlib.sha256((text + json.dumps(graph, sort_keys=True)).encode()).hexdigest(),
        }

    def _write(self, rec: dict) -> None:
        meta = {"kind": rec["kind"], "tier": _TIER_OF_KIND.get(rec["kind"], rec["kind"]),
                "scope": rec["scope"], "confidence": round(rec["confidence"], 2),
                "reinforced": rec["reinforced"], "last_reinforced": rec["last_reinforced"],
                "status": rec["status"], "contradicts": rec.get("contradicts"),
                "superseded_by": rec.get("superseded_by"), "source": rec.get("source"),
                "created": rec["created"]}
        memfile.write_atomic(rec["path"], memfile.render(rec["name"], rec["description"], meta,
                                                         rec["body"]))
        g = rec.get("graph") or {}
        side = self._sidecar(rec["scope_dir"], rec["name"])
        if g.get("entities"):
            os.makedirs(os.path.dirname(side), exist_ok=True)
            with open(side, "w", encoding="utf-8") as fh:
                json.dump(g, fh, indent=1, ensure_ascii=False)
        elif os.path.exists(side):
            os.remove(side)

    def _scope_dirs(self) -> list[str]:
        return sorted(d for d in os.listdir(self.root)
                      if os.path.isdir(os.path.join(self.root, d)) and not d.startswith((".", "_")))

    # ------------------------------------------------------------ index
    def sync(self) -> dict:
        """Index new/changed files, drop rows for removed ones. Cheap when nothing changed."""
        stats = {"indexed": 0, "removed": 0, "unchanged": 0}
        with self._lock, self.db:
            known = {r["id"]: (r["mtime"], r["size"]) for r in
                     self.db.execute("SELECT id, mtime, size FROM memories")}
            present = set()
            for sd in self._scope_dirs():
                for fn in sorted(os.listdir(os.path.join(self.root, sd))):
                    if not fn.endswith(".md") or fn.startswith(".") or fn == "MEMORY.md":
                        continue
                    path = os.path.join(self.root, sd, fn)
                    mem_id = f"{sd}/{fn[:-3]}"
                    present.add(mem_id)
                    st = os.stat(path)
                    side = self._sidecar(sd, fn[:-3])
                    mtime = max(st.st_mtime, os.path.getmtime(side) if os.path.exists(side) else 0)
                    if known.get(mem_id) == (mtime, st.st_size):
                        stats["unchanged"] += 1
                        continue
                    self._index(self._read(path), mtime, st.st_size)
                    stats["indexed"] += 1
            for mem_id in set(known) - present:
                self.db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
                stats["removed"] += 1
            self.db.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity FROM mentions)")
            emb = self.embedder
            if emb:          # memories indexed before vector search was available, or by another model
                model = getattr(emb, "model", "embedder")
                todo = self.db.execute(
                    "SELECT m.id, m.name, m.description, m.body, m.hash FROM memories m "
                    "LEFT JOIN vectors v ON v.id=m.id WHERE v.id IS NULL OR v.model!=?",
                    (model,)).fetchall()
                if todo:
                    vecs = emb([" ".join([r["name"].replace("-", " "), r["description"] or "",
                                          r["body"] or ""])[:2000] for r in todo])
                    for r, v in zip(todo, vecs):
                        self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                                        (r["id"], model, _pack(v), r["hash"]))
                    stats["embedded"] = len(todo)
        return stats

    def _index(self, rec: dict, mtime: float | None = None, size: int | None = None) -> None:
        if mtime is None:
            st = os.stat(rec["path"])
            side = self._sidecar(rec["scope_dir"], rec["name"])
            mtime = max(st.st_mtime, os.path.getmtime(side) if os.path.exists(side) else 0)
            size = st.st_size
        old = self.db.execute("SELECT hash FROM memories WHERE id=?", (rec["id"],)).fetchone()
        if old and old["hash"] == rec["hash"]:
            self.db.execute("UPDATE memories SET mtime=?, size=? WHERE id=?",
                            (mtime, size, rec["id"]))
            return
        self.db.execute("DELETE FROM memories WHERE id=?", (rec["id"],))      # cascades
        self.db.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec["id"], rec["scope"], rec["scope_dir"], rec["kind"], rec["name"],
             rec["description"], rec["body"], rec["status"], rec["confidence"],
             rec["reinforced"], rec["last_reinforced"], rec["created"], rec["contradicts"],
             rec["source"], rec["path"], mtime, size, rec["hash"]))
        g = clean_graph(rec["graph"].get("entities"), rec["graph"].get("relations"))
        ids = {}
        for e in g["entities"]:
            norm = e["name"].lower()
            row = self.db.execute("SELECT id, type FROM entities WHERE norm=?", (norm,)).fetchone()
            if row:
                eid = row["id"]
                if row["type"] == "concept" and e["type"] != "concept":
                    self.db.execute("UPDATE entities SET type=? WHERE id=?", (e["type"], eid))
            else:
                eid = self.db.execute("INSERT INTO entities(name, norm, type) VALUES (?,?,?)",
                                      (e["name"], norm, e["type"])).lastrowid
            ids[norm] = eid
            self.db.execute("INSERT OR IGNORE INTO mentions VALUES (?,?)", (rec["id"], eid))
        for r in g["relations"]:
            self.db.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                            (ids[r["src"].lower()], r["rel"], ids[r["dst"].lower()], rec["id"]))
        emb = self.embedder
        if emb:
            text = " ".join([rec["name"].replace("-", " "), rec["description"], rec["body"]])[:2000]
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                            (rec["id"], getattr(emb, "model", "embedder"),
                             _pack(emb([text])[0]), rec["hash"]))

    def _reindex(self, rec_path: str) -> None:
        with self._lock, self.db:
            self._index(self._read(rec_path))

    def _unindex(self, mem_id: str) -> None:
        with self._lock, self.db:
            self.db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            self.db.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity FROM mentions)")

    # ------------------------------------------------------------ queries
    def rows(self, scopes: list[str], kinds=None, statuses=("active",)) -> list[sqlite3.Row]:
        kinds = list(kinds or KINDS)
        q = (f"SELECT * FROM memories WHERE scope_dir IN ({','.join('?' * len(scopes))}) "
             f"AND kind IN ({','.join('?' * len(kinds))}) "
             f"AND status IN ({','.join('?' * len(statuses))})")
        with self._lock:
            return self.db.execute(q, [*scopes, *kinds, *statuses]).fetchall()

    def vector_scores(self, text: str, ids: list[str]) -> dict[str, float]:
        emb = self.embedder
        if not emb or not ids:
            return {}
        q = _unit(emb([text])[0])
        with self._lock:
            rows = self.db.execute(
                f"SELECT id, vec FROM vectors WHERE id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
        out = {}
        for r in rows:
            v = array.array("f")
            v.frombytes(r["vec"])
            if len(v) == len(q):
                out[r["id"]] = sum(a * b for a, b in zip(q, v))
        return out

    def graph_scores(self, text: str, scopes: list[str], hub_mentions: int = 3,
                     neighbor_weight: float = 0.5) -> dict[str, tuple[float, list[str]]]:
        """Memories naming an entity that appears in `text` (1.0 each) or a non-hub 1-hop
        neighbor of one (neighbor_weight)."""
        low = text.lower()
        toks = set(re.findall(r"[a-z0-9]+", low))
        sm = ",".join("?" * len(scopes))
        with self._lock:
            cands = self.db.execute(
                f"SELECT DISTINCT e.id, e.name, e.norm FROM entities e JOIN mentions x ON "
                f"x.entity=e.id JOIN memories m ON m.id=x.memory WHERE m.scope_dir IN ({sm})",
                scopes).fetchall()
            seeds = {r["id"]: r["name"] for r in cands if _named_in(r["norm"], low, toks)}
            if not seeds:
                return {}
            ids = list(seeds)
            im = ",".join("?" * len(ids))
            neighbors = {r[0] for r in self.db.execute(
                f"SELECT n FROM (SELECT dst AS n FROM edges WHERE src IN ({im}) UNION "
                f"SELECT src FROM edges WHERE dst IN ({im})) "
                f"WHERE (SELECT COUNT(*) FROM mentions WHERE entity=n) <= ?",
                [*ids, *ids, hub_mentions])} - set(ids)
            out: dict[str, tuple[float, list[str]]] = {}
            for weight, group in ((1.0, ids), (neighbor_weight, sorted(neighbors))):
                if not group:
                    continue
                gm = ",".join("?" * len(group))
                for r in self.db.execute(
                        f"SELECT x.memory, x.entity FROM mentions x JOIN memories m ON "
                        f"m.id=x.memory WHERE x.entity IN ({gm}) AND m.scope_dir IN ({sm})",
                        [*group, *scopes]):
                    s, names = out.get(r[0], (0.0, []))
                    if r[1] in seeds:
                        names = names + [seeds[r[1]]]
                    out[r[0]] = (s + weight, names)
        return out

    # ------------------------------------------------------------ operations
    def remember(self, kind: str, content: str, scope: str | None = None,
                 description: str | None = None, name: str | None = None,
                 confidence: float | None = None, provisional: bool = False,
                 contradicts: str | None = None, entities=None, relations=None,
                 source: str | None = None) -> dict:
        kind = _KIND_OF_TIER.get((kind or "").strip().lower(), (kind or "").strip().lower())
        if kind not in KINDS:
            raise MemoryError_(f"kind must be one of {', '.join(KINDS)}")
        content = (content or "").strip()
        if not content:
            raise MemoryError_("content is empty")
        if len(content) > MAX_BODY:
            raise MemoryError_(f"content is longer than {MAX_BODY} characters; store a summary "
                               "and point at the source instead")
        description = " ".join((description or content.splitlines()[0]).split())[:MAX_DESCRIPTION]
        for field in (content, description, name or "", source or "",
                      json.dumps([entities, relations], default=str)):
            hit = memfile.find_secret(field)
            if hit:
                raise MemoryError_(f"refused: looks like a secret ({hit}). Never store "
                                   "credentials in memory.")
        key = scope_key(scope)
        sd = scope_dir_name(key)
        today = date.today().isoformat()
        graph = clean_graph(entities, relations)
        if contradicts:
            self._path(contradicts)                                   # validates the id
        self.sync()                                                   # pick up manual edits
        # near-duplicate of an existing memory -> reinforce it instead of a second file
        if not contradicts:
            toks = memfile.tokens(description + " " + content)
            best, best_ov = None, 0.0
            for r in self.rows([sd], [kind], ("active", "provisional")):
                ov = memfile.jaccard(toks, memfile.tokens(f"{r['description']} {r['body']}"))
                if ov > best_ov:
                    best, best_ov = r, ov
            if best is not None and best_ov >= DUP_OVERLAP:
                if provisional and best["status"] == "provisional" and best["created"] == today:
                    # an inference repeated in the same day is not independent corroboration
                    return {"action": "unchanged", "id": best["id"], "status": "provisional",
                            "overlap": round(best_ov, 2)}
                res = self.reinforce(best["id"], graph=graph)
                return dict(res, action="reinforced", overlap=round(best_ov, 2))
        base = memfile.slugify(name or description)
        nm, i = base, 2
        while os.path.exists(os.path.join(self.root, sd, nm + ".md")):
            nm, i = f"{base}-{i}", i + 1
        rec = {"id": f"{sd}/{nm}", "scope": key, "scope_dir": sd, "kind": kind, "name": nm,
               "description": description, "body": content,
               "status": "provisional" if provisional else "active",
               "confidence": PROVISIONAL_CONFIDENCE if provisional else
               max(0.0, min(1.0, confidence if confidence is not None else DEFAULT_CONFIDENCE[kind])),
               "reinforced": 0, "last_reinforced": today, "created": today,
               "contradicts": contradicts or "", "source": source or "",
               "path": os.path.join(self.root, sd, nm + ".md"), "graph": graph}
        with self._lock:
            self._write(rec)
            superseded = self._supersede(rec) if rec["status"] == "active" else None
            self._reindex(rec["path"])
        out = {"action": "created", "id": rec["id"], "status": rec["status"],
               "confidence": rec["confidence"]}
        if superseded:
            out["superseded"] = superseded
        return out

    def _supersede(self, rec: dict) -> str | None:
        """An active memory that contradicts another retires it to _archive."""
        old_id = rec.get("contradicts")
        if not old_id or old_id == rec["id"]:
            return None
        old_path = self._path(old_id)
        if not os.path.exists(old_path):
            return None
        old = self._read(old_path)
        old.update(status="superseded", superseded_by=rec["id"])
        self._write(old)
        self._archive(old)
        return old_id

    def _archive(self, rec: dict) -> None:
        adir = os.path.join(self.root, rec["scope_dir"], "_archive")
        os.makedirs(adir, exist_ok=True)
        shutil.move(rec["path"], os.path.join(adir, rec["name"] + ".md"))
        side = self._sidecar(rec["scope_dir"], rec["name"])
        if os.path.exists(side):
            os.makedirs(os.path.join(adir, "_graph"), exist_ok=True)
            shutil.move(side, os.path.join(adir, "_graph", rec["name"] + ".json"))
        self._unindex(rec["id"])

    def reinforce(self, mem_id: str, graph: dict | None = None) -> dict:
        """The memory proved true/useful again: confidence +0.1, decay clock reset,
        provisional -> active (which also applies its pending supersession)."""
        path = self._path(mem_id)
        if not os.path.exists(path):
            raise MemoryError_(f"no such memory: {mem_id}")
        with self._lock:
            rec = self._read(path)
            was = rec["status"]
            base = max(rec["confidence"], DEFAULT_CONFIDENCE[rec["kind"]] - 0.1) \
                if was == "provisional" else rec["confidence"]
            rec.update(confidence=min(1.0, base + 0.1), reinforced=rec["reinforced"] + 1,
                       last_reinforced=date.today().isoformat(), status="active")
            if graph and graph.get("entities"):
                g = rec["graph"] or {}
                rec["graph"] = clean_graph((g.get("entities") or []) + graph["entities"],
                                           (g.get("relations") or []) + graph.get("relations", []))
            self._write(rec)
            superseded = self._supersede(rec) if was == "provisional" else None
            self._reindex(path)
        out = {"id": mem_id, "status": "active", "confidence": round(rec["confidence"], 2),
               "reinforced": rec["reinforced"], "was": was}
        if superseded:
            out["superseded"] = superseded
        return out

    def forget(self, mem_id: str) -> dict:
        """Delete for real: the file, its graph, any archived copy, and the index rows."""
        path = self._path(mem_id)
        sd, name = mem_id.split("/", 1)
        targets = [path, self._sidecar(sd, name),
                   os.path.join(self.root, sd, "_archive", name + ".md"),
                   os.path.join(self.root, sd, "_archive", "_graph", name + ".json")]
        removed = [p for p in targets if os.path.exists(p)]
        if not removed:
            raise MemoryError_(f"no such memory: {mem_id}")
        with self._lock:
            for p in removed:
                os.remove(p)
            self._unindex(mem_id)
        return {"id": mem_id, "deleted_files": len(removed)}

    def get(self, mem_id: str) -> dict:
        path = self._path(mem_id)
        if not os.path.exists(path):
            raise MemoryError_(f"no such memory: {mem_id}")
        rec = self._read(path)
        rec["score"] = round(decayed(rec["kind"], rec["confidence"], rec["last_reinforced"]), 3)
        for k in ("path", "hash"):
            rec.pop(k)
        return rec

    def maintain(self) -> dict:
        """Archive faded memories; delete traces past TRACE_TTL_DAYS."""
        today = date.today()
        archived, expired = [], []
        with self._lock:
            for sd in self._scope_dirs():
                for fn in os.listdir(os.path.join(self.root, sd)):
                    if not fn.endswith(".md") or fn.startswith("."):
                        continue
                    rec = self._read(os.path.join(self.root, sd, fn))
                    if rec["kind"] == "trace":
                        created = _date(rec["created"] or rec["last_reinforced"])
                        if created and (today - created).days > TRACE_TTL_DAYS:
                            self.forget(rec["id"])
                            expired.append(rec["id"])
                        continue
                    if rec["status"] == "superseded":
                        self._archive(rec)
                        archived.append(rec["id"])
                    elif decayed(rec["kind"], rec["confidence"], rec["last_reinforced"],
                                 today) < ARCHIVE_BELOW:
                        self._archive(rec)
                        archived.append(rec["id"])
        return {"archived": archived, "expired_traces": expired}

    def stats(self) -> dict:
        with self._lock:
            q = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
            by = {}
            for r in self.db.execute("SELECT scope, kind, status, COUNT(*) n FROM memories "
                                     "GROUP BY 1, 2, 3"):
                by.setdefault(r["scope"], {}).setdefault(r["kind"], {})[r["status"]] = r["n"]
            return {"home": self.home, "memories": q("SELECT COUNT(*) FROM memories"),
                    "vectors": q("SELECT COUNT(*) FROM vectors"),
                    "entities": q("SELECT COUNT(*) FROM entities"),
                    "relations": q("SELECT COUNT(*) FROM edges"),
                    "vector_search": bool(self.embedder), "by_scope": by}


def _named_in(norm: str, low: str, toks: set[str]) -> bool:
    if len(norm) < 3:
        return False
    if any(c in norm for c in " /._-"):
        return norm in low
    return norm in toks


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _pack(v) -> bytes:
    return array.array("f", _unit(v)).tobytes()
