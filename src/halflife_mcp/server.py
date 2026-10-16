"""MCP server (stdio) + small CLI.

    halflife-mcp                      serve over stdio (what MCP clients launch)
    halflife-mcp serve --http [--port 8765]
                                      serve over HTTP at http://127.0.0.1:8765/mcp, for local
                                      clients that connect by URL. No authentication: it only
                                      binds to loopback unless --allow-network is given.
    halflife-mcp stats                counts per scope / kind / status, data location
    halflife-mcp brief [--scope S]
    halflife-mcp recall "<query>" [--scope S]
    halflife-mcp maintain             archive faded memories, delete expired traces
    halflife-mcp rebuild              delete the index and rebuild it from the files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import recall as _recall
from .store import KINDS, Store, scope_dir_name, scope_key

INSTRUCTIONS = """\
Long-term memory that persists across conversations on this computer. Five kinds:
preference (how the user likes things done), procedural (workflows that worked), fact
(durable facts not derivable from code or docs), episodic (what happened in a past task),
trace (short-lived log of files/commands, for resuming work).

Scope: pass the absolute project/workspace path as `scope` when working in a project (it
is normalized to the git root); omit it or use "global" for things that apply everywhere.
Project queries always include global memories.

When to use:
- Start of a non-trivial task: call memory_brief (preferences + procedures always apply),
  then memory_recall with a short description of the task.
- Treat memories as context, not authority: the current user request and the actual
  code/docs win. Verify facts before relying on them.
- When the user states a preference or corrects you: memory_remember right away
  (kind=preference, scope global unless project-specific).
- When you infer something the user did not say: memory_remember with provisional=true; it
  stays hidden until confirmed later.
- When a memory turned out useful or was confirmed: memory_reinforce.
- When a memory is wrong or outdated: memory_remember the correction with contradicts=<old id>
  (the old one is retired, not deleted).
- End of a substantive task: one kind=episodic memory: goal, what was done, decisions,
  open items. Pointers (paths, URLs) rather than copies.
- When the user asks you to forget something: memory_forget (permanent).
Never store secrets, credentials, raw transcripts or unnecessary personal data; such writes
are refused.
"""


def create_server(get_store: Callable[[], Store]) -> MCPServer:
    mcp = MCPServer("halflife", instructions=INSTRUCTIONS)
    ro = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    write = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    @mcp.tool(annotations=ro)
    def memory_brief(scope: str | None = None) -> dict[str, Any]:
        """What to know before starting work: all active preferences and procedures, the
        strongest facts, and the most recent episodes for this scope (plus global)."""
        return _recall.brief(get_store(), scope)

    @mcp.tool(annotations=ro)
    def memory_recall(query: str, scope: str | None = None, k: int = 5,
                      kinds: list[str] | None = None,
                      include_provisional: bool = False) -> dict[str, Any]:
        """Search memory with hybrid keyword + vector + graph ranking. `query`: a short
        description of the task or question (names of files, tools and projects help).
        `kinds` defaults to all except trace. Returns the best matches, best first, with the
        signals that found them. Read-only."""
        hits = _recall.recall(get_store(), query, scope, k, kinds, include_provisional)
        return {"query": query, "scope": scope_key(scope), "results": hits}

    @mcp.tool(annotations=write)
    def memory_remember(kind: str, content: str, scope: str | None = None,
                        description: str | None = None, name: str | None = None,
                        provisional: bool = False, contradicts: str | None = None,
                        entities: list[dict] | None = None,
                        relations: list[dict] | None = None,
                        source: str | None = None, confidence: float | None = None) -> dict[str, Any]:
        """Store one memory. kind: preference | procedural | fact | episodic | trace.
        content: the memory itself, short (for preference/procedural include why and how to
        apply). description: one keyword-rich line (used for matching; defaults to the first
        line). provisional: true when inferred rather than stated by the user. contradicts:
        id of a memory this one replaces. entities: [{"name", "type"}] for the specific
        files/tools/projects/people/concepts it is about (type: file, repo, tool, library,
        service, person, concept, ticket, command, project, organization). relations:
        [{"src", "rel", "dst"}] between those entities (rel: uses, depends_on, part_of,
        configures, fixed, caused, contradicts, supersedes, related_to, owns, prefers).
        source: where it came from (URL, file path). A near-duplicate reinforces the
        existing memory instead of creating a new one."""
        return get_store().remember(kind, content, scope, description, name, confidence,
                                    provisional, contradicts, entities, relations, source)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                          idempotent_hint=False, open_world_hint=False))
    def memory_reinforce(id: str) -> dict[str, Any]:
        """A memory was confirmed or proved useful: raises its confidence, resets its decay
        and activates it if provisional."""
        return get_store().reinforce(id)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                          open_world_hint=False))
    def memory_forget(id: str) -> dict[str, Any]:
        """Permanently delete a memory (file, graph, archived copy, index). Only when the user
        asks to forget something."""
        return get_store().forget(id)

    @mcp.tool(annotations=ro)
    def memory_get(id: str) -> dict[str, Any]:
        """One memory in full, with its metadata and graph."""
        return get_store().get(id)

    @mcp.tool(annotations=ro)
    def memory_list(scope: str | None = None, kind: str | None = None,
                    status: str = "active", limit: int = 50) -> dict[str, Any]:
        """List memories of a scope (plus global), newest first. status: active |
        provisional. Use when the user asks what you remember."""
        store = get_store()
        store.sync()
        sd = scope_dir_name(scope_key(scope))
        scopes = ["global"] if sd == "global" else [sd, "global"]
        kinds = [kind] if kind in KINDS else None
        rows = sorted(store.rows(scopes, kinds, (status,)),
                      key=lambda r: r["last_reinforced"] or "", reverse=True)
        return {"scope": scope_key(scope), "count": len(rows),
                "memories": [{"id": r["id"], "kind": r["kind"], "status": r["status"],
                              "description": r["description"],
                              "last_reinforced": r["last_reinforced"]}
                             for r in rows[:max(1, min(limit, 500))]]}

    @mcp.tool(annotations=ro)
    def memory_stats() -> dict[str, Any]:
        """Counts per scope, kind and status; where the data lives; whether vector search
        is on."""
        store = get_store()
        store.sync()
        return store.stats()

    return mcp


_STORE: Store | None = None


def _is_loopback(host: str) -> bool:
    import ipaddress
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
        _STORE.maintain()
        _STORE.sync()
    return _STORE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="halflife-mcp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sv = sub.add_parser("serve", help="serve over stdio (default) or local HTTP")
    sv.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--allow-network", action="store_true",
                    help="allow a non-loopback --host (anyone who can reach it can read and "
                         "change your memory: there is no authentication)")
    sub.add_parser("stats")
    b = sub.add_parser("brief")
    b.add_argument("--scope")
    r = sub.add_parser("recall")
    r.add_argument("query")
    r.add_argument("--scope")
    r.add_argument("-k", type=int, default=5)
    sub.add_parser("maintain")
    sub.add_parser("rebuild")
    a = p.parse_args(argv)
    if a.cmd in (None, "serve"):
        if not getattr(a, "http", False):
            create_server(get_store).run("stdio")
            return 0
        if not _is_loopback(a.host) and not a.allow_network:
            print(f"refusing to listen on {a.host}: the HTTP mode has no authentication. "
                  "Use the default 127.0.0.1, or pass --allow-network if you really mean it.",
                  file=sys.stderr)
            return 2
        print(f"halflife-mcp: http://{a.host}:{a.port}/mcp (data: {get_store().home})",
              file=sys.stderr)
        create_server(get_store).run("streamable-http", host=a.host, port=a.port)
        return 0
    if a.cmd == "rebuild":
        st = Store()
        st.db.close()
        for suffix in ("", "-wal", "-shm"):
            path = os.path.join(st.home, "index.sqlite3" + suffix)
            if os.path.exists(path):
                os.remove(path)
        out = Store().sync()
    elif a.cmd == "stats":
        out = get_store().stats()
    elif a.cmd == "brief":
        out = _recall.brief(get_store(), a.scope)
    elif a.cmd == "recall":
        out = _recall.recall(get_store(), a.query, a.scope, a.k)
    else:
        out = Store().maintain()
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
