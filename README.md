# halflife-mcp

Local long-term memory for AI agents, served over MCP. It runs on your own laptop: no account,
no cloud service, no telemetry. Works with any MCP client that can launch a stdio server, such as
Claude Code, Claude Desktop, Cursor, Zed or VS Code.

The agent gets five kinds of memory:

| Kind | Holds | Fades (half-life) |
|---|---|---|
| `preference` | how you like things done ("use uv, not pip") | 180 days |
| `procedural` | workflows that worked ("release = tag, then `make release`") | 365 days |
| `fact` | durable facts not in the code or docs | 180 days |
| `episodic` | what happened in a past task: goal, decisions, open items | 30 days |
| `trace` | short-lived log of files/commands, for resuming work | 7 days, deleted after 30 |

Memories you use get reinforced and stay. Unused ones fade and are archived: moved, never
silently deleted. Inferred memories stay *provisional* (hidden) until confirmed. A correction
*supersedes* the old memory instead of contradicting it.

## Install

You need [uv](https://docs.astral.sh/uv/). Then add the server to your client.

**Claude Code**

```bash
claude mcp add memory --scope user -- uvx halflife-mcp
```

**Claude Desktop** (Settings → Developer → Edit Config → `claude_desktop_config.json`), Cursor
(`~/.cursor/mcp.json`) and most other clients use the same shape:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["halflife-mcp"]
    }
  }
}
```

If the client can't find `uvx`, put its absolute path in `command` (`which uvx`).

### Local HTTP mode

For clients that connect by URL instead of launching a command, run the server yourself:

```bash
uvx halflife-mcp serve --http  # http://127.0.0.1:8765/mcp
claude mcp add --transport http memory http://127.0.0.1:8765/mcp  # e.g. Claude Code
```

There is no authentication, so it listens on loopback only: other machines can't reach it.
Requests whose `Host` or `Origin` isn't local are rejected, which stops a web page from
talking to it. `--port` changes the port. A non-loopback `--host` is refused unless you pass
`--allow-network`; don't use that flag on a network you don't control.

**claude.ai (web/mobile) can't use this.** Its custom connectors are called from Anthropic's
servers, so they can't reach your `localhost`. They need a public HTTPS URL with
authentication, which this project doesn't provide.

### Optional: vector search

The default install is light: keyword (BM25) and graph search, with no ML dependencies. For
semantic matching (finding "k8s workloads" from a memory about "the kubernetes cluster"), use
the `vector` extra:

```json
"args": ["--with", "fastembed>=0.4", "--with", "numpy>=1.24", "halflife-mcp"]
```

This adds fastembed (ONNX, CPU only). The first start downloads a 65 MB model
(`BAAI/bge-small-en-v1.5`); after that it runs offline. Existing memories are embedded
automatically.

## Tools

| Tool | What it does |
|---|---|
| `memory_brief(scope?)` | Start-of-task pack: top preferences and procedures, the strongest facts, recent episodes (capped at 8/6/5/3) |
| `memory_recall(query, scope?, k?, kinds?)` | Hybrid search: keyword + vector + graph, fused by reciprocal-rank fusion. Read-only |
| `memory_remember(kind, content, scope?, description?, provisional?, contradicts?, entities?, relations?, source?)` | Store a memory; a near-duplicate reinforces the existing one instead |
| `memory_reinforce(id)` | A memory was confirmed or useful: confidence up, decay reset, provisional → active |
| `memory_forget(id)` | Permanently delete (file, graph, archived copy, index) |
| `memory_get(id)` / `memory_list(scope?, kind?, status?)` / `memory_stats()` | Inspect |

The server sends the model instructions on when to use each tool. It should call brief and
recall at the start of a task, remember preferences when you state them, write an episode at
the end, and forget on request. Memory only works as well as the model follows them, because
an MCP server can't see the conversation. If memory isn't being used, tell the agent "check
your memory" or "remember that…".

**Scope.** Pass the project path as `scope` and it is normalized to the git root, so
subfolders share one project memory. Omit it, or use `global`, for things that apply
everywhere. Project queries include global memories but not other projects' memories. Scope
keeps memory organized; it is not access control: any connected client can ask for any scope,
and project paths that differ only in punctuation (`my-app`, `my_app`) share one scope.

## Where your data lives

| OS | Default location |
|---|---|
| Linux | `~/.local/share/halflife` (or `$XDG_DATA_HOME/halflife`) |
| macOS | `~/Library/Application Support/halflife` |
| Windows | `%APPDATA%\halflife` |

Set `HALFLIFE_HOME` to put it elsewhere, for example a synced folder.

```
memories/<scope>/<name>.md            one memory per Markdown file: readable, editable, greppable
memories/<scope>/_graph/<name>.json   its entities and relations
memories/<scope>/_archive/            faded and superseded memories
index.sqlite3                         derived index (SQL, vectors, graph); rebuildable
models/                               embedding model (vector extra only)
```

The Markdown files are the source of truth. Edit or delete them by hand and the index catches
up on the next call. `halflife-mcp rebuild` recreates the index from the files.

## Privacy

- Memories are stored only on your machine, and the server makes no network calls apart from
  the one-time model download with the vector extra. But whatever the agent recalls goes into
  its prompt, so a client that uses a cloud model sends those memories to its provider.
- Writes that look like secrets are **refused**, not redacted. That covers API keys, tokens,
  passwords, private keys, `user:pass@` URLs and JWTs.
- `memory_forget` deletes for real, including archived copies.
- The data directory is created with mode 0700.

## CLI

```bash
uvx halflife-mcp stats              # what's stored, where
uvx halflife-mcp brief --scope ~/code/my-app
uvx halflife-mcp recall "how do we deploy" --scope ~/code/my-app
uvx halflife-mcp maintain           # archive faded, expire traces
uvx halflife-mcp rebuild            # rebuild the index from files
```

## Configuration

| Variable | Default | |
|---|---|---|
| `HALFLIFE_HOME` | per-OS path above | data directory |
| `HALFLIFE_EMBED` | on (if the extra is installed) | `off` disables vector search |
| `HALFLIFE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | another fastembed model (recall thresholds are tuned for the default) |

## Development

```bash
uv venv && uv pip install -e ".[vector]" pytest anyio
.venv/bin/python -m pytest -q
```

`uvx --from <local path>` caches the built package. After changing the code, run it once with
`uvx --no-cache --from …`, or use the editable venv's `.venv/bin/halflife-mcp` as the
client command.

## License

Apache License 2.0; see the `LICENSE` file.
