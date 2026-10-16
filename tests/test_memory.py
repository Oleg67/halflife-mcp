import json
import os
from datetime import date, timedelta

import pytest

from halflife_mcp import memfile
from halflife_mcp.recall import brief, recall
from halflife_mcp.store import MemoryError_, Store, decayed, scope_key

CONCEPTS = [("kubernetes", "k8s", "cluster", "pod"), ("postgres", "database", "sql"),
            ("deploy", "release", "ship"), ("python", "pip", "uv")]


def fake_embed(texts):
    """Deterministic 'semantic' vectors: one dimension per concept group (+ a small bias)."""
    return [[sum(t.lower().count(w) for w in grp) for grp in CONCEPTS] + [0.05] for t in texts]


fake_embed.model = "fake"


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path), embedder=None)


@pytest.fixture
def vstore(tmp_path):
    return Store(str(tmp_path), embedder=fake_embed)


def files(store, sd="global"):
    d = os.path.join(store.root, sd)
    return sorted(f for f in os.listdir(d) if f.endswith(".md")) if os.path.isdir(d) else []


# ------------------------------------------------------------------ store
def test_remember_writes_file_and_index(store):
    r = store.remember("preference", "Use uv instead of pip for Python deps.\n**Why:** speed",
                       description="prefers uv over pip for python dependencies")
    assert r["action"] == "created" and r["id"] == "global/prefers-uv-over-pip-for-python-dependencies"
    meta, body = memfile.parse(open(os.path.join(store.root, r["id"] + ".md")).read())
    assert (meta["kind"], meta["tier"], meta["status"], meta["confidence"]) == \
        ("preference", "preference", "active", "0.7")
    assert body.startswith("Use uv")
    assert store.stats()["memories"] == 1


def test_fact_uses_semantic_tier_for_file_compat(store):
    r = store.remember("fact", "Postgres listens on 5433 locally", scope="global")
    meta, _ = memfile.parse(open(os.path.join(store.root, r["id"] + ".md")).read())
    assert (meta["kind"], meta["tier"]) == ("fact", "semantic")
    assert store.get(r["id"])["kind"] == "fact"


def test_unknown_kind_empty_and_long_rejected(store):
    with pytest.raises(MemoryError_):
        store.remember("opinion", "x")
    with pytest.raises(MemoryError_):
        store.remember("fact", "   ")
    with pytest.raises(MemoryError_):
        store.remember("fact", "x" * 5000)


@pytest.mark.parametrize("secret", [
    "api_key=abcdef123456", "Authorization: Bearer abcdefghijklmnopqrstu",
    "token: sk-proj-abcdefghijklmnopqrstuv", "https://bob:hunter2@example.com/x",
    "-----BEGIN RSA PRIVATE KEY-----", "AKIAABCDEFGHIJKLMNOP", "password: hunter22"])
def test_secrets_refused_everywhere(store, secret):
    with pytest.raises(MemoryError_, match="secret"):
        store.remember("fact", f"note {secret}")
    with pytest.raises(MemoryError_, match="secret"):
        store.remember("fact", "harmless", entities=[{"name": secret}])
    assert files(store) == []


def test_near_duplicate_reinforces(store):
    a = store.remember("preference", "user prefers uv over pip for python dependencies")
    b = store.remember("preference", "user prefers uv over plain pip for python dependencies")
    assert b["action"] == "reinforced" and b["id"] == a["id"]
    assert len(files(store)) == 1
    assert store.get(a["id"])["reinforced"] == 1


def test_provisional_hidden_until_reinforced(vstore):
    r = vstore.remember("preference", "user likes terse answers without headers",
                        provisional=True)
    assert r["status"] == "provisional" and r["confidence"] == 0.3
    assert recall(vstore, "terse answers without headers") == []
    assert brief(vstore)["provisional_pending"] == 1
    out = vstore.reinforce(r["id"])
    assert (out["status"], out["was"], out["confidence"]) == ("active", "provisional", 0.7)
    assert recall(vstore, "terse answers without headers")[0]["id"] == r["id"]


def test_same_day_inference_does_not_self_corroborate(store):
    a = store.remember("fact", "the build uses gradle not maven", provisional=True)
    b = store.remember("fact", "the build uses gradle not maven", provisional=True)
    assert b["action"] == "unchanged"
    assert store.get(a["id"])["status"] == "provisional"
    c = store.remember("fact", "the build uses gradle not maven")     # stated explicitly
    assert c["action"] == "reinforced" and store.get(a["id"])["status"] == "active"


def test_contradicts_supersedes_and_archives(store):
    old = store.remember("fact", "postgres listens on port 5433", scope="/tmp/proj-x")
    new = store.remember("fact", "postgres moved to port 5544", scope="/tmp/proj-x",
                         contradicts=old["id"])
    assert new["superseded"] == old["id"]
    sd = old["id"].split("/")[0]
    assert files(store, sd) == [new["id"].split("/")[1] + ".md"]
    arch = os.path.join(store.root, sd, "_archive", old["id"].split("/")[1] + ".md")
    meta, body = memfile.parse(open(arch).read())
    assert (meta["status"], meta["superseded_by"]) == ("superseded", new["id"])
    assert "5433" in body                                              # text kept


def test_provisional_contradiction_waits_for_confirmation(store):
    old = store.remember("fact", "deploys happen on fridays")
    new = store.remember("fact", "deploys now happen on tuesdays", contradicts=old["id"],
                         provisional=True)
    assert "superseded" not in new
    assert store.get(old["id"])["status"] == "active"
    assert store.reinforce(new["id"])["superseded"] == old["id"]
    with pytest.raises(MemoryError_):
        store.get(old["id"])


def test_forget_deletes_everything(store):
    r = store.remember("fact", "redis caches sessions", entities=[{"name": "redis",
                                                                   "type": "service"}])
    sd, nm = r["id"].split("/")
    assert os.path.exists(os.path.join(store.root, sd, "_graph", nm + ".json"))
    assert store.forget(r["id"])["deleted_files"] == 2
    assert files(store) == [] and store.stats()["memories"] == 0
    assert store.stats()["entities"] == 0
    with pytest.raises(MemoryError_):
        store.forget(r["id"])


def test_bad_ids_rejected(store):
    for bad in ("../etc/passwd", "global/../../x", "noslash", "a/b/c", ""):
        with pytest.raises(MemoryError_):
            store.get(bad)


def test_scope_normalizes_to_git_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src" / "pkg").mkdir(parents=True)
    assert scope_key(str(repo / "src" / "pkg")) == str(repo)
    assert scope_key(None) == scope_key("GLOBAL") == "global"
    assert scope_key("my-project") == "my-project"


def test_project_recall_includes_global_not_other_projects(store):
    store.remember("procedural", "run the test suite with make test before committing",
                   scope="/tmp/alpha")
    store.remember("procedural", "run the test suite with tox before committing",
                   scope="/tmp/beta")
    store.remember("preference", "commit messages start with a ticket id before committing")
    ids = [h["id"] for h in recall(store, "run the test suite before committing",
                                   scope="/tmp/alpha")]
    assert any(i.startswith("tmp-alpha/") for i in ids)
    assert not any(i.startswith("tmp-beta/") for i in ids)


def test_manual_edits_and_deletes_are_picked_up(store):
    r = store.remember("fact", "the api gateway runs on port 8080")
    path = os.path.join(store.root, r["id"] + ".md")
    text = open(path).read().replace("8080", "9090")
    open(path, "w").write(text)
    os.utime(path, (os.path.getmtime(path) + 5,) * 2)
    assert "9090" in recall(store, "api gateway port")[0]["content"]
    os.remove(path)
    assert recall(store, "api gateway port") == []


def test_hand_written_claude_code_file_is_readable(store):
    os.makedirs(os.path.join(store.root, "global"), exist_ok=True)
    open(os.path.join(store.root, "global", "feedback_x.md"), "w").write(
        '---\nname: feedback-x\ndescription: "always rebase before merging feature branches"\n'
        "metadata:\n  type: feedback\n  tier: procedural\n  confidence: 0.85\n---\n\nbody\n")
    hit = recall(store, "rebase before merging")[0]
    assert (hit["id"], hit["kind"]) == ("global/feedback_x", "procedural")


def test_maintain_archives_faded_and_expires_traces(store):
    r = store.remember("episodic", "worked on the login flow refactor")
    t = store.remember("trace", "ran pytest tests/test_login.py: 3 failed")
    old = (date.today() - timedelta(days=400)).isoformat()
    for mem_id in (r["id"], t["id"]):
        p = os.path.join(store.root, mem_id + ".md")
        text = open(p).read()
        open(p, "w").write(text.replace(date.today().isoformat(), old))
    out = store.maintain()
    assert out == {"archived": [r["id"]], "expired_traces": [t["id"]]}
    assert os.path.exists(os.path.join(store.root, "global", "_archive",
                                       r["id"].split("/")[1] + ".md"))
    assert files(store) == []


def test_decay_half_life():
    last = (date.today() - timedelta(days=30)).isoformat()
    assert decayed("episodic", 1.0, last) == pytest.approx(0.5)
    assert decayed("procedural", 0.8, date.today().isoformat()) == pytest.approx(0.8)


# ------------------------------------------------------------------ recall
def test_short_or_unrelated_queries_return_nothing(store):
    store.remember("fact", "graph storage tests need the gss conda env")
    assert recall(store, "go ahead") == []
    assert recall(store, "plan a birthday party on mars") == []


def test_trace_excluded_by_default(store):
    store.remember("trace", "edited src/login.py and ran the login tests")
    assert recall(store, "login tests src/login.py") == []
    assert recall(store, "login tests src/login.py", kinds=["trace"])


def test_recall_is_read_only(store):
    r = store.remember("fact", "the staging database is refreshed nightly")
    before = store.get(r["id"])
    for _ in range(3):
        recall(store, "staging database refresh")
    after = store.get(r["id"])
    assert (after["confidence"], after["reinforced"]) == (before["confidence"], before["reinforced"])


def test_turning_on_vectors_embeds_existing_memories(tmp_path):
    plain = Store(str(tmp_path), embedder=None)
    plain.remember("fact", "the k8s cluster runs three pods")
    assert plain.stats()["vectors"] == 0
    plain.db.close()
    vec = Store(str(tmp_path), embedder=fake_embed)
    assert vec.sync()["embedded"] == 1
    assert recall(vec, "restart every kubernetes workload tonight")[0]["signals"] == ["vector"]


def test_vector_finds_paraphrase(vstore):
    vstore.remember("fact", "the k8s cluster runs three pods")
    vstore.remember("fact", "postgres database tuning notes")
    hits = recall(vstore, "restart every kubernetes workload tonight")
    assert [h["id"] for h in hits] == ["global/the-k8s-cluster-runs-three-pods"]
    assert hits[0]["signals"] == ["vector"]


def test_keyword_hit_without_semantic_support_dropped(vstore):
    vstore.remember("fact", "postgres database sql notes: deploy cluster slides")
    assert recall(vstore, "deploy kubernetes cluster tonight") == []


def test_graph_finds_named_entity_and_neighbor(store):
    store.remember("fact", "settings live in one yaml",
                   entities=[{"name": "config/app.yaml", "type": "file"},
                             {"name": "Pydantic", "type": "library"}],
                   relations=[{"src": "Pydantic", "rel": "configures", "dst": "config/app.yaml"}])
    store.remember("fact", "validation errors are verbose", entities=[{"name": "Pydantic",
                                                                       "type": "library"}])
    hits = {h["id"]: h for h in recall(store, "open config/app.yaml quickly")}
    assert "graph:config/app.yaml" in hits["global/settings-live-in-one-yaml"]["signals"]
    g = store.graph_scores("open config/app.yaml quickly", ["global"])
    assert g["global/settings-live-in-one-yaml"] == (1.5, ["config/app.yaml"])  # seed + neighbor
    assert g["global/validation-errors-are-verbose"] == (0.5, [])             # 1-hop neighbor


def test_hub_entities_not_expanded(store):
    store.remember("fact", "langfuse traces go missing in the UI",
                   entities=[{"name": "Langfuse"}, {"name": "engine"}],
                   relations=[{"src": "engine", "rel": "uses", "dst": "Langfuse"}])
    for topic in ("billing invoices", "dark mode colors", "keyboard shortcuts", "font sizes"):
        store.remember("fact", f"notes about {topic}", entities=[{"name": "engine"}])
    hits = recall(store, "langfuse is down again")
    assert [h["id"] for h in hits] == ["global/langfuse-traces-go-missing-in-the-ui"]


def test_signals_fuse(vstore):
    vstore.remember("fact", "k8s cluster upgrade procedure for pods",
                    entities=[{"name": "k8s", "type": "tool"}])
    vstore.remember("fact", "kubernetes is mentioned here too")
    top = recall(vstore, "k8s cluster upgrade")[0]
    assert top["signals"] == ["bm25", "vector", "graph:k8s"]


def test_brief_groups_and_limits(store):
    store.remember("preference", "answers in english, short")
    store.remember("procedural", "always run the linter before pushing code")
    for i in range(5):
        store.remember("episodic", f"session {i}: worked on feature{i} implementation{i}")
    b = brief(store)
    assert list(b["memories"]) == ["preference", "procedural", "episodic"]
    assert len(b["memories"]["episodic"]) == 3


# ------------------------------------------------------------------ MCP
@pytest.mark.anyio
async def test_mcp_tools_end_to_end(tmp_path):
    from mcp import Client

    from halflife_mcp.server import create_server

    st = Store(str(tmp_path), embedder=None)
    async with Client(create_server(lambda: st)) as client:
        assert "Long-term memory" in (client.instructions or "")
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"memory_brief", "memory_recall", "memory_remember", "memory_reinforce",
                         "memory_forget", "memory_get", "memory_list", "memory_stats"}
        r = await client.call_tool("memory_remember", {
            "kind": "procedural", "content": "deploy with make release after tagging",
            "scope": "/tmp/proj-z"})
        mem_id = r.structured_content["id"]
        rec = await client.call_tool("memory_recall", {"query": "how do I deploy a release",
                                                       "scope": "/tmp/proj-z"})
        assert rec.structured_content["results"][0]["id"] == mem_id
        b = await client.call_tool("memory_brief", {"scope": "/tmp/proj-z"})
        assert b.structured_content["memories"]["procedural"][0]["id"] == mem_id
        bad = await client.call_tool("memory_remember", {"kind": "fact",
                                                         "content": "password: hunter22"})
        assert bad.is_error
        assert (await client.call_tool("memory_forget", {"id": mem_id})).structured_content[
            "deleted_files"] == 1
        lst = await client.call_tool("memory_list", {"scope": "/tmp/proj-z"})
        assert lst.structured_content["count"] == 0


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_http_mode_refuses_non_loopback_without_flag(capsys):
    from halflife_mcp.server import _is_loopback, main

    assert _is_loopback("127.0.0.1") and _is_loopback("::1") and _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0") and not _is_loopback("192.168.1.5")
    assert main(["serve", "--http", "--host", "0.0.0.0"]) == 2
    assert "no authentication" in capsys.readouterr().err


def test_two_letter_terms_are_searchable(store):
    store.remember("preference", "Use uv, not pip, for Python environments.")
    assert [h["id"] for h in recall(store, "uv or pip?")] == ["global/use-uv-not-pip-for-python-environments"]


@pytest.mark.parametrize("note", [
    "In CI, GITHUB_TOKEN: provided automatically by GitHub Actions.",
    "Run the eval with --api-key=$OPENAI_API_KEY from .env.local.",
    "Set max_tokens = 100000 for long outputs.",
    "The CSRF cookie: SameSite=Lax broke the login redirect.",
    "OAuth token_endpoint: https://auth.example.com/oauth/token"])
def test_harmless_notes_about_secrets_are_stored(store, note):
    assert store.remember("fact", note)["action"] == "created"


def test_new_memory_does_not_reuse_an_archived_name(store):
    old = store.remember("procedural", "Deploy the service with make deploy.", name="deploy")
    store.remember("procedural", "Release by pushing a version tag; CI releases.",
                   contradicts=old["id"])
    new = store.remember("fact", "Production deploys need an approved change ticket.", name="deploy")
    assert new["id"] != old["id"]
    store.forget(new["id"])
    assert files(store, "global/_archive") == ["deploy.md"]


def test_archiving_never_overwrites_an_archived_memory(store):
    a = store.remember("fact", "first claim about the build cache", name="cache")
    store.remember("fact", "replacement one for the cache claim", contradicts=a["id"])
    path = os.path.join(store.root, "global", "cache.md")
    with open(path, "w", encoding="utf-8") as fh:        # a hand-made file reusing the name
        fh.write("---\nname: cache\ndescription: \"second claim\"\nmetadata:\n  tier: semantic\n---\n\nbody\n")
    store.sync()
    store.remember("fact", "replacement two for the cache claim", contradicts="global/cache")
    assert files(store, "global/_archive") == ["cache-2.md", "cache.md"]
