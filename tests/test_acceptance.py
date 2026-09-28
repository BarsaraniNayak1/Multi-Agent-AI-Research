from __future__ import annotations

from research_copilot.crawler import crawl_urls, extract_clean_text
from research_copilot.llm import GroqJSONClient, call_json_or_fallback
from research_copilot.pipeline import _cite_document, run_research_pipeline
from research_copilot.store import ProjectStore
from research_copilot.vector_store import LocalVectorStore


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(_self):
        if _self.status_code >= 400:
            raise RuntimeError("http error")


def test_clean_extraction_removes_scripts_and_normalizes_whitespace():
    text = extract_clean_text("<h1>Title</h1><script>bad()</script>  useful   text")
    assert text == "Title useful text"


def test_crawler_respects_domain_depth_and_persists(monkeypatch, tmp_path):
    pages = {
        "https://site.test/": '<main>Home</main><a href="/next">Next</a><a href="https://other.test/no">Other</a>',
        "https://site.test/next": "<main>Next page</main>",
    }

    monkeypatch.setattr("research_copilot.crawler.robotparser.RobotFileParser.read", lambda _self: None)
    monkeypatch.setattr("research_copilot.crawler.robotparser.RobotFileParser.can_fetch", lambda _self, _agent, _url: True)
    monkeypatch.setattr("research_copilot.crawler.requests.get", lambda url, **_kwargs: FakeResponse(pages[url]))
    store = LocalVectorStore("crawl", tmp_path)

    documents = crawl_urls(["https://site.test/"], "web", max_depth=1, storage=store)

    assert {document["url"] for document in documents} == {"https://site.test/", "https://site.test/next"}
    assert all(document["storage_id"] for document in documents)
    assert {item["id"] for item in store.query("Home", record_type="source_document")} >= {
        documents[0]["storage_id"]
    }


def test_crawler_reports_http_failure(monkeypatch):
    monkeypatch.setattr("research_copilot.crawler.robotparser.RobotFileParser.read", lambda _self: None)
    monkeypatch.setattr("research_copilot.crawler.robotparser.RobotFileParser.can_fetch", lambda _self, _agent, _url: True)
    monkeypatch.setattr("research_copilot.crawler.requests.get", lambda *_args, **_kwargs: FakeResponse("", 403))
    errors = []

    documents = crawl_urls(["https://source.test/private"], "web", errors=errors)

    assert documents == []
    assert errors == ["HTTP 403 while fetching https://source.test/private."]


def test_invalid_seed_url_is_reported():
    try:
        crawl_urls(["not-a-url"], "web")
    except ValueError as error:
        assert "Invalid seed URL" in str(error)
    else:
        raise AssertionError("invalid URL should be rejected")


def test_vector_retrieval_filters_context_notes(tmp_path):
    store = LocalVectorStore("typed", tmp_path)
    store.upsert({"id": "doc", "text": "market evidence", "source_type": "web"})
    store.upsert_context_note("note", "market context")

    assert [item["id"] for item in store.query("market", record_type="source_document")] == ["doc"]
    assert [item["id"] for item in store.query("market", record_type="context_note")] == ["note"]


def test_malformed_or_unavailable_llm_uses_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    result = call_json_or_fallback("system", "user", lambda: {"ok": True})
    assert result == {"ok": True}


def test_citation_fallback_contains_url_and_access_date():
    result = _cite_document({"id": "1", "url": "https://source.test", "text": ""}, GroqJSONClient())
    assert "https://source.test" in result["apa"]
    assert "https://source.test" in result["mla"]
    assert result["accessed_at"]


def test_pause_state_can_resume_without_re_crawling(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("RESEARCH_HUMAN_REVIEW_THRESHOLD", "0")
    monkeypatch.setattr(
        "research_copilot.pipeline.crawl_urls",
        lambda *_args, **_kwargs: [{"id": "doc-1", "url": "https://source.test", "text": "review me", "source_type": "web"}],
    )

    first = run_research_pipeline("resume-project", [], "web", human_review=True)
    assert first["status"] == "paused"
    assert (tmp_path / ".research_projects" / "resume-project" / "run_state.json").exists()

    second = run_research_pipeline("resume-project", ["https://should-not-run.test"], "web", resume=True, human_decisions={})
    assert second["status"] == "paused"
    third = run_research_pipeline("resume-project", [], "web", resume=True, human_decisions={"doc-1": "approve"})
    assert third["status"] == "completed"
    assert "stage_durations" in second
    assert third["review_queue"] == []
    saved_document = ProjectStore("resume-project").load_document("doc-1")
    assert saved_document["review"]["human_decision"] == "approved"
    assert saved_document["review"]["flagged_for_human_review"] is False


def test_store_updates_document_idempotently(tmp_path):
    store = ProjectStore("idempotent", tmp_path)
    store.save_document("same", {"id": "same", "text": "old"})
    store.save_document("same", {"id": "same", "text": "new"})
    assert store.load_document("same")["text"] == "new"
    assert len(store.list_documents()) == 1


def test_web_run_payload_validation():
    from research_copilot.web import _validate_run_payload

    valid = _validate_run_payload({
        "project_id": "web-demo",
        "urls": ["https://example.com"],
        "source_type": "web",
    })
    assert valid["urls"] == ["https://example.com"]

    invalid_payloads = [
        {"project_id": "../outside", "urls": ["https://example.com"]},
        {"project_id": "web-demo", "urls": ["not-a-url"]},
        {"project_id": "web-demo", "urls": []},
    ]
    for payload in invalid_payloads:
        try:
            _validate_run_payload(payload)
        except ValueError:
            continue
        raise AssertionError(f"invalid dashboard payload was accepted: {payload}")