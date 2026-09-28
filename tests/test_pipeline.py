from research_copilot.pipeline import analyze_trends, run_research_pipeline
from research_copilot.store import ProjectStore


def test_project_store_keeps_context_oldest_first(tmp_path):
    store = ProjectStore(project_id="proj-1", root_dir=tmp_path)
    store.add_context_note("first note")
    store.add_context_note("second note")

    notes = store.load_context_notes()
    assert [note["content"] for note in notes] == ["first note", "second note"]


def test_empty_trend_analysis_returns_empty_structure():
    result = analyze_trends([], project_id="proj-1")
    assert result["topics"] == []
    assert result["grounded_documents"] == 0
    assert result["narrative"] == "No historical signals available yet."


def test_end_to_end_pipeline_runs_with_seed_urls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr("research_copilot.pipeline.crawl_urls", lambda *_args, **_kwargs: [])
    result = run_research_pipeline(
        project_id="proj-2",
        urls=["https://example.com"],
        source_type="web",
        human_review=False,
    )

    assert "documents" in result
    assert isinstance(result["documents"], list)
    assert result["status"] in {"completed", "paused"}
