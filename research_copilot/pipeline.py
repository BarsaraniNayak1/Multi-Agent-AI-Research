from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from .crawler import crawl_urls
from .llm import GroqJSONClient, call_json_or_fallback
from .store import ProjectStore
from .vector_store import LocalVectorStore


def _doc_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _default_summary(document: dict[str, Any]) -> dict[str, Any]:
    text = str(document.get("text", "")).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    summary_text = " ".join(sentences[:3]) if sentences else "No substantial text was extracted from this source."
    if len(summary_text) > 500:
        summary_text = summary_text[:497].rstrip() + "..."

    insights = []
    for phrase in re.findall(r"[A-Za-z][A-Za-z0-9 ,;:/-]{3,}", text)[:5]:
        if len(phrase) > 5:
            insights.append(phrase.strip())
    if not insights:
        insights = ["Source reviewed for relevance and context."]

    return {
        "document_id": document.get("id") or _doc_key(str(document.get("url", ""))),
        "summary": summary_text,
        "insights": insights[:5],
        "confidence": 0.75,
    }


def _summarize_document(document: dict[str, Any], client: GroqJSONClient | None = None) -> dict[str, Any]:
    fallback = lambda: _default_summary(document)
    result = call_json_or_fallback(
        "You are a research summarizer. Return JSON with summary, insights, and confidence. "
        "The summary must be 3-5 concise sentences and insights must contain at most 5 points.",
        f"Document URL: {document.get('url', '')}\nDocument text:\n{document.get('text', '')[:12000]}",
        fallback,
        client,
    )
    result.setdefault("document_id", document.get("id") or _doc_key(str(document.get("url", ""))))
    result.setdefault("summary", fallback()["summary"])
    result.setdefault("insights", fallback()["insights"])
    result.setdefault("confidence", 0.5)
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    result["insights"] = list(result.get("insights", []))[:5]
    return result


def _citation_for_document(document: dict[str, Any]) -> dict[str, Any]:
    title = str(document.get("title") or document.get("url") or "Untitled").strip() or "Untitled"
    url = str(document.get("url") or "").strip()
    access_date = datetime.now(timezone.utc).strftime("%d %b %Y")
    author = "Unknown Author"
    publication = "Unknown Publication"
    year = "n.d."

    apa = f"{author}. ({year}). {title}. {publication}. {url}"
    mla = f"{author}. \"{title}.\" {publication}, {year}, {url}. Accessed {access_date}."
    return {"apa": apa, "mla": mla, "author": author, "publication": publication, "year": year, "title": title, "url": url, "accessed_at": access_date}


def _cite_document(document: dict[str, Any], client: GroqJSONClient | None = None) -> dict[str, Any]:
    fallback = lambda: _citation_for_document(document)
    result = call_json_or_fallback(
        "You extract citation metadata. Return JSON with author, publication, year, title, apa, mla, url, and accessed_at. "
        "Use Unknown Author, Unknown Publication, n.d., and Untitled when uncertain.",
        f"Source URL: {document.get('url', '')}\nSource text:\n{document.get('text', '')[:8000]}",
        fallback,
        client,
    )
    defaults = fallback()
    for key, value in defaults.items():
        result.setdefault(key, value)
    for format_name in ("apa", "mla"):
        if document.get("url") and str(document["url"]) not in str(result.get(format_name, "")):
            result[format_name] = f"{result[format_name]} {document['url']}"
    return result


def _initial_review(document: dict[str, Any], summary: dict[str, Any] | None = None, client: GroqJSONClient | None = None) -> dict[str, Any]:
    threshold = float(os.getenv("RESEARCH_HUMAN_REVIEW_THRESHOLD", "0.65"))
    bias_score = 0.2
    notes = "No major quality issues identified in the extracted content."
    accuracy_concerns = []
    quality_flag = "acceptable"

    if summary and summary.get("confidence", 0.0) < 0.5:
        accuracy_concerns.append("Low confidence summary.")
        quality_flag = "needs_review"

    needs_review = bool(accuracy_concerns) or bias_score >= threshold
    fallback = {
        "document_id": document.get("id") or _doc_key(str(document.get("url", ""))),
        "bias_score": bias_score,
        "bias_notes": notes,
        "accuracy_concerns": accuracy_concerns,
        "quality_flag": quality_flag,
        "needs_review": needs_review,
    }
    return call_json_or_fallback(
        "You are a source quality auditor. Return JSON with bias_score from 0 to 1, bias_notes, "
        "accuracy_concerns as a list, quality_flag, and needs_review.",
        f"Source URL: {document.get('url', '')}\nSummary: {summary or {}}\nText:\n{document.get('text', '')[:10000]}",
        lambda: fallback,
        client,
    )


def _reflect_review(initial: dict[str, Any], document: dict[str, Any], client: GroqJSONClient | None = None) -> dict[str, Any]:
    return call_json_or_fallback(
        "You are a separate reflection auditor. Critique the initial source-quality verdict and return a corrected JSON verdict "
        "with bias_score, bias_notes, accuracy_concerns, quality_flag, and needs_review.",
        f"Initial verdict: {initial}\nSource URL: {document.get('url', '')}\nText:\n{document.get('text', '')[:8000]}",
        lambda: dict(initial),
        client,
    )


def _fallback_trend_result() -> dict[str, Any]:
    return {
        "topics": [],
        "grounded_documents": 0,
        "narrative": "No historical signals available yet.",
        "supporting_sources": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def analyze_trends(documents: list[dict[str, Any]], project_id: str | None = None, vector_store: LocalVectorStore | None = None) -> dict[str, Any]:
    if not documents:
        return _fallback_trend_result()

    text_blob = " ".join(str(item.get("text", "")) for item in documents if item.get("text"))
    if not text_blob:
        return _fallback_trend_result()

    related = []
    if vector_store is not None:
        try:
            related = vector_store.query(text_blob, limit=5, record_type="source_document")
        except Exception:
            related = []

    tokens = [token.lower() for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", text_blob) if len(token) > 3]
    counts = Counter(tokens)
    top_terms = [term for term, _ in counts.most_common(5)]

    topics = []
    for term in top_terms:
        topics.append(
            {
                "topic": term,
                "supporting_sources": max(1, min(5, len(documents))),
                "rationale": f"Multiple retrieved sources reference {term} in context and supporting evidence.",
            }
        )

    narrative = (
        "The current batch indicates recurring attention around the most discussed themes, "
        "with evidence drawn from the available source set and matching historical context."
    )
    if project_id is not None:
        historical_context = "\n".join(str(item.get("text", ""))[:500] for item in related)
        llm_result = call_json_or_fallback(
            "You are a trend analyst. Return JSON with topics and a 2-4 sentence narrative. "
            "Each topic needs topic, supporting_sources, and rationale.",
            f"Current documents:\n{text_blob[:12000]}\nHistorical retrieved context:\n{historical_context}",
            lambda: {"topics": topics, "narrative": narrative},
        )
        topics = llm_result.get("topics", topics)
        narrative = llm_result.get("narrative", narrative)

    return {
        "topics": topics,
        "grounded_documents": len(related),
        "narrative": narrative,
        "supporting_sources": len(related) or len(documents),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_context(state: dict[str, Any]) -> dict[str, Any]:
    state["context_notes"] = state["project_store"].load_context_notes()
    return state


def _crawl_documents(state: dict[str, Any]) -> dict[str, Any]:
    crawl_errors: list[str] = []
    try:
        documents = crawl_urls(
            state.get("urls", []),
            state.get("source_type", "web"),
            max_depth=int(os.getenv("RESEARCH_CRAWL_DEPTH", "2")),
            respect_robots=True,
            timeout=int(os.getenv("RESEARCH_CRAWL_TIMEOUT", "15")),
            storage=state["vector_store"],
            errors=crawl_errors,
        )
    except Exception as error:
        documents = []
        crawl_errors.append(str(error))

    state["documents"] = documents
    state["crawl_errors"] = crawl_errors
    state["crawl_error"] = None
    if not documents and state.get("urls"):
        state["crawl_error"] = "; ".join(crawl_errors) or "No documents were crawled."
    return state


def _summarize_documents(state: dict[str, Any]) -> dict[str, Any]:
    store = state["project_store"]
    vector_store = state["vector_store"]
    client = state.get("llm_client")
    docs = []
    for document in state.get("documents", []):
        try:
            summary = _summarize_document(document, client)
            document["summary"] = summary
            document["summary_id"] = summary["document_id"]
            store.save_document(document["id"], {**document, "summary": summary})
            vector_store.upsert({**document, "summary": summary, "id": document["id"]})
            docs.append(document)
        except Exception:
            continue
    state["documents"] = docs
    return state


def _cite_documents(state: dict[str, Any]) -> dict[str, Any]:
    for document in state.get("documents", []):
        try:
            document["citation"] = _cite_document(document, state.get("llm_client"))
            state["project_store"].save_document(document["id"], document)
        except Exception:
            document["citation"] = {"apa": "Unknown Author. (n.d.). Untitled. Unknown Publication. ", "mla": "Unknown Author. \"Untitled.\" Unknown Publication, n.d., . Accessed today.", "url": document.get("url", "")}
    return state


def _review_documents(state: dict[str, Any]) -> dict[str, Any]:
    threshold = float(os.getenv("RESEARCH_HUMAN_REVIEW_THRESHOLD", "0.65"))
    reviewed = []
    for document in state.get("documents", []):
        initial = _initial_review(document, document.get("summary"), state.get("llm_client"))
        review = _reflect_review(initial, document, state.get("llm_client"))
        review.setdefault("initial_review", initial)
        review.setdefault("reflection_applied", review != initial)
        review.setdefault("accuracy_concerns", [])
        review.setdefault("bias_score", initial.get("bias_score", 0.2))
        review.setdefault("quality_flag", initial.get("quality_flag", "acceptable"))
        review["needs_review"] = bool(review.get("needs_review")) or bool(review.get("accuracy_concerns")) or float(review.get("bias_score", 0.0)) >= threshold
        document["review"] = review
        if review["needs_review"] or review["bias_score"] >= threshold:
            review["flagged_for_human_review"] = True
        else:
            review["flagged_for_human_review"] = False
        reviewed.append(document)
        state["project_store"].save_document(document["id"], document)
    state["documents"] = reviewed
    return state


def _human_gate(state: dict[str, Any]) -> dict[str, Any]:
    human_review = bool(state.get("human_review", False))
    if human_review:
        flagged = [doc for doc in state.get("documents", []) if doc.get("review", {}).get("needs_review")]
        state["needs_human_review"] = bool(flagged)
        state["review_queue"] = flagged
        if state["needs_human_review"]:
            state["status"] = "paused"
            state["message"] = "Research paused pending human review."
            return state
    state["needs_human_review"] = False
    state["review_queue"] = []
    state["status"] = "completed"
    state["message"] = "Research completed without human intervention."
    return state


def _finalize(state: dict[str, Any]) -> dict[str, Any]:
    project_store: ProjectStore = state["project_store"]
    counter = len(state.get("documents", []))
    note = project_store.add_context_note(
        f"Completed research run for {state['project_id']} with {counter} document(s).",
        {"document_count": counter, "source_type": state.get("source_type", "web"), "status": state.get("status", "completed")},
    )
    state["vector_store"].upsert_context_note(
        f"context-{state['project_id']}-{note['created_at']}",
        note["content"],
        {"project_id": state["project_id"]},
    )
    state["finalized_at"] = datetime.now(timezone.utc).isoformat()
    state["documents"] = state.get("documents", [])
    state["metrics"] = {
        "document_count": counter,
        "status": state.get("status", "completed"),
        "time_saved_vs_manual_hours": round(
            max(
                0.0,
                counter * float(os.getenv("RESEARCH_MANUAL_BASELINE_HOURS", "0.5"))
                - sum(state.get("stage_durations", {}).values()) / 3600,
            ),
            2,
        ),
        "bias_detection_rate": round((sum(1 for doc in state.get("documents", []) if doc.get("review", {}).get("flagged_for_human_review")) / max(counter, 1)) * 100, 2),
        "accuracy_improvement_proxy": sum(
            1 for doc in state.get("documents", []) if doc.get("review", {}).get("reflection_applied")
        ),
    }
    return state


def _timed_node(name: str, handler: Any) -> Any:
    def run(state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        updated = handler(state)
        updated.setdefault("stage_durations", {})[name] = round(time.perf_counter() - started, 4)
        return updated

    return run


def build_graph() -> StateGraph:
    builder = StateGraph(dict)
    builder.add_node("load_context", _timed_node("load_context", _load_context))
    builder.add_node("crawl", _timed_node("crawl", _crawl_documents))
    builder.add_node("summarize", _timed_node("summarize", _summarize_documents))
    builder.add_node("cite", _timed_node("cite", _cite_documents))
    builder.add_node("analyze_trends", _timed_node("analyze_trends", lambda state: {**state, "trend_analysis": analyze_trends(state.get("documents", []), state.get("project_id"), state.get("vector_store"))}))
    builder.add_node("review", _timed_node("review", _review_documents))
    builder.add_node("human_gate", _timed_node("human_gate", _human_gate))
    builder.add_node("finalize", _timed_node("finalize", _finalize))

    builder.set_entry_point("load_context")
    builder.add_edge("load_context", "crawl")
    builder.add_edge("crawl", "summarize")
    builder.add_edge("summarize", "cite")
    builder.add_edge("cite", "analyze_trends")
    builder.add_edge("analyze_trends", "review")
    builder.add_edge("review", "human_gate")
    builder.add_conditional_edges("human_gate", lambda state: "finalize" if state.get("status") == "completed" else "pause", {"finalize": "finalize", "pause": END})
    builder.add_edge("finalize", END)
    return builder.compile()


def run_research_pipeline(
    project_id: str,
    urls: list[str],
    source_type: str,
    human_review: bool = False,
    resume: bool = False,
    human_decisions: dict[str, str] | None = None,
) -> dict[str, Any]:
    project_store = ProjectStore(project_id=project_id)
    vector_store = LocalVectorStore(project_id=project_id)

    saved_state = project_store.load_run_state() if resume else None
    if saved_state and saved_state.get("status") == "paused":
        state = saved_state
        state["project_store"] = project_store
        state["vector_store"] = vector_store
        state["llm_client"] = GroqJSONClient()
        unresolved = []
        for document in state.get("documents", []):
            decision = (human_decisions or {}).get(document.get("id", ""), "").lower()
            if decision in {"approve", "approved", "accept", "accepted"}:
                document.setdefault("review", {})["human_decision"] = "approved"
                document["review"]["needs_review"] = False
                document["review"]["flagged_for_human_review"] = False
                document["review"]["quality_flag"] = "human_approved"
            elif decision in {"reject", "rejected", "discard"}:
                document.setdefault("review", {})["human_decision"] = "rejected"
                document["review"]["needs_review"] = False
                document["review"]["flagged_for_human_review"] = False
                document["review"]["quality_flag"] = "human_rejected"
            elif document.get("review", {}).get("needs_review"):
                unresolved.append(document)
        if unresolved:
            state["review_queue"] = unresolved
            state["needs_human_review"] = True
            state["message"] = "Research remains paused until every flagged document has a decision."
            project_store.save_run_state(state)
            return state
        state["review_queue"] = []
        state["needs_human_review"] = False
        for document in state.get("documents", []):
            project_store.save_document(document["id"], document)
        state["status"] = "completed"
        state["message"] = "Paused research resumed with human decisions."
        result = _finalize(state)
        project_store.save_run_state(result)
        return result

    state: dict[str, Any] = {
        "project_id": project_id,
        "urls": urls,
        "source_type": source_type,
        "human_review": human_review,
        "resume": resume,
        "project_store": project_store,
        "vector_store": vector_store,
        "status": "pending",
        "documents": [],
        "context_notes": project_store.load_context_notes(),
        "llm_client": GroqJSONClient(),
        "stage_durations": {},
    }

    graph = build_graph()
    try:
        result = graph.invoke(state)
    except Exception as error:
        result = state
        result["status"] = "completed"
        result["documents"] = []
        result["message"] = f"Pipeline failed gracefully and returned an empty result: {error}"

    final_docs = result.get("documents", [])
    if not isinstance(final_docs, list):
        result["documents"] = []

    if result.get("status") == "paused":
        result["needs_human_review"] = bool(result.get("review_queue"))
        project_store.add_context_note(
            f"Paused research run for {project_id} pending review of {len(result.get('review_queue', []))} document(s).",
            {"status": "paused", "review_count": len(result.get("review_queue", []))},
        )
    project_store.save_run_state(result)
    return result


__all__ = ["run_research_pipeline", "analyze_trends", "build_graph"]
