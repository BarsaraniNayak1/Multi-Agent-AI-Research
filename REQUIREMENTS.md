# AI-Powered Research Copilot — Requirements & Acceptance Criteria

## 1. Overview

A multi-agent system that automates academic and market research: crawling
sources, summarizing them, extracting formatted citations, identifying
trends, and auditing sources for bias — with a human approval gate before
questionable findings are finalized. The system runs entirely on a single
machine with no external database service, and persists research context
across sessions.

**Tech stack:** LangGraph (orchestration), Groq LLMs (reasoning), ChromaDB
(embedded vector store), Scrapy (crawling).

---

## 2. System-Level Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| SYS-1 | Orchestrate five agents as a directed pipeline | A compiled `StateGraph` with nodes `load_context → crawl → summarize → cite → analyze_trends → review → human_gate → finalize`, executable via a single `run()` call |
| SYS-2 | Run without a managed database service | No agent requires a network call to a DB server; the vector store persists to local disk |
| SYS-3 | Persist state across process restarts for a given project | Re-invoking a run with the same project ID reloads prior context notes before crawling begins |
| SYS-4 | Expose a CLI for triggering and resuming runs | Command-line entry point accepts project ID, seed URLs, source type, and human decisions; exits cleanly on success |
| SYS-5 | Degrade gracefully on partial agent failure | A failed crawl (timeout, network error) returns an empty result rather than raising, allowing downstream stages to proceed |

---

## 3. Agent-Level Functional Requirements

### 3.1 Crawler Agent (Data Collector)

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| CRW-1 | Accept a list of seed URLs and a source type | Callable with `(urls: list[str], source_type: str)` |
| CRW-2 | Extract clean page text, not raw HTML | Tags stripped, whitespace normalized before output |
| CRW-3 | Respect `robots.txt` by default | Crawler politeness setting enabled by default |
| CRW-4 | Follow same-domain links up to a configurable depth | Depth limit configurable via environment variable; cross-domain links not followed |
| CRW-5 | Persist every successfully-crawled document to storage before returning | Each document with non-empty text is written to the vector store and the returned record includes its storage ID |
| CRW-6 | Not crash the pipeline on crawl timeout | Timeout is caught; partial results (if any) are still used |
| CRW-7 | Skip documents with empty extracted text | Empty-text documents excluded from storage |

### 3.2 Summarizer Agent (Insight Condenser)

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| SUM-1 | Generate a concise (3–5 sentence) summary per document | Enforced via prompt instructions |
| SUM-2 | Extract up to 5 key insights as discrete points | Insights returned as a list, consumed downstream by metrics and trend analysis |
| SUM-3 | Report a self-assessed confidence score (0–1) per summary | Defaults to a neutral value if the model omits it |
| SUM-4 | Persist the summary back to storage, keyed to the same document ID | Storage write updates the existing record rather than creating a duplicate |
| SUM-5 | Process a batch of documents independently | One document's failure should not abort the rest of the batch *(currently a gap — see §7)* |

### 3.3 Citation Agent (Reference Manager)

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| CIT-1 | Extract author(s), publication, year, and title from document text | Returned as structured fields, nulls allowed when unknown |
| CIT-2 | Produce an APA-formatted citation | Falls back to sensible defaults ("Unknown Author", "n.d.", "Untitled") when fields are missing — never errors on incomplete metadata |
| CIT-3 | Produce an MLA-formatted citation | Same fallback behavior as CIT-2 |
| CIT-4 | Retain the original source URL as a live reference in both formats | URL present in both citation strings |
| CIT-5 | Record the citation's access date | Current date included in both formats |

### 3.4 Trend Analyzer Agent (Market Analyst)

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| TRD-1 | Identify emerging topics across the current document batch | Output includes topic, supporting source count, and rationale per topic |
| TRD-2 | Ground analysis in previously-ingested documents, not just the current batch (RAG) | A similarity search against stored history runs before the LLM call, and results are injected into the prompt |
| TRD-3 | Report how many historical documents grounded the analysis | Count of retrieved related documents included in output |
| TRD-4 | Handle an empty input batch without calling the LLM | Empty batch returns a default empty structure with no API call made |
| TRD-5 | Produce a short (2–4 sentence) overall trend narrative | Enforced via prompt instructions |

### 3.5 Reviewer Agent (Quality Auditor)

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| REV-1 | Produce an initial bias/quality assessment per document | Includes bias score (0–1), bias notes, accuracy concerns, and a quality flag |
| REV-2 | Run a self-reflection pass that critiques and can revise the initial assessment | A second, separately-prompted pass produces a final verdict distinct from the initial one |
| REV-3 | Fall back to the initial review if the reflection pass fails to parse | No crash on malformed reflection output |
| REV-4 | Flag a document for human review when warranted | Triggered by high bias score, an explicit "needs review" flag, or any accuracy concern |
| REV-5 | Persist the final bias score and quality flag back to storage | Storage record updated with final verdict |
| REV-6 | Human-escalation threshold is configurable | Adjustable via environment variable, not hardcoded |

---

## 4. Cross-Cutting Engineering Patterns

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| PAT-1 | Pipeline pauses when any document needs human review | Execution halts at a dedicated gate and returns the review queue to the caller |
| PAT-2 | A paused run is resumable with human decisions, without re-running upstream agents | Resuming restores state at the pause point rather than re-executing prior stages |
| PAT-3 | Human-in-the-loop is toggleable off entirely | When disabled, the pause is skipped regardless of queue contents |
| PAT-4 | Long-term context notes are scoped per project and retrievable in insertion order | Notes filtered by project ID and returned oldest-first |
| PAT-5 | A context note is written at the end of every run | Always executed, regardless of whether human review was triggered |
| PAT-6 | Retrieval is filterable by document type | Source documents and context notes never cross-contaminate search results |
| PAT-7 | Retrieval does not error on an empty/fresh store | A guard prevents failure when no documents have been ingested yet |
| PAT-8 | Self-reflection is a distinct, separately-prompted step | Not the same call merely asked to "double check itself" — a genuinely separate pass with its own prompt and inputs |

---

## 5. Non-Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| NFR-1 | All credentials/endpoints are environment-driven | No hardcoded secrets; every configurable value documented in an example env file |
| NFR-2 | LLM calls retry on transient failure | Automatic retry with backoff on API/network errors |
| NFR-3 | Malformed LLM JSON output does not crash the pipeline | Parse failures are caught and returned as a flagged error object |
| NFR-4 | Re-processing the same document updates rather than duplicates its record | Storage writes are idempotent, keyed by a stable document ID |
| NFR-5 | Every pipeline stage reports its own duration | Timing captured and merged into run-level metrics |
| NFR-6 | Crawling throttles itself to avoid overloading target sites | Auto-throttling and a minimum delay between requests enabled by default |
| NFR-7 | The system runs without any external database server | No database connection string or container required to operate |

---

## 6. Evaluation Metrics Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| MET-1 | Report estimated time saved vs. manual research | Computed from pipeline wall-clock time against a configurable manual-research baseline |
| MET-2 | Report a bias detection rate | Percentage of processed documents flagged by the Reviewer |
| MET-3 | Report an accuracy-improvement proxy | Derived from how much the self-reflection pass corrected the initial review |
| MET-4 | Metrics compute correctly even on partial/incomplete runs | No error thrown when expected fields are missing from state |

---

## 7. Known Gaps (Not Yet Met)

- **SUM-5 not fully met** — a failure on one document during batch summarization, citation, or review currently halts the whole batch instead of skipping just that document.
- **No automated test suite** — no unit or integration tests exist yet; all verification so far has been manual.
- **No input validation on seed URLs** — malformed URLs fail silently (empty result) rather than surfacing a clear error.
- **HITL pause/resume is in-process only** — restarting the application loses a paused run; a persistent checkpointer is required for that to survive restarts.
- **No accuracy ground-truth** — the accuracy-improvement metric (MET-3) is an internal proxy, not validated against a labeled benchmark dataset.

---

## 8. Out of Scope (this version)

- Multi-user concurrent access / authentication
- A web UI (CLI only)
- Non-English source documents
- Per-domain crawl extraction rules (e.g. arXiv-specific parsing)
- Distributed/multi-machine deployment
