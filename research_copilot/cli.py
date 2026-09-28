from __future__ import annotations

import argparse
import json

from .pipeline import run_research_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Research Copilot")
    parser.add_argument("--project-id", required=True, help="Project identifier for state persistence")
    parser.add_argument("--seed-url", action="append", default=[], help="Seed URL to start crawling from")
    parser.add_argument("--source-type", default="web", help="Source type such as web, news, or academic")
    parser.add_argument("--human-review", action="store_true", help="Pause on flagged documents for a human decision")
    parser.add_argument("--resume", action="store_true", help="Resume a paused run")
    parser.add_argument("--decision", action="append", default=[], metavar="DOC_ID=DECISION", help="Human decision, for example DOC_ID=approve")
    args = parser.parse_args()

    decisions = {}
    for raw_decision in args.decision:
        if "=" in raw_decision:
            document_id, decision = raw_decision.split("=", 1)
            decisions[document_id] = decision

    result = run_research_pipeline(
        project_id=args.project_id,
        urls=args.seed_url,
        source_type=args.source_type,
        human_review=args.human_review,
        resume=args.resume,
        human_decisions=decisions,
    )

    print(json.dumps({k: v for k, v in result.items() if k not in {"project_store", "vector_store", "llm_client"}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
