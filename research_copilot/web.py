from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .pipeline import run_research_pipeline
from .store import ProjectStore


_PROJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_PROJECT_ROOT = Path(".research_projects")


def _validate_project_id(value: Any) -> str:
    project_id = str(value or "").strip()
    if not _PROJECT_ID_PATTERN.fullmatch(project_id) or project_id in {".", ".."}:
        raise ValueError("Use 1-64 letters, numbers, dots, underscores, or hyphens for the project ID.")
    return project_id


def _validate_run_payload(payload: Any, *, resume: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    project_id = _validate_project_id(payload.get("project_id"))
    urls = payload.get("urls", [])
    if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
        raise ValueError("URLs must be a list of URL strings.")
    urls = [url.strip() for url in urls if url.strip()]
    if not resume and not urls:
        raise ValueError("Enter at least one seed URL to start a run.")
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Enter a complete HTTP or HTTPS URL: {url}")
    source_type = str(payload.get("source_type", "web")).strip()
    if not source_type:
        raise ValueError("Choose a source type.")
    decisions = payload.get("decisions", {})
    if not isinstance(decisions, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in decisions.items()):
        raise ValueError("Human decisions must map document IDs to decision strings.")
    return {
        "project_id": project_id,
        "urls": urls,
        "source_type": source_type,
        "human_review": bool(payload.get("human_review", False)),
        "decisions": decisions,
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"project_store", "vector_store", "llm_client"}}


class _ResearchHandler(BaseHTTPRequestHandler):
    server_version = "ResearchCopilot/1.0"

    def do_GET(self) -> None:
        request = urlparse(self.path)
        if request.path == "/":
            self._send_file(Path(__file__).with_name("index.html"), "text/html; charset=utf-8")
        elif request.path == "/api/projects":
            self._send_json(self._list_projects())
        elif request.path == "/api/project":
            self._get_project(parse_qs(request.query).get("project_id", [""])[0])
        else:
            self._send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:
        if self.path not in {"/api/run", "/api/resume"}:
            self._send_json({"error": "Not found."}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("Request body is too large.")
            payload = json.loads(self.rfile.read(length))
            request = _validate_run_payload(payload, resume=self.path == "/api/resume")
            result = run_research_pipeline(
                project_id=request["project_id"],
                urls=request["urls"],
                source_type=request["source_type"],
                human_review=request["human_review"],
                resume=self.path == "/api/resume",
                human_decisions=request["decisions"],
            )
            self._send_json(_public_result(result))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:
            self._send_json({"error": f"The research run could not be completed: {error}"}, status=500)

    def _list_projects(self) -> dict[str, Any]:
        projects = []
        if _PROJECT_ROOT.exists():
            for path in sorted(_PROJECT_ROOT.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_dir() or not _PROJECT_ID_PATTERN.fullmatch(path.name):
                    continue
                store = ProjectStore(path.name, _PROJECT_ROOT)
                state = store.load_run_state() or {}
                projects.append({
                    "project_id": path.name,
                    "document_count": len(store.list_documents()),
                    "status": state.get("status", "ready"),
                })
        return {"projects": projects}

    def _get_project(self, raw_project_id: str) -> None:
        try:
            project_id = _validate_project_id(raw_project_id)
            project_dir = _PROJECT_ROOT / project_id
            if not project_dir.is_dir():
                self._send_json({"error": "Project not found."}, status=404)
                return
            store = ProjectStore(project_id, _PROJECT_ROOT)
            state = store.load_run_state() or {}
            self._send_json({
                "project_id": project_id,
                "documents": store.list_documents(),
                "context_notes": store.load_context_notes(),
                "status": state.get("status", "ready"),
                "metrics": state.get("metrics", {}),
                "trend_analysis": state.get("trend_analysis", {}),
                "stage_durations": state.get("stage_durations", {}),
                "review_queue": state.get("review_queue", []),
                "message": state.get("message", ""),
                "crawl_error": state.get("crawl_error"),
                "crawl_errors": state.get("crawl_errors", []),
            })
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            content = path.read_bytes()
        except OSError:
            self._send_json({"error": "The dashboard file is unavailable."}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Open the local AI Research Copilot dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (defaults to loopback only)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _ResearchHandler)
    print(f"Research Copilot is ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())