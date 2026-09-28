from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProjectStore:
    def __init__(self, project_id: str, root_dir: str | Path | None = None) -> None:
        self.project_id = project_id
        self.root_dir = Path(root_dir) if root_dir is not None else Path(".research_projects")
        self.project_dir = self.root_dir / project_id
        self.document_dir = self.project_dir / "documents"
        self.context_file = self.project_dir / "context_notes.jsonl"
        self.run_state_file = self.project_dir / "run_state.json"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.context_file.touch(exist_ok=True)

    def add_context_note(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        note = {
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        with self.context_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(note, ensure_ascii=False) + "\n")
        return note

    def load_context_notes(self) -> list[dict[str, Any]]:
        if not self.context_file.exists() or self.context_file.stat().st_size == 0:
            return []
        notes: list[dict[str, Any]] = []
        with self.context_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    notes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return sorted(notes, key=lambda item: item.get("created_at", ""))

    def save_document(self, doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        json_path = self.document_dir / f"{doc_id}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload

    def load_document(self, doc_id: str) -> dict[str, Any] | None:
        json_path = self.document_dir / f"{doc_id}.json"
        if not json_path.exists():
            return None
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(self.document_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                try:
                    docs.append(json.load(handle))
                except json.JSONDecodeError:
                    continue
        return docs

    def save_run_state(self, state: dict[str, Any]) -> None:
        serializable = {key: value for key, value in state.items() if key not in {"project_store", "vector_store"}}
        with self.run_state_file.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2, default=str)

    def load_run_state(self) -> dict[str, Any] | None:
        if not self.run_state_file.exists():
            return None
        try:
            with self.run_state_file.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
