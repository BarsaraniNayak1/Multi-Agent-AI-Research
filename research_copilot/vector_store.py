from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:  # pragma: no cover - handled by dependency installation
    chromadb = None


class LocalVectorStore:
    def __init__(self, project_id: str, root_dir: str | Path | None = None) -> None:
        self.project_id = project_id
        self.root_dir = Path(root_dir) if root_dir is not None else Path(".research_projects")
        self.persist_dir = self.root_dir / project_id / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._use_chroma = chromadb is not None

        if self._use_chroma:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(name="research_documents")
        else:
            self._json_store = self.root_dir / project_id / "vector_store.json"
            if not self._json_store.exists():
                self._json_store.write_text("[]", encoding="utf-8")

    def upsert(self, doc: dict[str, Any]) -> str:
        doc_id = str(doc.get("id") or doc.get("document_id") or doc.get("url") or "doc")
        metadata = {
            key: value
            for key, value in doc.items()
            if key not in {"text", "summary", "citation", "review", "insights"}
            and isinstance(value, (str, int, float, bool))
        }
        metadata.setdefault("record_type", "source_document")
        if self._use_chroma:
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc.get("text", "")],
                metadatas=[metadata],
            )
        else:
            data = json.loads(self._json_store.read_text(encoding="utf-8"))
            existing = {entry.get("id"): entry for entry in data}
            existing[doc_id] = doc
            self._json_store.write_text(json.dumps(list(existing.values()), indent=2), encoding="utf-8")
        return doc_id

    def upsert_context_note(self, note_id: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        payload = {"id": note_id, "text": text, "record_type": "context_note", **(metadata or {})}
        return self.upsert(payload)

    def query(self, query_text: str, limit: int = 5, record_type: str = "source_document") -> list[dict[str, Any]]:
        if not query_text.strip():
            return []
        if self._use_chroma:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={"record_type": record_type},
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            merged: list[dict[str, Any]] = []
            for doc_id, text, meta in zip(ids, docs, metas):
                merged.append({"id": doc_id, "text": text, "metadata": meta or {}})
            return merged

        data = json.loads(self._json_store.read_text(encoding="utf-8"))
        if not data:
            return []
        filtered = []
        for item in data:
            text = str(item.get("text", ""))
            if item.get("record_type", "source_document") != record_type:
                continue
            query_words = set(query_text.lower().split())
            if query_words.intersection(text.lower().split()):
                filtered.append(item)
        return filtered[:limit]
