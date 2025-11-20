from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .coverage import extract_keywords
from .data import clean_text


@dataclass(slots=True)
class RetrievedDoc:
    text: str
    keywords: set[str]
    source_url: Optional[str]


class DatasetRetriever:
    """Keyword-overlap retriever for prepared Fastcrawl datasets."""

    def __init__(self, dataset_dir: Path):
        resolved = dataset_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"dataset directory not found: {resolved}")
        self.dataset_dir = resolved
        self.documents = self._load_documents()

    def _load_documents(self) -> List[RetrievedDoc]:
        docs: List[RetrievedDoc] = []
        for name in ("train.jsonl", "eval.jsonl"):
            path = self.dataset_dir / name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text")
                    if isinstance(text, str) and text.strip():
                        cleaned = clean_text(text)
                        keywords = set(extract_keywords(cleaned))
                        if keywords:
                            docs.append(
                                RetrievedDoc(
                                    text=cleaned,
                                    keywords=keywords,
                                    source_url=record.get("source_url"),
                                )
                            )
        return docs

    def retrieve(self, query: str, top_k: int = 1) -> List[RetrievedDoc]:
        if not self.documents:
            return []
        query_keywords = set(extract_keywords(query))
        if not query_keywords:
            return []
        scored: List[tuple[float, RetrievedDoc]] = []
        for doc in self.documents:
            overlap = len(query_keywords & doc.keywords)
            if overlap == 0:
                continue
            score = overlap / len(query_keywords)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
