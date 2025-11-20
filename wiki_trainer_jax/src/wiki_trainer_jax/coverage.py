from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from .data import clean_text

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "into",
    "onto",
    "about",
    "for",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "will",
    "would",
    "could",
    "should",
    "there",
    "their",
    "them",
    "they",
    "your",
    "you're",
    "you",
    "our",
    "ours",
    "his",
    "her",
    "hers",
    "its",
    "it's",
    "not",
    "but",
    "can",
    "able",
    "also",
    "over",
    "under",
    "than",
    "then",
    "been",
    "does",
    "did",
    "done",
    "how",
    "much",
    "many",
    "more",
    "most",
    "some",
    "such",
    "each",
    "per",
    "any",
    "all",
    "youre",
    "ever",
    "just",
    "only",
    "very",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def extract_keywords(text: str, min_length: int = 4) -> List[str]:
    tokens = WORD_RE.findall(text.lower())
    keywords: List[str] = []
    for token in tokens:
        normalized = token.strip("'\"-")
        if len(normalized) < min_length:
            continue
        if normalized in STOPWORDS:
            continue
        keywords.append(normalized)
    return keywords


@dataclass(slots=True)
class DatasetCoverage:
    dataset_dir: Path
    texts: List[str] = field(init=False)

    def __post_init__(self) -> None:
        resolved = self.dataset_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"dataset directory not found: {resolved}")
        self.dataset_dir = resolved
        self.texts = self._load_texts()

    def _load_texts(self) -> List[str]:
        texts: List[str] = []
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
                        texts.append(clean_text(text).lower())
        return texts

    def has_topic(self, query: str, min_matches: int = 2, match_ratio: float = 0.4) -> bool:
        if not self.texts:
            return False
        keywords = extract_keywords(query)
        if not keywords:
            return False
        threshold = max(1, min(len(keywords), min_matches))
        if match_ratio > 0:
            threshold = max(threshold, math.ceil(len(keywords) * match_ratio))
        for text in self.texts:
            hits = 0
            for keyword in keywords:
                if keyword in text:
                    hits += 1
                    if hits >= threshold:
                        return True
        return False

    def explain_miss(self, query: str) -> Sequence[str]:
        keywords = extract_keywords(query)
        present = set()
        for text in self.texts:
            for keyword in keywords:
                if keyword in text:
                    present.add(keyword)
        return [kw for kw in keywords if kw not in present]
