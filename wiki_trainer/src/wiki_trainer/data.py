from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .config import DatasetConfig

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class Sample:
    text: str
    source_url: Optional[str]
    chunk_id: Optional[int]
    section_path: Optional[str]

    def as_json(self) -> Dict[str, Optional[str]]:
        payload: Dict[str, Optional[str]] = {
            "text": self.text,
            "source_url": self.source_url,
            "chunk_id": self.chunk_id,
            "section_path": self.section_path,
        }
        return payload


def clean_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return WHITESPACE_RE.sub(" ", text)


def detect_url(record: Dict) -> Optional[str]:
    if isinstance(record.get("url"), str):
        return record["url"]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        url = metadata.get("url")
        if isinstance(url, str):
            return url
    return None


def section_path_string(section_path: Optional[Iterable[Dict]]) -> Optional[str]:
    if not section_path:
        return None
    titles: List[str] = []
    for node in section_path:
        title = node.get("title") if isinstance(node, dict) else None
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return " > ".join(titles) if titles else None


def iter_text_fields(record: Dict) -> Iterator[Tuple[str, Dict]]:
    if isinstance(record.get("text"), str):
        yield record["text"], record
    if isinstance(record.get("body_text"), str):
        yield record["body_text"], record
    chunks = record.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                yield chunk["text"], chunk


def collect_samples(cfg: DatasetConfig) -> List[Sample]:
    samples: List[Sample] = []
    path = cfg.resolved_input()
    if not path.exists():
        raise FileNotFoundError(f"input JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            url = detect_url(record)
            for text_value, text_record in iter_text_fields(record):
                cleaned = clean_text(text_value)
                if len(cleaned) < cfg.min_chars:
                    continue
                if cfg.max_chars and len(cleaned) > cfg.max_chars:
                    continue
                chunk_id = text_record.get("chunk_id")
                if isinstance(chunk_id, str):
                    try:
                        chunk_id = int(chunk_id)
                    except ValueError:
                        chunk_id = None
                elif not isinstance(chunk_id, int):
                    chunk_id = None
                section_path = section_path_string(text_record.get("section_path"))
                samples.append(
                    Sample(
                        text=cleaned,
                        source_url=text_record.get("url", url),
                        chunk_id=chunk_id,
                        section_path=section_path,
                    )
                )
                if cfg.max_chunks and len(samples) >= cfg.max_chunks:
                    return samples
    return samples


def write_jsonl(path: Path, rows: Iterable[Sample]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row.as_json(), handle, ensure_ascii=False)
            handle.write("\n")
            count += 1
    return count


def prepare_dataset(cfg: DatasetConfig) -> Dict[str, int]:
    samples = collect_samples(cfg)
    if not samples:
        raise ValueError("no samples matched the filtering criteria")

    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(samples)

    eval_size = max(1, int(len(samples) * cfg.eval_ratio)) if cfg.eval_ratio > 0 else 0
    train_size = len(samples) - eval_size
    train_rows = samples[:train_size] if train_size > 0 else samples
    eval_rows = samples[train_size:] if eval_size > 0 else []

    output_dir = cfg.resolved_output()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": write_jsonl(output_dir / "train.jsonl", train_rows),
        "eval": write_jsonl(output_dir / "eval.jsonl", eval_rows),
    }
    return counts
