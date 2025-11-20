from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .data import Sample, write_jsonl
from .retrieval import DatasetRetriever


def _load_questions(path: Path) -> List[str]:
    questions: List[str] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = {}
                if isinstance(record, dict):
                    question = record.get("question") or record.get("prompt")
                    if isinstance(question, str) and question.strip():
                        questions.append(question.strip())
                        continue
                questions.append(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    questions.append(line)
    return questions


@dataclass(slots=True)
class QAPrepConfig:
    dataset_dir: Path
    questions_path: Path
    output_path: Path
    top_k: int = 1

    def resolved_dataset(self) -> Path:
        return self.dataset_dir.expanduser().resolve()

    def resolved_questions(self) -> Path:
        return self.questions_path.expanduser().resolve()

    def resolved_output(self) -> Path:
        return self.output_path.expanduser().resolve()


def prepare_qa_dataset(cfg: QAPrepConfig) -> int:
    dataset_dir = cfg.resolved_dataset()
    question_file = cfg.resolved_questions()
    questions = _load_questions(question_file)
    if not questions:
        raise ValueError(f"no questions found in {question_file}")
    retriever = DatasetRetriever(dataset_dir)
    samples: List[Sample] = []
    for question in questions:
        docs = retriever.retrieve(question, top_k=max(1, cfg.top_k))
        if not docs:
            continue
        answer = docs[0].text.strip()
        if not answer:
            continue
        text = f"Question: {question.strip()}\nAnswer: {answer}"
        samples.append(
            Sample(
                text=text,
                source_url=docs[0].source_url,
                chunk_id=None,
                section_path=None,
            )
        )
    if not samples:
        raise ValueError("no QA samples could be generated; check question coverage")
    output_path = cfg.resolved_output()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(output_path, samples)
