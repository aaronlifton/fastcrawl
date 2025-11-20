from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

import orjson
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from .config import TokenizerConfig
from .data import clean_text


def _iter_text_lines(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                text = record.get("text")
                if isinstance(text, str) and text.strip():
                    yield clean_text(text)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line


def train_sentencepiece(cfg: TokenizerConfig) -> Path:
    """Train a SentencePiece model from the input corpus."""

    input_path = cfg.resolved_input()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    prefix = cfg.resolved_prefix()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for text in _iter_text_lines(input_path):
            tmp.write(text.replace("\n", " ").strip())
            tmp.write("\n")
        tmp_path = tmp.name

    try:
        SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=str(prefix),
            vocab_size=cfg.vocab_size,
            character_coverage=cfg.character_coverage,
            model_type=cfg.model_type,
            max_sentence_length=cfg.max_sentence_length,
            pad_id=3,
            pad_piece="<pad>",
            bos_id=1,
            eos_id=2,
            unk_id=0,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    model_path = prefix.with_suffix(".model")
    if not model_path.exists():
        raise RuntimeError(f"failed to create SentencePiece model at {model_path}")
    return model_path


def load_tokenizer(path: Path) -> SentencePieceProcessor:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {resolved}")
    processor = SentencePieceProcessor()
    processor.Load(str(resolved))
    return processor


def pad_id(processor: SentencePieceProcessor) -> int:
    pad = processor.pad_id()
    if pad >= 0:
        return pad
    # Fall back to unk id to keep masking behavior deterministic.
    unk = processor.unk_id()
    if unk >= 0:
        return unk
    return 0
