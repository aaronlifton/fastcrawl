from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import json
import numpy as np
import orjson
from sentencepiece import SentencePieceProcessor

from .data import clean_text


def _iter_texts(path: Path) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    record = json.loads(line)
                text = record.get("text")
                if isinstance(text, str) and text.strip():
                    yield clean_text(text)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


def encode_sequences(
    path: Path,
    tokenizer: SentencePieceProcessor,
    seq_length: int,
    pad_token_id: int,
) -> np.ndarray:
    """Tokenize the corpus into (seq_length + 1) slices for next-token prediction."""

    bos = tokenizer.bos_id() if tokenizer.bos_id() >= 0 else None
    eos = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None
    sequences: list[np.ndarray] = []
    tol = seq_length + 1
    for text in _iter_texts(path):
        tokens = tokenizer.EncodeAsIds(text)
        if bos is not None:
            tokens = [bos] + tokens
        if eos is not None:
            tokens = tokens + [eos]
        if len(tokens) < 2:
            continue
        start = 0
        while start < len(tokens) - 1:
            window = tokens[start : start + tol]
            if len(window) < tol:
                window = window + [pad_token_id] * (tol - len(window))
            sequences.append(np.array(window, dtype=np.int32))
            start += seq_length
    if not sequences:
        raise ValueError(f"no usable sequences found in {path}")
    return np.stack(sequences)


class SequenceBatcher:
    """Utility that yields language-model batches from encoded sequences."""

    def __init__(self, sequences: np.ndarray, batch_size: int, shuffle: bool = True):
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.sequences)

    def batches(self, *, seed: int = 0) -> Iterator[dict[str, np.ndarray]]:
        order = np.arange(len(self.sequences))
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.sequences[idx]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            yield {
                "inputs": inputs,
                "targets": targets,
            }
