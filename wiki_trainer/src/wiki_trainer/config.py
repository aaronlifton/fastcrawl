from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for converting Fastcrawl wiki chunks into training splits."""

    input_path: Path
    output_dir: Path
    min_chars: int = 200
    max_chars: int = 1600
    max_chunks: Optional[int] = 200_000
    eval_ratio: float = 0.02
    seed: int = 13
    shuffle: bool = True
    include_keywords: Tuple[str, ...] = field(default_factory=tuple)
    require_keywords: Tuple[str, ...] = field(default_factory=tuple)

    def resolved_input(self) -> Path:
        return self.input_path.expanduser().resolve()

    def resolved_output(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def train_file(self) -> Path:
        return self.resolved_output() / "train.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.resolved_output() / "eval.jsonl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_keywords", tuple(self._normalize_keywords(self.include_keywords)))
        object.__setattr__(self, "require_keywords", tuple(self._normalize_keywords(self.require_keywords)))

    @staticmethod
    def _normalize_keywords(values) -> Tuple[str, ...]:
        return tuple(sorted({value.strip().lower() for value in values if isinstance(value, str) and value.strip()}))


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for fine-tuning a Hugging Face causal language model."""

    dataset_dir: Path
    model_name: str = "distilgpt2"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/checkpoints"))
    context_length: int = 512
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    bf16: bool = False
    fp16: bool = False
    pin_memory: bool = True

    def train_file(self) -> Path:
        return (self.dataset_dir / "train.jsonl").resolve()

    def eval_file(self) -> Path:
        return (self.dataset_dir / "eval.jsonl").resolve()

    def resolved_output(self) -> Path:
        return self.output_dir.expanduser().resolve()
