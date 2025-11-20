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
        object.__setattr__(self, "include_keywords", tuple(self._normalize(self.include_keywords)))
        object.__setattr__(self, "require_keywords", tuple(self._normalize(self.require_keywords)))

    @staticmethod
    def _normalize(values) -> Tuple[str, ...]:
        cleaned = set()
        for value in values:
            if isinstance(value, str):
                stripped = value.strip().lower()
                if stripped:
                    cleaned.add(stripped)
        return tuple(sorted(cleaned))


@dataclass(slots=True)
class TokenizerConfig:
    """SentencePiece tokenizer training configuration."""

    input_path: Path
    model_prefix: Path
    vocab_size: int = 32_000
    character_coverage: float = 0.9995
    model_type: str = "bpe"
    max_sentence_length: int = 2000

    def resolved_input(self) -> Path:
        return self.input_path.expanduser().resolve()

    def resolved_prefix(self) -> Path:
        return self.model_prefix.expanduser().resolve()


@dataclass(slots=True)
class TrainingConfig:
    """JAX + MetraX training configuration."""

    dataset_dir: Path
    sp_model_path: Path
    output_dir: Path = field(default_factory=lambda: Path("artifacts/jax_checkpoints"))
    seq_length: int = 512
    batch_size: int = 8
    epochs: int = 1
    shuffle_seed: int = 13
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    mlp_dim: int = 2048
    dropout_rate: float = 0.1
    eval_interval: int = 200
    max_steps: Optional[int] = None
    save_every: int = 500
    keep_last: int = 2

    def resolved_dataset(self) -> Path:
        return self.dataset_dir.expanduser().resolve()

    def resolved_tokenizer(self) -> Path:
        return self.sp_model_path.expanduser().resolve()

    def resolved_output(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def train_file(self) -> Path:
        return self.resolved_dataset() / "train.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.resolved_dataset() / "eval.jsonl"


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for chat/inference."""

    checkpoint_dir: Path
    sp_model_path: Path
    dataset_dir: Optional[Path] = None
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    greedy: bool = False
    min_keyword_matches: int = 2
    unknown_response: str = "I don't know. That topic wasn't in my training data."
    context_documents: int = 2

    def resolved_checkpoint(self) -> Path:
        return self.checkpoint_dir.expanduser().resolve()

    def resolved_tokenizer(self) -> Path:
        return self.sp_model_path.expanduser().resolve()

    def resolved_dataset(self) -> Optional[Path]:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir.expanduser().resolve()


@dataclass(slots=True)
class EvalConfig:
    dataset_dir: Path
    sp_model_path: Path
    checkpoint_dir: Path
    batch_size: int = 8

    def resolved_dataset(self) -> Path:
        return self.dataset_dir.expanduser().resolve()

    def resolved_tokenizer(self) -> Path:
        return self.sp_model_path.expanduser().resolve()

    def resolved_checkpoint(self) -> Path:
        return self.checkpoint_dir.expanduser().resolve()
