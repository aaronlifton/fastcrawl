from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .config import (
    DatasetConfig,
    EvalConfig,
    GenerationConfig,
    TokenizerConfig,
    TrainingConfig,
)
from .coverage import DatasetCoverage
from .data import prepare_dataset
from .evaluation import evaluate_checkpoint
from .inference import chat_loop
from .tokenizer import train_sentencepiece
from .training import train_model

app = typer.Typer(add_completion=False, help="JAX + MetraX tooling for Fastcrawl wiki fine-tuning.")


@app.command("prepare-data")
def prepare_data_cmd(
    input_path: Path = typer.Argument(..., help="Path to wiki JSONL or embedding JSONL file"),
    output_dir: Path = typer.Option(Path("artifacts/datasets"), help="Directory for the train/eval splits"),
    min_chars: int = typer.Option(200, help="Minimum characters per chunk"),
    max_chars: int = typer.Option(1600, help="Maximum characters per chunk"),
    max_chunks: Optional[int] = typer.Option(200_000, help="Limit on processed chunks (None for all)"),
    eval_ratio: float = typer.Option(0.02, help="Portion of samples reserved for evaluation"),
    seed: int = typer.Option(13, help="RNG seed for shuffling"),
    no_shuffle: bool = typer.Option(False, help="Disable deterministic shuffling"),
    include_keyword: Optional[List[str]] = typer.Option(
        None,
        "--include-keyword",
        help="Keep only chunks containing at least one of these substrings (case-insensitive).",
        show_default=False,
    ),
    require_keyword: Optional[List[str]] = typer.Option(
        None,
        "--require-keyword",
        help="Ensure at least one chunk hits this substring; fails if missing.",
        show_default=False,
    ),
):
    cfg = DatasetConfig(
        input_path=input_path,
        output_dir=output_dir,
        min_chars=min_chars,
        max_chars=max_chars,
        max_chunks=max_chunks,
        eval_ratio=eval_ratio,
        seed=seed,
        shuffle=not no_shuffle,
        include_keywords=tuple(include_keyword or []),
        require_keywords=tuple(require_keyword or []),
    )
    counts = prepare_dataset(cfg)
    typer.echo(f"Wrote {counts['train']} train samples and {counts['eval']} eval samples to {cfg.resolved_output()}")


@app.command("train-tokenizer")
def train_tokenizer_cmd(
    input_path: Path = typer.Argument(..., help="JSONL corpus (train split) to build the tokenizer on"),
    model_prefix: Path = typer.Option(
        ..., "--model-prefix", "-o", help="Output prefix for SentencePiece files (.model/.vocab)"
    ),
    vocab_size: int = typer.Option(32_000, help="SentencePiece vocab size"),
    character_coverage: float = typer.Option(0.9995, help="SentencePiece character coverage"),
    model_type: str = typer.Option("bpe", help="SentencePiece model type (bpe/unigram/char/word)"),
    max_sentence_length: int = typer.Option(2000, help="Max chars per sentence when parsing input"),
):
    cfg = TokenizerConfig(
        input_path=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
    )
    model_path = train_sentencepiece(cfg)
    typer.echo(f"Tokenizer saved to {model_path}")


@app.command("train")
def train_cmd(
    dataset_dir: Path = typer.Argument(Path("artifacts/datasets"), help="Directory with train.jsonl and eval.jsonl"),
    sp_model: Path = typer.Option(..., "--sp-model", help="SentencePiece model path (.model)"),
    output_dir: Path = typer.Option(Path("artifacts/jax_checkpoints"), help="Checkpoint output directory"),
    seq_length: int = typer.Option(512, help="Training sequence length"),
    batch_size: int = typer.Option(8, help="Batch size"),
    epochs: int = typer.Option(1, help="Epochs"),
    learning_rate: float = typer.Option(2e-4, help="Peak learning rate"),
    warmup_steps: int = typer.Option(100, help="Warmup steps for LR schedule"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    grad_clip_norm: float = typer.Option(1.0, help="Gradient clipping norm"),
    model_dim: int = typer.Option(512, help="Transformer width"),
    num_heads: int = typer.Option(8, help="Attention heads"),
    num_layers: int = typer.Option(8, help="Transformer layers"),
    mlp_dim: int = typer.Option(2048, help="Feed-forward hidden size"),
    dropout_rate: float = typer.Option(0.1, help="Dropout rate"),
    eval_interval: int = typer.Option(200, help="Steps between eval passes"),
    save_every: int = typer.Option(500, help="Steps between checkpoints"),
    max_steps: Optional[int] = typer.Option(None, help="Optional hard stop on global steps"),
    shuffle_seed: int = typer.Option(13, help="Data shuffling seed"),
):
    cfg = TrainingConfig(
        dataset_dir=dataset_dir,
        sp_model_path=sp_model,
        output_dir=output_dir,
        seq_length=seq_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        eval_interval=eval_interval,
        save_every=save_every,
        max_steps=max_steps,
        shuffle_seed=shuffle_seed,
    )
    final_dir = train_model(cfg)
    typer.echo(f"Training complete. Checkpoints + metadata saved under {final_dir}")


@app.command("evaluate")
def evaluate_cmd(
    checkpoint_dir: Path = typer.Argument(..., help="Directory containing checkpoints + metadata.json"),
    dataset_dir: Path = typer.Option(Path("artifacts/datasets"), help="Dataset dir with eval.jsonl"),
    sp_model: Path = typer.Option(..., "--sp-model", help="SentencePiece model path"),
    batch_size: int = typer.Option(8, help="Batch size for evaluation"),
):
    cfg = EvalConfig(
        dataset_dir=dataset_dir,
        sp_model_path=sp_model,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
    )
    metrics = evaluate_checkpoint(cfg)
    typer.echo(f"Eval loss={metrics['loss']:.4f} perplexity={metrics['perplexity']:.2f}")


@app.command("chat")
def chat_cmd(
    checkpoint_dir: Path = typer.Argument(..., help="Checkpoint directory with metadata.json"),
    sp_model: Path = typer.Option(..., "--sp-model", help="SentencePiece model path for decoding"),
    dataset_dir: Optional[Path] = typer.Option(
        Path("artifacts/datasets"), help="Dataset directory for coverage checks", show_default=True
    ),
    max_new_tokens: int = typer.Option(160, help="Max tokens generated per turn"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p cutoff"),
    greedy: bool = typer.Option(False, help="Disable sampling and pick argmax tokens"),
    min_keyword_matches: int = typer.Option(2, help="Coverage keywords required to answer"),
    context_docs: int = typer.Option(2, help="Number of retrieved documents to prepend"),
    unknown_response: str = typer.Option(
        "I don't know. That topic wasn't in my training data.", help="Fallback when coverage is missing."
    ),
    prompt: Optional[str] = typer.Option(None, help="Optional single prompt to run before the REPL starts"),
):
    cfg = GenerationConfig(
        checkpoint_dir=checkpoint_dir,
        sp_model_path=sp_model,
        dataset_dir=dataset_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        greedy=greedy,
        min_keyword_matches=min_keyword_matches,
        unknown_response=unknown_response,
        context_documents=context_docs,
    )
    chat_loop(cfg, prompt=prompt)


@app.command("verify-question")
def verify_question_cmd(
    question: str = typer.Argument(..., help="Question to verify against the prepared dataset"),
    dataset_dir: Path = typer.Option(Path("artifacts/datasets"), help="Directory containing train/eval JSONL files"),
    min_keyword_matches: int = typer.Option(2, help="Minimum keyword overlaps required to call it covered"),
):
    coverage = DatasetCoverage(dataset_dir)
    if coverage.has_topic(question, min_matches=min_keyword_matches):
        typer.echo("Question appears in the dataset vocabulary; safe to train.")
    else:
        typer.echo("Dataset lacks coverage for that question.", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
