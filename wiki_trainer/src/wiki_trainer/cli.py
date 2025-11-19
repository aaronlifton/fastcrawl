from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .config import DatasetConfig, TrainingConfig
from .coverage import DatasetCoverage
from .data import prepare_dataset
from .training import train_model
from .inference import ChatConfig, chat_loop

app = typer.Typer(add_completion=False, help="Prepare Fastcrawl wiki chunks and fine-tune a local model.")


@app.command("prepare-data")
def prepare_data(
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
        help="Keep only chunks whose text contains at least one of these substrings (case-insensitive).",
        show_default=False,
    ),
    require_keyword: Optional[List[str]] = typer.Option(
        None,
        "--require-keyword",
        help="Ensure at least one chunk contains this substring; fails if missing. Repeatable.",
        show_default=False,
    ),
):
    """Convert Fastcrawl wiki chunks into train/eval JSONL files."""

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


@app.command()
def train(
    dataset_dir: Path = typer.Argument(Path("artifacts/datasets"), help="Directory containing train.jsonl and eval.jsonl"),
    model_name: str = typer.Option("distilgpt2", help="Base Hugging Face model to fine-tune"),
    output_dir: Path = typer.Option(Path("artifacts/checkpoints"), help="Where to store checkpoints"),
    context_length: int = typer.Option(512, help="Max tokens per sample"),
    epochs: float = typer.Option(1.0, help="Training epochs"),
    batch_size: int = typer.Option(2, help="Per-device batch size"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    warmup_ratio: float = typer.Option(0.03, help="Warmup ratio"),
    logging_steps: int = typer.Option(25, help="Logging interval (steps)"),
    eval_steps: Optional[int] = typer.Option(None, help="Evaluation frequency (steps). Defaults to per-epoch."),
    save_steps: Optional[int] = typer.Option(None, help="Override checkpoint frequency (steps)"),
    bf16: bool = typer.Option(False, help="Enable bfloat16 training"),
    fp16: bool = typer.Option(False, help="Enable float16 training"),
    pin_memory: bool = typer.Option(True, help="Pin dataloader memory (disable on CPU-only runs)"),
):
    """Fine-tune a causal language model using the prepared dataset."""

    cfg = TrainingConfig(
        dataset_dir=dataset_dir,
        model_name=model_name,
        output_dir=output_dir,
        context_length=context_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        bf16=bf16,
        fp16=fp16,
        pin_memory=pin_memory,
    )
    final_dir = train_model(cfg)
    typer.echo(f"Training run complete. Checkpoints saved to {final_dir}")


@app.command("chat")
def chat(
    checkpoint_dir: Path = typer.Argument(Path("artifacts/checkpoints"), help="Directory with the fine-tuned checkpoint"),
    max_new_tokens: int = typer.Option(160, help="Maximum tokens generated per turn"),
    temperature: float = typer.Option(0.7, help="Sampling temperature (ignored when --greedy is set)"),
    top_p: float = typer.Option(0.9, help="Top-p sampling cutoff"),
    greedy: bool = typer.Option(False, help="Disable sampling and use greedy decoding"),
    dataset_dir: Optional[Path] = typer.Option(
        Path("artifacts/datasets"), help="Dataset directory used for training (enables coverage checks)."
    ),
    min_keyword_matches: int = typer.Option(2, help="Minimum keyword overlaps required to trust an answer"),
    unknown_response: str = typer.Option(
        "I don't know. That topic wasn't in my training data.", help="Fallback response when coverage is missing."
    ),
    prompt: Optional[str] = typer.Option(None, help="Optional one-off prompt to run before the REPL starts"),
):
    """Chat with a trained checkpoint using an interactive REPL."""

    cfg = ChatConfig(
        checkpoint_dir=checkpoint_dir,
        dataset_dir=dataset_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=not greedy,
        min_keyword_matches=min_keyword_matches,
        unknown_response=unknown_response,
    )
    chat_loop(cfg, prompt=prompt)


@app.command("verify-question")
def verify_question(
    question: str = typer.Argument(..., help="Question to verify against the prepared dataset"),
    dataset_dir: Path = typer.Option(Path("artifacts/datasets"), help="Directory containing train/eval JSONL files"),
    min_keyword_matches: int = typer.Option(2, help="Minimum keyword overlaps required to call it covered"),
):
    """Check if a dataset contains enough topical overlap to answer a question."""

    coverage = DatasetCoverage(dataset_dir)
    if coverage.has_topic(question, min_matches=min_keyword_matches):
        typer.echo("Question appears in the dataset vocabulary; safe to train.")
    else:
        typer.echo("Dataset lacks coverage for that question.", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
