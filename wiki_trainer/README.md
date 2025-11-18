# Wiki Trainer

Utilities for turning Fastcrawl's Wikipedia chunks into Hugging Face datasets and fine-tuning a causal language model using `transformers` + `uv`.

## Prerequisites

- Python 3.13 (already provided by the `uv` shim installed at repository root)
- `uv` >= 0.9 for dependency + virtualenv management
- GPU drivers/tooling that can run PyTorch (install CUDA/cuDNN or use CPU for smoke tests)
- A local snapshot of chunks, e.g. `data/wiki_embeddings.jsonl` produced by Fastcrawl's embedder pipeline

## Setup

```sh
cd wiki_trainer
UV_CACHE_DIR=../.cache/uv uv sync  # creates .venv and installs dependencies (torch via the `training` extra)
source .venv/bin/activate
```

`uv sync` respects the `pyproject.toml` optional dependency group named `training`, so PyTorch + bitsandbytes are installed automatically. Adjust `UV_CACHE_DIR` if you keep cache files elsewhere (the repo root already has `.cache/uv`).

## Converting chunks to train/eval JSONL

Run the `prepare-data` subcommand to down-select and split the chunk corpus. By default it expects OpenAI-style embedding JSONL rows (with `text`, `url`, etc.), but it also works with normalized Fastcrawl pages that include `body_text` or `chunks[].text`.

```sh
uv run wiki-trainer prepare-data \
  ../data/wiki_embeddings.jsonl \
  --output-dir artifacts/datasets \
  --min-chars 200 \
  --max-chars 1600 \
  --max-chunks 50000 \
  --eval-ratio 0.02
```

The command writes `train.jsonl` and `eval.jsonl` into `artifacts/datasets`. Each row keeps the original text plus metadata columns (`source_url`, `chunk_id`, `section_path`) so you can trace model behavior back to specific chunks.

## Fine-tuning a model

Once the dataset exists, call `wiki-trainer train` with your preferred Hugging Face checkpoint. The defaults target `distilgpt2`, but you can swap in any causal LM (TinyLlama, Mistral, etc.) so long as it fits on your hardware.

```sh
uv run wiki-trainer train \
  artifacts/datasets \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir artifacts/checkpoints/tinyllama \
  --context-length 1024 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 1e-4 \
  --eval-steps 100
```

The CLI wraps Hugging Face's `Trainer` so standard knobs (batch size, gradient accumulation, precision flags) are exposed. Logs/checkpoints land under `artifacts/checkpoints/...` by default.

## Tips

- **Filtering.** Increase `--min-chars` to drop stubby chunks or pass `--max-chunks`/`--eval-ratio` to control dataset size.
- **Precision.** Use `--bf16` or `--fp16` once your hardware + drivers support it; otherwise leave them disabled for CPU proof-of-life runs.
- **Custom schedules.** Edit `wiki_trainer/config.py` to add weight-decay or warmup strategies, then re-export the CLI arguments if you need more control.
- **Streaming/large corpora.** `prepare-data` currently loads the filtered samples into memory before shuffling. For multi-million chunk runs consider chunked pre-processing or swapping the implementation for a disk-backed shuffle buffer.

## Repository integration

The project stays isolated inside `wiki_trainer/` so it can evolve independently of the Rust crawler. Use `uv run wiki-trainer --help` to see every flag, and keep data artifacts under `wiki_trainer/artifacts/` (already referenced in the defaults) so they stay out of the Rust workspace.
