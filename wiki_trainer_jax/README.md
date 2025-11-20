# Wiki Trainer JAX

`wiki_trainer_jax` mirrors the original Fastcrawl wiki trainer workflow but keeps the entire
stack inside the JAX ecosystem: SentencePiece tokenization, a Flax causal Transformer we own,
Optax optimizers, Orbax checkpoints, and MetraX metrics. No Hugging Face or PyTorch deps are
required, so everything runs on the “pure” JAX toolchain you described (JAX + Flax/NNX +
Optax + MetraX + Orbax).

## Prerequisites

- Python 3.12+ (same as the repository-level `uv` shim).
- `uv` ≥ 0.9 for dependency + venv management.
- Working JAX install (CPU-only works for smoke tests; GPU/TPU wheels recommended).
- Local wiki chunks, e.g. `data/wiki_embeddings.jsonl`.
- A SentencePiece model (`.model`). Use `train-tokenizer` to build one from the dataset or supply your own.

## Setup

```sh
cd wiki_trainer_jax
UV_CACHE_DIR=../.cache/uv uv sync
source .venv/bin/activate
```

`uv sync` installs the CPU wheels of JAX/Flax by default. If you need CUDA/ROCm wheels,
set `JAX_DEFAULT_DEVICE`/`JAX_PLATFORM_NAME` and reinstall `jaxlib` per the
[official instructions](https://jax.readthedocs.io/en/latest/installation.html)
after finishing `uv sync`.

## Workflow

1. **Prepare datasets** – identical CLI + filters as the PyTorch trainer:

   ```sh
   uv run wiki-trainer-jax prepare-data \
     ../data/wiki_embeddings.jsonl \
     --output-dir artifacts/datasets \
     --min-chars 200 \
     --max-chars 1600 \
     --max-chunks 50000
   ```

2. **Train or reuse a SentencePiece model** (optional if you already have one):

   ```sh
   uv run wiki-trainer-jax train-tokenizer \
     artifacts/datasets/train.jsonl \
     --model-prefix artifacts/tokenizers/wiki_sp \
     --vocab-size 32000
   ```

3. **Fine-tune with the Flax Transformer (pure JAX stack)**:

   ```sh
   uv run wiki-trainer-jax train \
     artifacts/datasets \
     --sp-model artifacts/tokenizers/wiki_sp.model \
     --output-dir artifacts/jax_checkpoints/wiki_small \
     --seq-length 512 \
     --epochs 2 \
     --batch-size 8
   ```

   The trainer logs loss + MetraX perplexity, saving Orbax checkpoints (`checkpoints/`)
   plus `metadata.json` so inference knows which tokenizer + hyperparameters to use.

4. **Evaluate or chat with retrieval-augmented prompting**:

   ```sh
   uv run wiki-trainer-jax evaluate \
     artifacts/jax_checkpoints/wiki_small \
     --dataset-dir artifacts/datasets \
     --sp-model artifacts/tokenizers/wiki_sp.model

   uv run wiki-trainer-jax chat \
     artifacts/jax_checkpoints/wiki_small \
     --sp-model artifacts/tokenizers/wiki_sp.model \
     --dataset-dir artifacts/datasets \
     --context-docs 3 \
     --prompt "Where is the Cherry Music Festival held?"
   ```

   The chat REPL retrieves on-topic chunks from your dataset, prepends them to the
   user question, and only answers if the coverage check finds overlapping keywords.

## Commands

- `prepare-data`: identical semantics to the PyTorch trainer.
- `train`: fine-tune the bundled Flax Transformer (SentencePiece tokenization + Optax).
- `evaluate`: compute loss + MetraX perplexity on `eval.jsonl`.
- `chat`: retrieval-augmented REPL with coverage gating and the custom Flax LM.
- `verify-question`: keyword coverage gate reused from the original trainer.

See `uv run wiki-trainer-jax --help` or per-command `--help` for every flag.
