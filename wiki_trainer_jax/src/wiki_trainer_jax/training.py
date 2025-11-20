from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import metrax
import optax
from flax.training import checkpoints, train_state
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import SequenceBatcher, encode_sequences
from .model import TransformerLM
from .tokenizer import load_tokenizer, pad_id


class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def _create_optimizer(cfg: TrainingConfig, total_steps: int) -> optax.GradientTransformation:
    warmup = max(1, min(cfg.warmup_steps, total_steps))
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, cfg.learning_rate, transition_steps=warmup),
            optax.linear_schedule(cfg.learning_rate, 0.0, transition_steps=max(total_steps - warmup, 1)),
        ],
        boundaries=[warmup],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )
    return tx


def _build_batcher(path: Path, tokenizer, cfg: TrainingConfig, pad_token_id: int, shuffle: bool) -> Optional[SequenceBatcher]:
    if not path.exists():
        return None
    sequences = encode_sequences(path, tokenizer, cfg.seq_length, pad_token_id)
    if len(sequences) == 0:
        raise ValueError(f"dataset at {path} did not yield any sequences")
    return SequenceBatcher(sequences, cfg.batch_size, shuffle=shuffle)


def _prepare_batch(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return {
        "inputs": jnp.array(batch["inputs"], dtype=jnp.int32),
        "targets": jnp.array(batch["targets"], dtype=jnp.int32),
    }


def train_model(cfg: TrainingConfig) -> Path:
    dataset_dir = cfg.resolved_dataset()
    tokenizer_path = cfg.resolved_tokenizer()
    tokenizer = load_tokenizer(tokenizer_path)
    pad_token_id = pad_id(tokenizer)

    train_batcher = _build_batcher(cfg.train_file, tokenizer, cfg, pad_token_id, shuffle=True)
    if train_batcher is None:
        raise FileNotFoundError(f"train.jsonl not found at {cfg.train_file}")
    eval_batcher = _build_batcher(cfg.eval_file, tokenizer, cfg, pad_token_id, shuffle=False)

    vocab_size = tokenizer.GetPieceSize()
    steps_per_epoch = math.ceil(len(train_batcher) / cfg.batch_size)
    if steps_per_epoch <= 0:
        raise ValueError("no training batches available")
    total_steps = cfg.max_steps or (steps_per_epoch * cfg.epochs)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    model = TransformerLM(
        vocab_size=vocab_size,
        max_length=cfg.seq_length,
        embed_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        mlp_dim=cfg.mlp_dim,
        dropout_rate=cfg.dropout_rate,
    )
    rng = jax.random.PRNGKey(cfg.shuffle_seed)
    init_rng, dropout_rng = jax.random.split(rng)
    dummy = jnp.ones((1, cfg.seq_length), dtype=jnp.int32)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy, train=True)
    tx = _create_optimizer(cfg, total_steps)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, dropout_rng=dropout_rng)

    pad_value = jnp.array(pad_token_id, dtype=jnp.int32)

    @jax.jit
    def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["inputs"], train=True, rngs={"dropout": dropout_rng})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
            mask = (batch["targets"] != pad_value).astype(jnp.float32)
            loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            metric_state = metrax.Perplexity.from_model_output(
                predictions=logits,
                labels=batch["targets"],
                sample_weights=mask,
                from_logits=True,
            )
            return loss, metric_state

        (loss, metric_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state.replace(dropout_rng=new_dropout_rng), loss, metric_state.compute()

    @jax.jit
    def eval_step(params, batch: Dict[str, jnp.ndarray]):
        logits = model.apply({"params": params}, batch["inputs"], train=False)
        mask = (batch["targets"] != pad_value).astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
        loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        metric_state = metrax.Perplexity.from_model_output(
            predictions=logits,
            labels=batch["targets"],
            sample_weights=mask,
            from_logits=True,
        )
        return loss, metric_state

    output_dir = cfg.resolved_output()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    global_step = 0
    progress = tqdm(total=total_steps, desc="Training", unit="step")
    for epoch in range(cfg.epochs):
        epoch_seed = cfg.shuffle_seed + epoch
        for batch in train_batcher.batches(seed=epoch_seed):
            batch_arrays = _prepare_batch(batch)
            state, loss_value, ppl_value = train_step(state, batch_arrays)
            global_step += 1
            progress.set_postfix({"loss": float(loss_value), "ppl": float(ppl_value)})
            progress.update(1)
            if eval_batcher and (global_step % cfg.eval_interval == 0 or global_step == 1):
                metrics = evaluate(model, state.params, eval_batcher, pad_value, eval_step)
                progress.write(
                    f"[step {global_step}] eval loss={metrics['loss']:.4f} "
                    f"perplexity={metrics['perplexity']:.2f}"
                )
            if global_step % cfg.save_every == 0:
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target=state,
                    step=global_step,
                    overwrite=True,
                    keep=cfg.keep_last,
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    progress.close()

    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=global_step,
        overwrite=True,
        keep=cfg.keep_last,
    )

    metadata = {
        "seq_length": cfg.seq_length,
        "vocab_size": vocab_size,
        "model_dim": cfg.model_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "mlp_dim": cfg.mlp_dim,
        "dropout_rate": cfg.dropout_rate,
        "pad_token_id": pad_token_id,
        "learning_rate": cfg.learning_rate,
        "total_steps": global_step,
        "tokenizer": str(tokenizer_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.model")
    except OSError:
        pass

    return output_dir


def evaluate(model, params, batcher: SequenceBatcher, pad_value: jnp.ndarray, eval_step_fn):
    total_loss = 0.0
    total_batches = 0
    metric_state = metrax.Perplexity.empty()
    for batch in batcher.batches(seed=0):
        loss, metric = eval_step_fn(params, _prepare_batch(batch))
        total_loss += float(loss)
        total_batches += 1
        metric_state = metric_state.merge(metric)
    if total_batches == 0:
        raise ValueError("evaluation dataset is empty")
    return {
        "loss": total_loss / total_batches,
        "perplexity": float(metric_state.compute()),
    }
