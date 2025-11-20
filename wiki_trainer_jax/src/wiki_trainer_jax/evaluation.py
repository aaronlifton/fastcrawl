from __future__ import annotations

import jax
import jax.numpy as jnp
import metrax
import optax

from .config import EvalConfig, GenerationConfig
from .dataset import SequenceBatcher, encode_sequences
from .inference import load_model
from .tokenizer import load_tokenizer, pad_id


def evaluate_checkpoint(cfg: EvalConfig) -> dict:
    dataset_dir = cfg.resolved_dataset()
    tokenizer = load_tokenizer(cfg.resolved_tokenizer())
    pad_token_id = pad_id(tokenizer)
    gen_cfg = GenerationConfig(
        checkpoint_dir=cfg.resolved_checkpoint(),
        sp_model_path=cfg.resolved_tokenizer(),
        dataset_dir=None,
        max_new_tokens=1,
    )
    artifacts = load_model(gen_cfg)
    sequences = encode_sequences(dataset_dir / "eval.jsonl", tokenizer, artifacts.seq_length, pad_token_id)
    batcher = SequenceBatcher(sequences, cfg.batch_size, shuffle=False)
    pad_value = jnp.array(pad_token_id, dtype=jnp.int32)

    @jax.jit
    def eval_step(batch):
        inputs = jnp.array(batch["inputs"], dtype=jnp.int32)
        targets = jnp.array(batch["targets"], dtype=jnp.int32)
        logits = artifacts.model.apply({"params": artifacts.params}, inputs, train=False)
        mask = (targets != pad_value).astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        metric_state = metrax.Perplexity.from_model_output(
            predictions=logits,
            labels=targets,
            sample_weights=mask,
            from_logits=True,
        )
        return loss, metric_state

    total_loss = 0.0
    total_batches = 0
    metric_state = metrax.Perplexity.empty()
    for batch in batcher.batches(seed=0):
        loss, metric = eval_step(batch)
        total_loss += float(loss)
        total_batches += 1
        metric_state = metric_state.merge(metric)
    if total_batches == 0:
        raise ValueError("evaluation dataset yielded no batches")
    return {"loss": total_loss / total_batches, "perplexity": float(metric_state.compute())}
