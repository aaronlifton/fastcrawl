from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from sentencepiece import SentencePieceProcessor

from .config import GenerationConfig
from .coverage import DatasetCoverage
from .model import TransformerLM
from .retrieval import DatasetRetriever
from .tokenizer import load_tokenizer, pad_id


@dataclass(slots=True)
class LoadedModel:
    model: TransformerLM
    params: dict
    tokenizer: SentencePieceProcessor
    seq_length: int
    pad_token_id: int
    metadata: dict


def _load_metadata(checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_params(ckpt_dir: Path):
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=None)
    if restored is None:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    if isinstance(restored, dict) and "params" in restored:
        return restored["params"]
    if hasattr(restored, "params"):
        return restored.params
    return restored


def load_model(cfg: GenerationConfig) -> LoadedModel:
    checkpoint_dir = cfg.resolved_checkpoint()
    metadata = _load_metadata(checkpoint_dir)
    tokenizer = load_tokenizer(cfg.resolved_tokenizer())
    pad_token_id = metadata.get("pad_token_id", pad_id(tokenizer))
    model = TransformerLM(
        vocab_size=metadata["vocab_size"],
        max_length=metadata["seq_length"],
        embed_dim=metadata["model_dim"],
        num_heads=metadata["num_heads"],
        num_layers=metadata["num_layers"],
        mlp_dim=metadata["mlp_dim"],
        dropout_rate=metadata["dropout_rate"],
    )
    params = _load_params(checkpoint_dir / "checkpoints")
    return LoadedModel(
        model=model,
        params=params,
        tokenizer=tokenizer,
        seq_length=metadata["seq_length"],
        pad_token_id=pad_token_id,
        metadata=metadata,
    )


def _format_prompt(question: str, docs: List[str]) -> str:
    if not docs:
        return f"Question: {question.strip()}\nAnswer:"
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        formatted.append(f"Document {idx}:\n{doc.strip()}")
    block = "\n\n".join(formatted)
    return f"{block}\n\nQuestion: {question.strip()}\nAnswer:"


def _sample_next_token(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator, greedy: bool) -> int:
    if greedy:
        return int(np.argmax(logits))
    if temperature <= 0 or np.isnan(temperature):
        temperature = 1.0
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    if top_p <= 0 or top_p >= 1:
        return int(rng.choice(len(probs), p=probs))
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = cumulative <= top_p
    if not np.any(cutoff):
        cutoff[0] = True
    filtered_idx = sorted_idx[cutoff]
    filtered_probs = sorted_probs[cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()
    return int(rng.choice(filtered_idx, p=filtered_probs))


def _prepare_context(tokens: List[int], seq_length: int, pad_token_id: int) -> jnp.ndarray:
    context = tokens[-seq_length:]
    if len(context) < seq_length:
        context = [pad_token_id] * (seq_length - len(context)) + context
    return jnp.array([context], dtype=jnp.int32)


def generate_answer(cfg: GenerationConfig, question: str) -> str:
    artifacts = load_model(cfg)
    tokenizer = artifacts.tokenizer
    rng = np.random.default_rng(seed=0)
    docs: List[str] = []
    dataset_path = cfg.resolved_dataset()
    retriever = None
    if dataset_path:
        try:
            retriever = DatasetRetriever(dataset_path)
            retrieved = retriever.retrieve(question, top_k=max(1, cfg.context_documents))
            docs = [doc.text for doc in retrieved]
        except FileNotFoundError:
            retriever = None
            docs = []
    prompt_text = _format_prompt(question, docs)
    tokens = tokenizer.EncodeAsIds(prompt_text)
    if tokenizer.bos_id() >= 0:
        tokens = [tokenizer.bos_id()] + tokens
    eos_id = tokenizer.eos_id()

    def next_token(current_tokens: List[int]) -> int:
        context = _prepare_context(current_tokens, artifacts.seq_length, artifacts.pad_token_id)
        logits = artifacts.model.apply({"params": artifacts.params}, context, train=False)
        next_logits = np.array(logits[0, -1, :])
        return _sample_next_token(next_logits, cfg.temperature, cfg.top_p, rng, cfg.greedy)

    generated: List[int] = []
    for _ in range(cfg.max_new_tokens):
        token_id = next_token(tokens)
        tokens.append(token_id)
        generated.append(token_id)
        if eos_id >= 0 and token_id == eos_id:
            break
    return tokenizer.DecodeIds(generated).strip()


def chat_loop(cfg: GenerationConfig, prompt: Optional[str] = None) -> None:
    checkpoint_dir = cfg.resolved_checkpoint()
    dataset_path = cfg.resolved_dataset()
    coverage = None
    retriever = None
    if dataset_path:
        try:
            coverage = DatasetCoverage(dataset_path)
            retriever = DatasetRetriever(dataset_path)
        except FileNotFoundError:
            coverage = None
            retriever = None

    artifacts = load_model(cfg)
    tokenizer = artifacts.tokenizer
    eos_id = tokenizer.eos_id()
    rng = np.random.default_rng(seed=0)

    def should_answer(question: str) -> bool:
        if not coverage:
            return True
        return coverage.has_topic(question, min_matches=cfg.min_keyword_matches)

    def respond(question: str) -> None:
        if not should_answer(question):
            print(cfg.unknown_response)
            return
        docs: List[str] = []
        if retriever:
            docs = [doc.text for doc in retriever.retrieve(question, top_k=max(1, cfg.context_documents))]
        prompt_text = _format_prompt(question, docs)
        tokens = tokenizer.EncodeAsIds(prompt_text)
        if tokenizer.bos_id() >= 0:
            tokens = [tokenizer.bos_id()] + tokens

        generated: List[int] = []

        def next_token(current_tokens: List[int]) -> int:
            context = _prepare_context(current_tokens, artifacts.seq_length, artifacts.pad_token_id)
            logits = artifacts.model.apply({"params": artifacts.params}, context, train=False)
            next_logits = np.array(logits[0, -1, :])
            return _sample_next_token(next_logits, cfg.temperature, cfg.top_p, rng, cfg.greedy)

        for _ in range(cfg.max_new_tokens):
            token_id = next_token(tokens)
            tokens.append(token_id)
            generated.append(token_id)
            if eos_id >= 0 and token_id == eos_id:
                break
        text = tokenizer.DecodeIds(generated).strip()
        if text:
            print(text)
        else:
            print(cfg.unknown_response)

    print(f"Loaded checkpoint from {checkpoint_dir}")
    print("Press Enter on an empty line to exit.\n")

    if prompt:
        respond(prompt)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not user_input:
            print("Exiting chat.")
            break
        respond(user_input)
