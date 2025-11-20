from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coverage import DatasetCoverage


@dataclass(slots=True)
class ChatConfig:
    checkpoint_dir: Path
    dataset_dir: Optional[Path] = None
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    min_keyword_matches: int = 2
    unknown_response: str = "I don't know. That topic wasn't in my training data."

    def resolved_checkpoint(self) -> Path:
        return self.checkpoint_dir.expanduser().resolve()

    def resolved_dataset(self) -> Optional[Path]:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir.expanduser().resolve()


def _ensure_tokenizer_pad(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chat_loop(cfg: ChatConfig, prompt: Optional[str] = None) -> None:
    checkpoint_path = cfg.resolved_checkpoint()
    tokenizer = _ensure_tokenizer_pad(AutoTokenizer.from_pretrained(checkpoint_path))
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.eval()

    coverage = None
    dataset_path = cfg.resolved_dataset()
    if dataset_path is not None:
        try:
            coverage = DatasetCoverage(dataset_path)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Coverage checks disabled.")
            coverage = None

    print("Loaded checkpoint from", checkpoint_path)
    print("Press Enter on an empty line to exit.\n")

    def run_turn(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][prompt_length:]
        if len(generated_ids) == 0:
            return ""
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.strip()

    def should_answer(text: str) -> bool:
        if not coverage:
            return True
        return coverage.has_topic(text, min_matches=cfg.min_keyword_matches)

    def respond(text: str) -> None:
        if not should_answer(text):
            print(cfg.unknown_response)
            return
        response = run_turn(text)
        if response:
            print(response)
        else:
            print(cfg.unknown_response)

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
