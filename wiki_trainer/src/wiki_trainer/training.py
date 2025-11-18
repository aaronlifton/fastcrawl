from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig

_TRAINING_ARGUMENTS_SIGNATURE = inspect.signature(TrainingArguments.__init__)


def _training_arguments_supports(param_name: str) -> bool:
    return param_name in _TRAINING_ARGUMENTS_SIGNATURE.parameters


def load_text_dataset(dataset_dir: Path) -> Dict[str, "Dataset"]:
    data_files = {}
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"
    if train_path.exists():
        data_files["train"] = str(train_path)
    if eval_path.exists():
        data_files["validation"] = str(eval_path)
    if not data_files:
        raise FileNotFoundError(
            "Expected train.jsonl and/or eval.jsonl in dataset directory; run `wiki-trainer prepare-data` first"
        )
    return load_dataset("json", data_files=data_files)


def train_model(cfg: TrainingConfig) -> str:
    dataset = load_text_dataset(cfg.dataset_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.context_length,
            padding="max_length",
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names if "train" in dataset else dataset[list(dataset.keys())[0]].column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataset = tokenized_datasets.get("validation")
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    eval_strategy = "no"
    eval_steps = None
    if has_eval:
        eval_strategy = "steps" if cfg.eval_steps else "epoch"
        eval_steps = cfg.eval_steps

    save_strategy = "steps" if (cfg.save_steps or (eval_strategy == "steps" and eval_steps)) else "epoch"
    save_steps = cfg.save_steps or (eval_steps if eval_strategy == "steps" and eval_steps else 500)

    training_kwargs = dict(
        output_dir=str(cfg.resolved_output()),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=save_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_steps=save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to="none",
    )

    if _training_arguments_supports("dataloader_pin_memory"):
        training_kwargs["dataloader_pin_memory"] = cfg.pin_memory
    elif cfg.pin_memory is False:
        warnings.warn(
            "Installed transformers version does not expose dataloader_pin_memory; cannot disable pinning.",
            stacklevel=2,
        )

    if _training_arguments_supports("evaluation_strategy"):
        training_kwargs["evaluation_strategy"] = eval_strategy
    elif _training_arguments_supports("eval_strategy"):
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        raise RuntimeError("Unsupported transformers version: missing evaluation strategy parameter")

    training_args = TrainingArguments(**training_kwargs)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    output_dir = Path(cfg.resolved_output())
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return str(output_dir)
