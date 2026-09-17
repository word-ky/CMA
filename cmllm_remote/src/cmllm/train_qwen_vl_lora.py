from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SafetyDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((448, 448))
    return image


@dataclass
class QwenVLCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for row in features:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["prompt"]},
                    ],
                },
                {"role": "assistant", "content": row["response"]},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(load_image(row["image_path"]))
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SafetyDataset(args.train_jsonl)
    val_ds = SafetyDataset(args.val_jsonl)
    collator = QwenVLCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=5,
        logging_steps=5,
        save_steps=40,
        eval_steps=40,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    processor.save_pretrained(str(args.output_dir / "final_adapter"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

