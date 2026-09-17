#!/usr/bin/env python3
"""LoRA SFT for the high-level controller policy LLM.

The input JSONL is produced by build_controller_policy_dataset.py and contains
OpenAI-style messages. This trainer masks the prompt tokens and optimizes only
the assistant action JSON.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_jsonl(path, max_rows=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


class ControllerSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=4096, max_rows=0):
        self.rows = read_jsonl(path, max_rows=max_rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _format(self, row):
        messages = row["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        answer = next(m["content"] for m in messages if m["role"] == "assistant")
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full = self.tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt = ""
            for msg in prompt_messages:
                prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
            prompt += "<|assistant|>\n"
            full = prompt + answer + (self.tokenizer.eos_token or "")
        return prompt, full

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt, full = self._format(row)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full, add_special_tokens=False).input_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            labels = labels[overflow:]
        if not any(x != -100 for x in labels):
            # Keep at least the final token supervised if an extremely long
            # prompt consumed the full window.
            labels[-1] = full_ids[-1]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for item in features:
            ids = item["input_ids"]
            lab = item["labels"]
            pad = max_len - len(ids)
            input_ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), ids]))
            attention_mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
            labels.append(torch.cat([torch.full((pad,), -100, dtype=torch.long), lab]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base text CausalLM path or HF id.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        ),
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = ControllerSFTDataset(args.train_jsonl, tokenizer, args.max_length, args.max_train_rows)
    val_ds = ControllerSFTDataset(args.val_jsonl, tokenizer, args.max_length, args.max_val_rows)

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    (Path(args.out_dir) / "controller_policy_sft_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
