#!/usr/bin/env python3
"""Lightweight DPO training for the high-level controller policy.

The preference JSONL is produced by build_controller_policy_dataset.py. This
trainer starts from an SFT LoRA adapter, keeps a frozen copy as the reference,
and optimizes the policy adapter with the standard DPO objective.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def format_prompt(tokenizer, messages):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    text = ""
    for msg in messages:
        text += f"<|{msg['role']}|>\n{msg['content']}\n"
    text += "<|assistant|>\n"
    return text


def format_full(tokenizer, messages, answer):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return format_prompt(tokenizer, messages) + answer + (tokenizer.eos_token or "")


def encode_pair(tokenizer, messages, answer, max_length):
    prompt = format_prompt(tokenizer, messages)
    full = format_full(tokenizer, messages, answer)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]
    if not any(x != -100 for x in labels):
        labels[-1] = full_ids[-1]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class PreferenceDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=2048, max_rows=0):
        self.rows = read_jsonl(path, max_rows=max_rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        chosen = encode_pair(self.tokenizer, row["messages"], row["chosen"], self.max_length)
        rejected = encode_pair(self.tokenizer, row["messages"], row["rejected"], self.max_length)
        return {
            "id": row.get("id"),
            "source": row.get("source"),
            "margin": float(row.get("margin", 1.0) or 1.0),
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
        }


class PreferenceCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def _pad(self, seqs, pad_value):
        max_len = max(len(x) for x in seqs)
        out = []
        mask = []
        for x in seqs:
            pad = max_len - len(x)
            out.append(torch.cat([torch.full((pad,), pad_value, dtype=x.dtype), x]))
            mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(x), dtype=torch.long)]))
        return torch.stack(out), torch.stack(mask)

    def __call__(self, batch):
        chosen_ids, chosen_mask = self._pad([x["chosen_input_ids"] for x in batch], self.pad_id)
        rejected_ids, rejected_mask = self._pad([x["rejected_input_ids"] for x in batch], self.pad_id)
        chosen_labels, _ = self._pad([x["chosen_labels"] for x in batch], -100)
        rejected_labels, _ = self._pad([x["rejected_labels"] for x in batch], -100)
        return {
            "ids": [x["id"] for x in batch],
            "sources": [x["source"] for x in batch],
            "margins": torch.tensor([x["margin"] for x in batch], dtype=torch.float32),
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
        }


def sequence_logps(model, input_ids, attention_mask, labels):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    target = labels[:, 1:]
    valid = target != -100
    safe_target = target.masked_fill(~valid, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid.float()).sum(dim=-1), valid.float().sum(dim=-1).clamp_min(1.0)


def batch_logps(model, batch):
    chosen_logp, chosen_len = sequence_logps(
        model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )
    rejected_logp, rejected_len = sequence_logps(
        model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )
    return chosen_logp, rejected_logp, chosen_len, rejected_len


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(policy, reference, loader, beta, device, max_batches=0):
    policy.eval()
    reference.eval()
    stats = {
        "pairs": 0,
        "loss_sum": 0.0,
        "chosen_logp_acc": 0,
        "dpo_reward_acc": 0,
        "policy_margin_sum": 0.0,
        "reward_margin_sum": 0.0,
    }
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = to_device(batch, device)
        pc, pr, _, _ = batch_logps(policy, batch)
        rc, rr, _, _ = batch_logps(reference, batch)
        policy_margin = pc - pr
        ref_margin = rc - rr
        reward_margin = policy_margin - ref_margin
        loss = -F.logsigmoid(beta * reward_margin)
        bs = pc.numel()
        stats["pairs"] += bs
        stats["loss_sum"] += float(loss.sum().item())
        stats["chosen_logp_acc"] += int((policy_margin > 0).sum().item())
        stats["dpo_reward_acc"] += int((reward_margin > 0).sum().item())
        stats["policy_margin_sum"] += float(policy_margin.sum().item())
        stats["reward_margin_sum"] += float(reward_margin.sum().item())
    pairs = max(stats["pairs"], 1)
    return {
        "pairs": stats["pairs"],
        "loss": stats["loss_sum"] / pairs,
        "chosen_logp_acc": stats["chosen_logp_acc"] / pairs,
        "dpo_reward_acc": stats["dpo_reward_acc"] / pairs,
        "mean_policy_margin": stats["policy_margin_sum"] / pairs,
        "mean_reward_margin": stats["reward_margin_sum"] / pairs,
    }


def load_policy_and_ref(args, dtype):
    base_policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_policy.config.use_cache = False
    policy = PeftModel.from_pretrained(base_policy, args.sft_adapter, is_trainable=True)
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    base_ref = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_ref.config.use_cache = False
    reference = PeftModel.from_pretrained(base_ref, args.sft_adapter, is_trainable=False)
    for p in reference.parameters():
        p.requires_grad_(False)
    return policy, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, reference = load_policy_and_ref(args, dtype)
    policy.to(device)
    reference.to(device)

    train_ds = PreferenceDataset(args.train_jsonl, tokenizer, args.max_length, args.max_train_rows)
    val_ds = PreferenceDataset(args.val_jsonl, tokenizer, args.max_length, args.max_val_rows)
    collator = PreferenceCollator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    planned_steps = int(math.ceil(args.epochs * steps_per_epoch)) if args.max_steps < 0 else args.max_steps

    state = {
        "args": vars(args),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "planned_steps": planned_steps,
        "history": [],
    }
    (out_dir / "controller_policy_dpo_config.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = evaluate(policy, reference, val_loader, args.beta, device, args.eval_max_batches)
    print("baseline_eval", json.dumps(baseline, ensure_ascii=False))

    policy.train()
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=planned_steps, desc="controller_dpo")

    while global_step < planned_steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            pc, pr, _, _ = batch_logps(policy, batch)
            with torch.no_grad():
                rc, rr, _, _ = batch_logps(reference, batch)
            reward_margin = (pc - pr) - (rc - rr)
            loss_vec = -F.logsigmoid(args.beta * reward_margin)
            loss = loss_vec.mean() / args.grad_accum
            loss.backward()
            micro_step += 1
            accum_loss += float(loss.item())

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.update(1)

                if global_step % args.logging_steps == 0:
                    log_row = {
                        "step": global_step,
                        "train_loss": accum_loss,
                        "mean_reward_margin": float(reward_margin.detach().mean().item()),
                        "chosen_logp_acc": float(((pc - pr) > 0).float().mean().item()),
                        "dpo_reward_acc": float((reward_margin > 0).float().mean().item()),
                    }
                    print("train_log", json.dumps(log_row, ensure_ascii=False))
                    state["history"].append(log_row)
                    accum_loss = 0.0

                if global_step % args.eval_steps == 0:
                    metrics = evaluate(policy, reference, val_loader, args.beta, device, args.eval_max_batches)
                    metrics["step"] = global_step
                    print("eval_log", json.dumps(metrics, ensure_ascii=False))
                    state["history"].append({"eval": metrics})
                    policy.train()

                if global_step % args.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    policy.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

                if global_step >= planned_steps:
                    break

        if args.max_steps < 0 and global_step >= planned_steps:
            break

    pbar.close()
    final_metrics = evaluate(policy, reference, val_loader, args.beta, device, args.eval_max_batches)
    final_metrics["step"] = global_step
    print("final_eval", json.dumps(final_metrics, ensure_ascii=False))
    state["final_eval"] = final_metrics
    (out_dir / "controller_policy_dpo_config.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
