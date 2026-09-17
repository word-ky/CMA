#!/usr/bin/env python3
"""Generation-time eval for the high-level controller policy.

This checks whether the SFT policy can produce valid action JSON on held-out
controller observations. It does not execute segmentation tools and does not
use oracle IoU.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path, max_items=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_items and len(rows) >= max_items:
                break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_first_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None, "no_json_object"
    raw = match.group(0)
    try:
        return json.loads(raw), None
    except Exception as exc:
        return None, f"json_parse_error:{type(exc).__name__}"


def normalize_dict(obj):
    if not isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def get_assistant_json(row):
    answer = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return json.loads(answer)


def get_allowed_actions(row):
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    try:
        obs = json.loads(user)
    except Exception:
        return []
    return obs.get("allowed_actions", [])


def make_prompt(tokenizer, row):
    prompt_messages = [m for m in row["messages"] if m["role"] != "assistant"]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt = ""
    for msg in prompt_messages:
        prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
    prompt += "<|assistant|>\n"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base text CausalLM path or HF id.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = read_jsonl(args.eval_jsonl, args.max_items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(args.device)
    model.eval()

    counters = Counter()
    by_action = defaultdict(Counter)
    confusion = Counter()
    examples = []

    for idx, row in enumerate(rows):
        expected = get_assistant_json(row)
        expected_action = expected.get("action")
        allowed = get_allowed_actions(row)
        prompt = make_prompt(tokenizer, row)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        parsed, parse_error = extract_first_json(generated)
        pred_action = parsed.get("action") if isinstance(parsed, dict) else None

        counters["total"] += 1
        by_action[expected_action]["total"] += 1

        if parsed is not None:
            counters["json_parse_ok"] += 1
            by_action[expected_action]["json_parse_ok"] += 1
        else:
            counters[f"parse_error:{parse_error}"] += 1
            by_action[expected_action][f"parse_error:{parse_error}"] += 1

        legal = bool(pred_action in allowed)
        if legal:
            counters["legal_action_ok"] += 1
            by_action[expected_action]["legal_action_ok"] += 1

        action_match = pred_action == expected_action
        if action_match:
            counters["action_match"] += 1
            by_action[expected_action]["action_match"] += 1

        source_match = bool(
            isinstance(parsed, dict)
            and parsed.get("source_action") == expected.get("source_action")
        )
        if source_match:
            counters["source_action_match"] += 1
            by_action[expected_action]["source_action_match"] += 1

        exact_json = normalize_dict(parsed) == normalize_dict(expected)
        if exact_json:
            counters["exact_json_match"] += 1
            by_action[expected_action]["exact_json_match"] += 1

        confusion[(expected_action, pred_action)] += 1

        examples.append(
            {
                "idx": idx,
                "id": row.get("id"),
                "episode_id": row.get("episode_id"),
                "expected": expected,
                "generated_text": generated,
                "parsed": parsed,
                "parse_error": parse_error,
                "legal_action": legal,
                "action_match": action_match,
                "source_action_match": source_match,
                "exact_json_match": exact_json,
            }
        )

    def ratio(name):
        total = counters["total"]
        return (counters[name] / total) if total else 0.0

    action_rows = {}
    for action, cnt in sorted(by_action.items()):
        total = cnt["total"]
        action_rows[action] = {
            "total": total,
            "json_parse_rate": cnt["json_parse_ok"] / total if total else 0.0,
            "legal_action_rate": cnt["legal_action_ok"] / total if total else 0.0,
            "action_accuracy": cnt["action_match"] / total if total else 0.0,
            "source_action_accuracy": cnt["source_action_match"] / total if total else 0.0,
            "exact_json_accuracy": cnt["exact_json_match"] / total if total else 0.0,
        }

    summary = {
        "num_samples": counters["total"],
        "json_parse_ok": counters["json_parse_ok"],
        "json_parse_rate": ratio("json_parse_ok"),
        "legal_action_ok": counters["legal_action_ok"],
        "legal_action_rate": ratio("legal_action_ok"),
        "action_match": counters["action_match"],
        "action_accuracy": ratio("action_match"),
        "source_action_match": counters["source_action_match"],
        "source_action_accuracy": ratio("source_action_match"),
        "exact_json_match": counters["exact_json_match"],
        "exact_json_accuracy": ratio("exact_json_match"),
        "by_expected_action": action_rows,
        "confusion_top": [
            {"expected": k[0], "predicted": k[1], "count": v}
            for k, v in confusion.most_common(50)
        ],
        "model": args.model,
        "adapter": args.adapter,
        "eval_jsonl": args.eval_jsonl,
    }

    (out_dir / "generation_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "generation_eval_examples.jsonl", examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
