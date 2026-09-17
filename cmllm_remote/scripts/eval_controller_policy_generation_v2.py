#!/usr/bin/env python3
"""Generation-time eval for strict action-index controller policies."""

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
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return None, "no_json_object"
    raw = match.group(0)
    try:
        return json.loads(raw), None
    except Exception as exc:
        return None, f"json_parse_error:{type(exc).__name__}"


def get_assistant_json(row):
    answer = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return json.loads(answer)


def get_observation(row):
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return json.loads(user)


def get_expected_index(row):
    expected = get_assistant_json(row)
    return expected.get("action_index")


def get_pred_index(parsed):
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("action_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def make_prompt(tokenizer, row):
    prompt_messages = [m for m in row["messages"] if m["role"] != "assistant"]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt = ""
    for msg in prompt_messages:
        prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
    prompt += "<|assistant|>\n"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(args.device)
    model.eval()

    counters = Counter()
    by_expected = defaultdict(Counter)
    confusion = Counter()
    examples = []

    for idx, row in enumerate(rows):
        expected = get_expected_index(row)
        obs = get_observation(row)
        valid_indices = set(obs.get("valid_action_indices", []))
        action_space_indices = {item.get("index") for item in obs.get("action_space", [])}
        prompt = make_prompt(tokenizer, row)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        parsed, parse_error = extract_first_json(generated)
        pred = get_pred_index(parsed)

        counters["total"] += 1
        by_expected[expected]["total"] += 1

        if parsed is not None:
            counters["json_parse_ok"] += 1
            by_expected[expected]["json_parse_ok"] += 1
        else:
            counters[f"parse_error:{parse_error}"] += 1
            by_expected[expected][f"parse_error:{parse_error}"] += 1

        schema_ok = isinstance(parsed, dict) and set(parsed.keys()) == {"action_index"} and pred is not None
        if schema_ok:
            counters["strict_schema_ok"] += 1
            by_expected[expected]["strict_schema_ok"] += 1

        in_action_space = pred in action_space_indices
        if in_action_space:
            counters["in_action_space_ok"] += 1
            by_expected[expected]["in_action_space_ok"] += 1

        valid = pred in valid_indices
        if valid:
            counters["valid_action_ok"] += 1
            by_expected[expected]["valid_action_ok"] += 1

        match = pred == expected
        if match:
            counters["action_index_match"] += 1
            by_expected[expected]["action_index_match"] += 1

        confusion[(expected, pred)] += 1
        examples.append(
            {
                "idx": idx,
                "id": row.get("id"),
                "episode_id": row.get("episode_id"),
                "expected_action_index": expected,
                "generated_text": generated,
                "parsed": parsed,
                "pred_action_index": pred,
                "parse_error": parse_error,
                "strict_schema_ok": schema_ok,
                "in_action_space": in_action_space,
                "valid_action": valid,
                "action_index_match": match,
            }
        )

    def ratio(name):
        total = counters["total"]
        return (counters[name] / total) if total else 0.0

    by_expected_summary = {}
    for expected, cnt in sorted(by_expected.items()):
        total = cnt["total"]
        by_expected_summary[str(expected)] = {
            "total": total,
            "json_parse_rate": cnt["json_parse_ok"] / total if total else 0.0,
            "strict_schema_rate": cnt["strict_schema_ok"] / total if total else 0.0,
            "in_action_space_rate": cnt["in_action_space_ok"] / total if total else 0.0,
            "valid_action_rate": cnt["valid_action_ok"] / total if total else 0.0,
            "action_index_accuracy": cnt["action_index_match"] / total if total else 0.0,
        }

    summary = {
        "num_samples": counters["total"],
        "json_parse_ok": counters["json_parse_ok"],
        "json_parse_rate": ratio("json_parse_ok"),
        "strict_schema_ok": counters["strict_schema_ok"],
        "strict_schema_rate": ratio("strict_schema_ok"),
        "in_action_space_ok": counters["in_action_space_ok"],
        "in_action_space_rate": ratio("in_action_space_ok"),
        "valid_action_ok": counters["valid_action_ok"],
        "valid_action_rate": ratio("valid_action_ok"),
        "action_index_match": counters["action_index_match"],
        "action_index_accuracy": ratio("action_index_match"),
        "by_expected_action_index": by_expected_summary,
        "confusion_top": [
            {"expected": k[0], "predicted": k[1], "count": v}
            for k, v in confusion.most_common(50)
        ],
        "model": args.model,
        "adapter": args.adapter,
        "eval_jsonl": args.eval_jsonl,
    }

    (out_dir / "generation_eval_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "generation_eval_examples_v2.jsonl", examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
