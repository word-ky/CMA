from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT = (
    "请根据这张煤矿井下钻孔作业图像，生成一句简洁、准确的安全监测表达。"
    "要求只描述图像中可见目标和可合理判断的安全关注点，不要编造人员、动作或事故。"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((448, 448))
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--eval-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    rows = read_jsonl(args.eval_jsonl)[: args.limit]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[load_image(row["image_path"])], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[1] :]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            f.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "target": row["response"],
                        "prediction": pred.strip(),
                        "image_path": row["image_path"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

