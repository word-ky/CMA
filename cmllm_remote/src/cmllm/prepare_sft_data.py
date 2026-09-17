from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROMPT = (
    "请根据这张煤矿井下钻孔作业图像，生成一句简洁、准确的安全监测表达。"
    "要求只描述图像中可见目标和可合理判断的安全关注点，不要编造人员、动作或事故。"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "image_path": row["image_path"] if "image_path" in row else None,
        "prompt": PROMPT,
        "response": row["final_safety_text"],
        "risk_level": row.get("risk_level"),
        "validation_passed": row.get("validation_passed", False),
        "objects": row.get("objects", []),
        "spatial_relations": row.get("spatial_relations", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dataset", required=True, type=Path)
    parser.add_argument("--subset-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    final_rows = read_jsonl(args.final_dataset)
    subset_by_id = {row["sample_id"]: row for row in read_jsonl(args.subset_manifest)}
    merged = []
    for row in final_rows:
        subset = subset_by_id.get(row["sample_id"], {})
        item = {**subset, **row}
        if item.get("image_path") and Path(item["image_path"]).exists():
            merged.append(convert(item))

    rng = random.Random(args.seed)
    rng.shuffle(merged)
    val_count = max(1, int(len(merged) * args.val_ratio)) if len(merged) > 1 else 0
    val_rows = merged[:val_count]
    train_rows = merged[val_count:]

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    meta = {
        "total": len(merged),
        "train": len(train_rows),
        "val": len(val_rows),
        "source_final_dataset": str(args.final_dataset),
        "source_subset_manifest": str(args.subset_manifest),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

