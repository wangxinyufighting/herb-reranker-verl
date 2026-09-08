"""从训练病例划分验证集；相同原始输入成组划分，不读取 GT 决定分组。"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .prepare_data import read_jsonl


def split_records(
    rows: list[dict], validation_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 (0, 1) 内")
    groups = {}
    seen = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("sample_id 缺失或重复")
        seen.add(sample_id)
        description = row.get("symptom_description")
        symptoms = row.get("symptoms")
        if (
            not isinstance(description, str)
            or not description.strip()
            or not isinstance(symptoms, list)
        ):
            raise ValueError("缺少原始症状输入")
        key = json.dumps([sorted(symptoms), description.strip()], ensure_ascii=False)
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError("至少需要两个不同输入组才能划分训练/验证集")
    ordered = sorted(
        groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest()
    )
    count = max(1, min(len(groups) - 1, round(len(groups) * validation_ratio)))
    val_keys = set(ordered[:count])
    train = [
        row for key, values in groups.items() if key not in val_keys for row in values
    ]
    validation = [
        row for key, values in groups.items() if key in val_keys for row in values
    ]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = [args.input, args.train_output, args.validation_output]
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("输入和两个输出必须使用不同路径")
    train, validation = split_records(
        read_jsonl(args.input), args.validation_ratio, args.seed
    )
    for path, rows in (
        (args.train_output, train),
        (args.validation_output, validation),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"训练 {len(train)} 条，验证 {len(validation)} 条；测试集未参与划分。")


if __name__ == "__main__":
    main()
