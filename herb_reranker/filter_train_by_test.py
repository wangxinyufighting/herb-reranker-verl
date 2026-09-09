"""根据测试输入的症状分布，构造更小的训练子集。

该脚本只读取测试集的症状字段（新协议为 ``extra_info.symptoms``，旧协议为
顶层 ``symptoms``），不会读取测试 GT、奖励或候选药材。
它适合缩短开发阶段的 GRPO 训练时间；正式论文实验应同时报告完整训练集结果，
或将相同筛选协议公平地应用到全部对比方法。

支持三种筛选方式：

``overlap``
    训练病例只要包含任一测试症状就保留。定义最宽松，通常缩减很少。
``exact``
    只保留症状组合在测试集中完整出现过的训练病例，不限制每组数量。
``matched``
    默认模式。按症状组合分组，并为每个测试病例最多保留
    ``train_per_test`` 条训练病例。若测试症状组合在训练集中不存在，则按症状
    Jaccard 相似度选择最近的训练组合。该模式能够显著缩小训练集，同时避免按
    文件顺序截断造成偏置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SymptomSignature = tuple[str, ...]


@dataclass
class SelectionStats:
    """训练子集构造统计。"""

    mode: str
    train_rows: int
    test_rows: int
    selected_rows: int
    dropped_rows: int
    retained_ratio: float
    train_signatures: int
    test_signatures: int
    exact_covered_test_rows: int
    fallback_covered_test_rows: int
    uncovered_test_rows: int
    exact_selected_rows: int
    fallback_selected_rows: int
    train_per_test: int
    seed: int
    used_test_fields: tuple[str, ...] = ("extra_info.symptoms",)
    used_test_labels: bool = False


def _symptom_signature(row: dict[str, Any], source: str, row_index: int) -> SymptomSignature:
    """提取与顺序无关的规范症状组合，并执行严格校验。"""

    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        symptoms = extra_info.get("symptoms")
        if not isinstance(symptoms, list) or not symptoms:
            raise ValueError(
                f"{source} 第 {row_index + 1} 行的 extra_info.symptoms 必须是非空列表"
            )
    else:
        # 旧版 smart_treatment Parquet 将症状保存为顶层字符串。
        raw_symptoms = row.get("symptoms")
        if not isinstance(raw_symptoms, str) or not raw_symptoms.strip():
            raise ValueError(
                f"{source} 第 {row_index + 1} 行缺少 extra_info.symptoms 或 symptoms"
            )
        symptoms = [
            item.strip()
            for item in re.split(r"[、,，;；]", raw_symptoms)
            if item.strip()
        ]

    cleaned: list[str] = []
    for symptom in symptoms:
        if not isinstance(symptom, str) or not symptom.strip():
            raise ValueError(
                f"{source} 第 {row_index + 1} 行的 extra_info.symptoms 含非法值"
            )
        cleaned.append(symptom.strip())

    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{source} 第 {row_index + 1} 行含重复症状")
    return tuple(sorted(cleaned))


def _sample_id(row: dict[str, Any], row_index: int) -> str:
    """读取样本 ID；缺失时使用行号，保证筛选仍可复现。"""

    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        sample_id = extra_info.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip():
            return sample_id.strip()
    return f"row-{row_index}"


def _stable_order(
    indices: Iterable[int], train_rows: Sequence[dict[str, Any]], seed: int
) -> list[int]:
    """用稳定哈希打乱组内病例，避免总是选中原文件前几行。"""

    def key(index: int) -> bytes:
        value = f"{seed}\0{_sample_id(train_rows[index], index)}\0{index}"
        return hashlib.sha256(value.encode("utf-8")).digest()

    return sorted(indices, key=key)


def _jaccard(left: SymptomSignature, right: SymptomSignature) -> float:
    """计算两个症状组合的 Jaccard 相似度。"""

    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def select_train_indices(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    mode: str = "matched",
    train_per_test: int = 2,
    seed: int = 42,
    nearest_fallback: bool = True,
) -> tuple[list[int], SelectionStats]:
    """返回应保留的训练行号以及完整统计信息。"""

    if mode not in {"overlap", "exact", "matched"}:
        raise ValueError("mode 必须是 overlap、exact 或 matched")
    if train_per_test <= 0:
        raise ValueError("train_per_test 必须大于 0")
    if not train_rows or not test_rows:
        raise ValueError("训练集和测试集都不能为空")

    train_groups: dict[SymptomSignature, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        train_groups[_symptom_signature(row, "训练集", index)].append(index)

    test_counts = Counter(
        _symptom_signature(row, "测试集", index)
        for index, row in enumerate(test_rows)
    )
    ordered_groups = {
        signature: _stable_order(indices, train_rows, seed)
        for signature, indices in train_groups.items()
    }

    exact_covered_test_rows = sum(
        count for signature, count in test_counts.items() if signature in train_groups
    )
    exact_selected: set[int] = set()
    fallback_selected: set[int] = set()
    fallback_covered_test_rows = 0
    uncovered_test_rows = len(test_rows) - exact_covered_test_rows

    if mode == "overlap":
        test_vocabulary = {symptom for signature in test_counts for symptom in signature}
        selected = {
            index
            for signature, indices in train_groups.items()
            if set(signature) & test_vocabulary
            for index in indices
        }
        exact_selected = {
            index
            for signature in test_counts
            for index in train_groups.get(signature, [])
        }
        fallback_selected = selected - exact_selected
        fallback_covered_test_rows = sum(
            count
            for signature, count in test_counts.items()
            if signature not in train_groups
            and any(set(signature) & set(train_signature) for train_signature in train_groups)
        )
        uncovered_test_rows -= fallback_covered_test_rows

    elif mode == "exact":
        selected = {
            index
            for signature in test_counts
            for index in train_groups.get(signature, [])
        }
        exact_selected = set(selected)

    else:
        # 每种测试症状组合的预算与该组合的测试病例数成正比。
        # 例如 train_per_test=2 且测试集中出现 3 次，则最多保留 6 条训练病例。
        for signature, test_count in sorted(test_counts.items()):
            group = ordered_groups.get(signature)
            if group:
                budget = train_per_test * test_count
                exact_selected.update(group[:budget])

        selected = set(exact_selected)
        missing_signatures = [
            signature for signature in test_counts if signature not in train_groups
        ]

        if nearest_fallback:
            train_signatures = list(train_groups)
            for test_signature in sorted(missing_signatures):
                # 只允许共享至少一个症状的近邻；零交集病例仍视为无关。
                ranked_signatures = sorted(
                    (
                        (_jaccard(test_signature, train_signature), train_signature)
                        for train_signature in train_signatures
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                ranked_signatures = [
                    signature for score, signature in ranked_signatures if score > 0.0
                ]
                if not ranked_signatures:
                    continue

                budget = train_per_test * test_counts[test_signature]
                chosen_for_signature: list[int] = []
                for train_signature in ranked_signatures:
                    for index in ordered_groups[train_signature]:
                        if index not in chosen_for_signature:
                            chosen_for_signature.append(index)
                        if len(chosen_for_signature) >= budget:
                            break
                    if len(chosen_for_signature) >= budget:
                        break

                if chosen_for_signature:
                    fallback_covered_test_rows += test_counts[test_signature]
                    fallback_selected.update(chosen_for_signature)
                    selected.update(chosen_for_signature)

            uncovered_test_rows -= fallback_covered_test_rows

    selected_indices = sorted(selected)
    stats = SelectionStats(
        mode=mode,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        selected_rows=len(selected_indices),
        dropped_rows=len(train_rows) - len(selected_indices),
        retained_ratio=len(selected_indices) / len(train_rows),
        train_signatures=len(train_groups),
        test_signatures=len(test_counts),
        exact_covered_test_rows=exact_covered_test_rows,
        fallback_covered_test_rows=fallback_covered_test_rows,
        uncovered_test_rows=uncovered_test_rows,
        exact_selected_rows=len(exact_selected),
        fallback_selected_rows=len(fallback_selected - exact_selected),
        train_per_test=train_per_test,
        seed=seed,
    )
    return selected_indices, stats


def filter_parquet(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    report_path: Path,
    mode: str,
    train_per_test: int,
    seed: int,
    nearest_fallback: bool,
) -> SelectionStats:
    """读取 VERL Parquet，筛选训练行并保持原始 Arrow schema。"""

    import pyarrow as pa
    import pyarrow.parquet as pq

    train_table = pq.read_table(train_path)
    # 测试集只加载症状字段，从读取层面排除 GT、候选和 reward。
    test_schema = set(pq.read_schema(test_path).names)
    symptom_columns = ["extra_info"] if "extra_info" in test_schema else ["symptoms"]
    test_table = pq.read_table(test_path, columns=symptom_columns)
    train_rows = train_table.to_pylist()
    test_rows = test_table.to_pylist()

    indices, stats = select_train_indices(
        train_rows=train_rows,
        test_rows=test_rows,
        mode=mode,
        train_per_test=train_per_test,
        seed=seed,
        nearest_fallback=nearest_fallback,
    )
    stats.used_test_fields = tuple(symptom_columns)
    if not indices:
        raise ValueError("筛选后训练集为空，请放宽筛选条件")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_table = train_table.take(pa.array(indices, type=pa.int64()))
    pq.write_table(filtered_table, output_path, compression="snappy")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="原始训练 Parquet")
    parser.add_argument("--test", type=Path, required=True, help="测试 Parquet")
    parser.add_argument("--output", type=Path, required=True, help="筛选后的训练 Parquet")
    parser.add_argument("--report", type=Path, required=True, help="筛选统计 JSON")
    parser.add_argument(
        "--mode",
        choices=("overlap", "exact", "matched"),
        default="matched",
    )
    parser.add_argument(
        "--train-per-test",
        type=int,
        default=2,
        help="matched 模式下，每个测试病例最多分配多少条相关训练病例",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-nearest-fallback",
        action="store_true",
        help="测试症状组合无精确训练组时，不使用 Jaccard 最近邻兜底",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = filter_parquet(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        report_path=args.report,
        mode=args.mode,
        train_per_test=args.train_per_test,
        seed=args.seed,
        nearest_fallback=not args.disable_nearest_fallback,
    )
    print("训练子集筛选统计：")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    print(f"筛选结果已写入：{args.output}")
    print(f"统计报告已写入：{args.report}")


if __name__ == "__main__":
    main()
