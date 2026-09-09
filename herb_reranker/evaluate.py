"""用旧版 batch_test 指标评测预测，并导出下一阶段的冻结参考。

预测 JSONL: {"sample_id": "...", "output": "<answer>药名>...</answer>"}。
也接受 ranking 药名列表；不接受候选 ID。导出参考时必须全部为合法完整答案。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .legacy_data import iter_parquet_rows, legacy_ground_truth, legacy_row_id
from .prepare_data import _convert_record, read_jsonl
from .reward import parse_answer, ranking_metrics, validate_reference


def _solution_from_prediction(item: dict, key: str) -> str:
    if "ranking" in item:
        ranking = item["ranking"]
        if not isinstance(ranking, list) or not all(
            isinstance(name, str) for name in ranking
        ):
            raise ValueError(f"{key}: ranking 必须是药名列表")
        return "<answer>" + ">".join(ranking) + "</answer>"
    return str(item.get("output", ""))


def _load_legacy_reward():
    """加载仓库根目录的历史 reward，避免被同名第三方模块遮蔽。"""

    path = Path(__file__).resolve().parents[1] / "reward.py"
    spec = importlib.util.spec_from_file_location("historical_herb_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载旧版 reward: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_legacy_parquet(
    data: Path, predictions: Path
) -> dict[str, float]:
    """用根目录旧 reward 评测已经构造好的 VERL Parquet。"""

    legacy_reward = _load_legacy_reward()

    indexed: dict[str, dict] = {}
    for item in read_jsonl(predictions):
        row_index = item.get("row_index")
        if isinstance(row_index, int) and row_index > 0:
            key = legacy_row_id(row_index)
        else:
            key = item.get("sample_id")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError("旧版预测中的 row_index/sample_id 缺失或重复")
        indexed[key] = item

    sums: dict[str, float] = {}
    seen: set[str] = set()
    for row_number, row in iter_parquet_rows(data):
        key = legacy_row_id(row_number)
        if key not in indexed:
            raise ValueError(f"{key}: 预测缺失")
        if key in seen:
            raise ValueError(f"{key}: 数据重复")
        seen.add(key)
        ground_truth = legacy_ground_truth(row, f"{data} 第 {row_number} 条")
        solution = _solution_from_prediction(indexed[key], key)
        result = legacy_reward.compute_score(
            solution,
            ground_truth,
            row.get("data_source", ""),
            {},
        )
        if not isinstance(result, dict):
            raise ValueError("旧 reward 必须返回包含 score 的字典")
        for name, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sums[name] = sums.get(name, 0.0) + float(value)

    if set(indexed) != seen or not seen:
        raise ValueError("预测与旧版 Parquet 的 row_index 必须完全对应且非空")
    output = {name: value / len(seen) for name, value in sums.items()}
    output["sample_count"] = float(len(seen))
    return output


def evaluate(
    data: Path,
    predictions: Path,
    output_k: int = 20,
    reference_output: Path | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, float]:
    if data.suffix.lower() == ".parquet":
        if reference_output is not None or checkpoint_id is not None:
            raise ValueError("旧版 reward 不支持导出阶段 reference")
        return evaluate_legacy_parquet(data, predictions)

    indexed = {}
    for item in read_jsonl(predictions):
        key = item.get("sample_id")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError("预测中的 sample_id 缺失或重复")
        indexed[key] = item
    sums: dict[str, float] = {}
    references, seen = [], set()
    for number, record in enumerate(read_jsonl(data), 1):
        row, reachable = _convert_record(record, number, output_k, output_k)
        info = row["extra_info"]
        key = info["sample_id"]
        if key in seen or key not in indexed:
            raise ValueError(f"{key}: 数据 ID 重复或预测缺失")
        seen.add(key)
        item = indexed[key]
        solution = _solution_from_prediction(item, key)
        ranking, valid, _ = parse_answer(solution)
        try:
            validate_reference(ranking, info["candidate_herbs"], output_k)
        except ValueError:
            valid = False
        sums["valid_output"] = sums.get("valid_output", 0.0) + float(valid)
        sums["no_reachable_gt"] = sums.get("no_reachable_gt", 0.0) + float(
            not reachable
        )
        gt = row["reward_model"]["ground_truth"]["gt_herbs"]
        for prefix, ranking_list in (
            ("model", ranking),
            ("gnn", info["candidate_herbs"]),
        ):
            for name, value in ranking_metrics(
                ranking_list, gt, info["candidate_herbs"]
            ).items():
                metric = f"{prefix}_{name}"
                sums[metric] = sums.get(metric, 0.0) + value
        if reference_output is not None:
            if not valid:
                raise ValueError(f"{key}: 非法或不足 {output_k} 味的预测不能导出参考")
            if item.get("input_sha256") != info["input_sha256"]:
                raise ValueError(f"{key}: 预测缺少输入指纹或指纹不匹配")
            references.append(
                {
                    "sample_id": key,
                    "ranking": ranking,
                    "input_sha256": info["input_sha256"],
                    "checkpoint_id": checkpoint_id,
                }
            )
    if set(indexed) != seen or not seen:
        raise ValueError("预测与数据的 sample_id 必须完全对应且非空")
    if reference_output is not None:
        if not checkpoint_id or not checkpoint_id.strip():
            raise ValueError("导出参考必须提供 --checkpoint-id")
        reference_output.parent.mkdir(parents=True, exist_ok=True)
        with reference_output.open("w", encoding="utf-8") as handle:
            for item in references:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = {name: value / len(seen) for name, value in sums.items()}
    for name in list(result):
        if name.startswith("model_"):
            result["delta_" + name[6:]] = result[name] - result["gnn_" + name[6:]]
    result["sample_count"] = float(len(seen))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--checkpoint-id")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.data,
                args.predictions,
                args.output_k,
                args.reference_output,
                args.checkpoint_id,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
