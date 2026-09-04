"""把病例 JSONL 转换为 VERL 可直接读取的 Parquet 数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是一个中药候选重排器。请依据规范症状、原始症状描述和 GNN 候选列表，输出候选序号的新顺序。
必须遵守以下规则：
1. ranking 中只能输出整数序号，不得输出药名；
2. 每个候选序号必须且只能出现一次；
3. 不得增加、删除或重复序号；
4. 越相关的中药排得越靠前；
5. 只输出一行合法 JSON，不要解释。"""


def _require_string_list(record: dict[str, Any], field: str, line_no: int) -> list[str]:
    """读取并严格检查字符串列表，尽早暴露数据对齐错误。"""

    value = record.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"第 {line_no} 行的 {field!r} 必须是非空列表")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"第 {line_no} 行的 {field!r} 含空值或非字符串元素")
        cleaned.append(item.strip())
    return cleaned


def _build_user_prompt(
    symptoms: list[str], symptom_description: str, candidate_herbs: list[str]
) -> str:
    """构造模型输入；真实药方绝不会进入提示词。"""

    numbered_candidates = "\n".join(
        f"{index}. {herb}" for index, herb in enumerate(candidate_herbs, start=1)
    )
    return (
        "/no_think\n"
        f"规范症状：{'、'.join(symptoms)}\n"
        f"原始症状描述：{symptom_description}\n"
        "GNN 候选中药（当前顺序仅作为召回器先验）：\n"
        f"{numbered_candidates}\n"
        f"请只输出一个含 ranking 字段的 JSON 对象；ranking 必须是长度为 "
        f"{len(candidate_herbs)} 的整数数组，并且是 1 到 {len(candidate_herbs)} "
        "所有序号的一个完整排列。不得输出药名。\n"
        '输出示例：{"ranking":[3,1,2,4]}（示例仅说明格式，实际必须输出全部序号）。'
    )


def _convert_record(
    record: dict[str, Any], line_no: int, min_candidates: int
) -> tuple[dict[str, Any], int]:
    """检查一条原始记录并构造 VERL 样本，同时返回可达 GT 数量。"""

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"第 {line_no} 行缺少非空 sample_id")
    sample_id = sample_id.strip()

    symptoms = _require_string_list(record, "symptoms", line_no)
    candidates = _require_string_list(record, "candidate_herbs", line_no)
    ground_truth = _require_string_list(record, "ground_truth_herbs", line_no)

    description = record.get("symptom_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"第 {line_no} 行缺少非空 symptom_description")
    description = description.strip()

    if len(candidates) < min_candidates:
        raise ValueError(
            f"第 {line_no} 行只有 {len(candidates)} 个候选，少于 --min-candidates={min_candidates}"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"第 {line_no} 行 candidate_herbs 含重复中药")
    if len(set(ground_truth)) != len(ground_truth):
        raise ValueError(f"第 {line_no} 行 ground_truth_herbs 含重复中药")

    reachable_count = len(set(candidates) & set(ground_truth))
    prompt = _build_user_prompt(symptoms, description, candidates)
    converted = {
        "data_source": "ptm_herb_rerank",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "listwise_reranking",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"ground_truth_herbs": ground_truth},
        },
        "extra_info": {
            "sample_id": sample_id,
            "symptoms": symptoms,
            "symptom_description": description,
            "candidate_herbs": candidates,
            "candidate_k": len(candidates),
            "retriever": "gnn",
        },
    }
    return converted, reachable_count


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    min_candidates: int,
    unreachable_policy: str,
) -> None:
    """转换完整文件，并按策略处理候选与 GT 无交集的病例。"""

    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    dropped = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"第 {line_no} 行必须是 JSON 对象")

            row, reachable_count = _convert_record(raw, line_no, min_candidates)
            sample_id = row["extra_info"]["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"sample_id 重复: {sample_id}")
            sample_ids.add(sample_id)

            if reachable_count == 0:
                if unreachable_policy == "error":
                    raise ValueError(f"样本 {sample_id} 的候选集与 GT 完全没有交集")
                if unreachable_policy == "drop":
                    dropped += 1
                    continue
            rows.append(row)

    if not rows:
        raise ValueError("转换后没有可写入的样本")

    # 直接使用 PyArrow，避免 datasets 为内存表生成指纹时引入额外序列化依赖。
    # VERL 最终读取的仍是完全标准的 Parquet，数据结构没有变化。
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path), compression="snappy")
    print(
        f"已写入 {len(rows)} 条样本到 {output_path}；"
        f"无可达 GT 而丢弃 {dropped} 条。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="输出 Parquet")
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=20,
        help="每条样本允许的最少候选数；Top-50 数据可改为 50",
    )
    parser.add_argument(
        "--unreachable-policy",
        choices=("drop", "keep", "error"),
        default="drop",
        help="候选集与 GT 无交集时：丢弃、保留或报错",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_candidates <= 0:
        raise ValueError("--min-candidates 必须大于 0")
    convert_jsonl(
        input_path=args.input,
        output_path=args.output,
        min_candidates=args.min_candidates,
        unreachable_policy=args.unreachable_policy,
    )


if __name__ == "__main__":
    main()
