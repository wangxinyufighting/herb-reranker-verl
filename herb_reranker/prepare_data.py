"""把病例 JSONL 转换为 VERL 可直接读取的 Parquet 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是一个中药候选重排器。请依据规范症状、原始症状描述和 GNN 候选列表，输出最相关候选的序号排序。
必须遵守以下规则：
1. ranking 中只能输出整数序号，不得输出药名；
2. 输出数量必须等于用户指定的 Top-K；
3. 序号必须来自候选列表，且不得重复；
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


def _candidate_id_view(
    sample_id: str, gnn_candidates: list[str]
) -> tuple[list[str], list[tuple[int, int, str]]]:
    """为候选分配可复现的非顺序 ID，消除 ``[1,2,...]`` 复制捷径。

    候选行仍按 GNN 名次展示，因此原始召回顺序没有丢失；只是输出 ID 与名次不再
    相同。排序由 SHA-256 决定，不依赖 Python 的随机种子或进程状态，同一病例在
    train/validation/inference 中始终得到相同 ID。

    返回：按 ID 索引的候选表，以及按 GNN 名次展示的 ``(ID, 名次, 药名)``。
    """

    candidate_count = len(gnn_candidates)
    candidate_ids = list(range(1, candidate_count + 1))
    candidate_ids.sort(
        key=lambda candidate_id: hashlib.sha256(
            f"{sample_id}:{candidate_id}".encode("utf-8")
        ).digest()
    )

    candidates_by_id = [""] * candidate_count
    display_rows: list[tuple[int, int, str]] = []
    for gnn_rank, (candidate_id, herb) in enumerate(
        zip(candidate_ids, gnn_candidates), start=1
    ):
        candidates_by_id[candidate_id - 1] = herb
        display_rows.append((candidate_id, gnn_rank, herb))
    return candidates_by_id, display_rows


def _build_user_prompt(
    symptoms: list[str],
    symptom_description: str,
    display_rows: list[tuple[int, int, str]],
    output_k: int,
) -> str:
    """构造模型输入；真实药方绝不会进入提示词。"""

    numbered_candidates = "\n".join(
        f"候选ID {candidate_id} | GNN名次 {gnn_rank} | {herb}"
        for candidate_id, gnn_rank, herb in display_rows
    )
    candidate_count = len(display_rows)
    return (
        "/no_think\n"
        f"规范症状：{'、'.join(symptoms)}\n"
        f"原始症状描述：{symptom_description}\n"
        "GNN 候选中药（按 GNN 名次展示；候选ID是输出时使用的标识）：\n"
        f"{numbered_candidates}\n"
        f"请从 {candidate_count} 个候选中选出最相关的 {output_k} 个并重新排序。"
        f"ranking 必须恰好包含 {output_k} 个互不重复的整数候选ID，"
        f"每个ID必须在 1 到 {candidate_count} 之间。不得输出药名。\n"
        "只输出一个含 ranking 字段的 JSON 对象；不要输出代码块、省略号或其他字段。"
    )


def _convert_record(
    record: dict[str, Any], line_no: int, min_candidates: int, output_k: int = 20
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
    if output_k <= 0:
        raise ValueError("output_k 必须大于 0")
    if output_k > len(candidates):
        raise ValueError(
            f"第 {line_no} 行只有 {len(candidates)} 个候选，少于 output_k={output_k}"
        )

    reachable_count = len(set(candidates) & set(ground_truth))
    candidates_by_id, display_rows = _candidate_id_view(sample_id, candidates)
    prompt = _build_user_prompt(symptoms, description, display_rows, output_k)
    converted = {
        "data_source": "ptm_herb_rerank",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "topk_listwise_reranking",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"ground_truth_herbs": ground_truth},
        },
        "extra_info": {
            "sample_id": sample_id,
            "symptoms": symptoms,
            "symptom_description": description,
            # candidate_herbs 按候选 ID 排列，reward 用 ranking 中的 ID 查询药名。
            "candidate_herbs": candidates_by_id,
            # GNN 顺序单独保存，用于计算不随 ID 编排变化的原始 baseline。
            "gnn_candidate_herbs": candidates,
            "candidate_k": len(candidates),
            "output_k": output_k,
            "candidate_id_scheme": "deterministic_permuted_v1",
            "retriever": "gnn",
        },
    }
    return converted, reachable_count


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    min_candidates: int,
    unreachable_policy: str,
    output_k: int = 20,
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

            row, reachable_count = _convert_record(
                raw, line_no, min_candidates, output_k=output_k
            )
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
    parser.add_argument(
        "--output-k",
        type=int,
        default=20,
        help="模型必须输出的候选数量；默认 20，与最高评测 cutoff 对齐",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_candidates <= 0:
        raise ValueError("--min-candidates 必须大于 0")
    if args.output_k <= 0:
        raise ValueError("--output-k 必须大于 0")
    convert_jsonl(
        input_path=args.input,
        output_path=args.output,
        min_candidates=args.min_candidates,
        unreachable_policy=args.unreachable_policy,
        output_k=args.output_k,
    )


if __name__ == "__main__":
    main()
