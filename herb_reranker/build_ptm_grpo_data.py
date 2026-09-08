"""合并 PTM 症状上下文与 GNN 候选列表，并生成 VERL/GRPO Parquet。

GNN 文件的每一行格式为：

    <症状ID，以空格分隔>\t<按相关性排序的中药ID，以空格分隔>

脚本不会只依赖行号盲目拼接，而会同时校验两侧的症状 ID。候选中药 ID 通过
``herb_mapping.txt`` 转换成药名，最终生成现有 ``prepare_data.py`` 约定的 JSONL，
再复用同一转换逻辑写成 VERL Parquet。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from .prepare_data import convert_jsonl


@dataclass
class BuildStats:
    """数据对齐与候选召回统计。"""

    input_rows: int = 0
    written_rows: int = 0
    dropped_unreachable_rows: int = 0
    candidate_k: int = 0
    ground_truth_count: int = 0
    reachable_ground_truth_count: int = 0
    macro_candidate_recall: float = 0.0
    micro_candidate_recall: float = 0.0
    alignment_mode: str = ""


def load_herb_mapping(path: Path) -> dict[int, str]:
    """读取 ``药名 ID`` 映射；药名允许包含空格，因此从行尾切分一次。"""

    mapping: dict[int, str] = {}
    seen_names: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                name, raw_id = text.rsplit(maxsplit=1)
                herb_id = int(raw_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行不是合法的“药名 ID”格式"
                ) from exc
            if herb_id in mapping:
                raise ValueError(f"{path} 第 {line_no} 行出现重复中药 ID: {herb_id}")
            if name in seen_names:
                raise ValueError(f"{path} 第 {line_no} 行出现重复药名: {name}")
            mapping[herb_id] = name
            seen_names.add(name)

    if not mapping:
        raise ValueError(f"中药映射为空: {path}")
    return mapping


def iter_context_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """逐行读取症状列表、原始文本和 GT 药方。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_no} 行必须是 JSON 对象")
            yield line_no, row


def iter_retrieval_rows(path: Path) -> Iterator[tuple[int, list[int], list[int]]]:
    """读取 GNN 输出，返回行号、症状 ID 和有序候选中药 ID。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\r\n")
            if not text:
                raise ValueError(f"{path} 第 {line_no} 行为空，无法保证按行对齐")
            fields = text.split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path} 第 {line_no} 行必须且只能包含一个制表符")
            try:
                symptom_ids = [int(item) for item in fields[0].split()]
                candidate_ids = [int(item) for item in fields[1].split()]
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_no} 行含非整数 ID") from exc
            if not symptom_ids or not candidate_ids:
                raise ValueError(f"{path} 第 {line_no} 行症状或候选列表为空")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"{path} 第 {line_no} 行候选中药 ID 重复")
            yield line_no, symptom_ids, candidate_ids


def _required_string_list(row: dict[str, Any], field: str, line_no: int) -> list[str]:
    """检查上下文中的核心列表字段。"""

    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"上下文第 {line_no} 行的 {field!r} 必须是非空列表")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"上下文第 {line_no} 行的 {field!r} 含空值或非字符串")
    return [item.strip() for item in value]


def merge_to_jsonl(
    context_jsonl: Path,
    retrieval_file: Path,
    herb_mapping_file: Path,
    output_jsonl: Path,
    candidate_k: int,
    unreachable_policy: str = "drop",
) -> BuildStats:
    """严格对齐三类信息，输出 ``prepare_data.py`` 约定的原始 JSONL。"""

    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")
    if unreachable_policy not in {"drop", "keep", "error"}:
        raise ValueError("unreachable_policy 必须是 drop、keep 或 error")

    herb_mapping = load_herb_mapping(herb_mapping_file)
    output_path = output_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = BuildStats(candidate_k=candidate_k, alignment_mode="symptom_ids_by_row")
    macro_recalls: list[float] = []

    context_iter = iter_context_rows(context_jsonl)
    retrieval_iter = iter_retrieval_rows(retrieval_file)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for pair_index, pair in enumerate(
            zip_longest(context_iter, retrieval_iter), start=1
        ):
            context_item, retrieval_item = pair
            if context_item is None or retrieval_item is None:
                raise ValueError(
                    "上下文与 GNN 文件行数不同："
                    f"在第 {pair_index} 个有效样本处发现一侧提前结束"
                )

            context_line, context = context_item
            retrieval_line, retrieval_symptom_ids, candidate_ids = retrieval_item
            stats.input_rows += 1

            raw_context_symptom_ids = context.get("symptom_ids")
            if not isinstance(raw_context_symptom_ids, list):
                raise ValueError(f"上下文第 {context_line} 行缺少 symptom_ids")
            try:
                context_symptom_ids = [int(item) for item in raw_context_symptom_ids]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"上下文第 {context_line} 行含非整数 symptom_ids"
                ) from exc
            if context_symptom_ids != retrieval_symptom_ids:
                raise ValueError(
                    "症状 ID 对齐失败："
                    f"上下文第 {context_line} 行为 {context_symptom_ids}，"
                    f"GNN 第 {retrieval_line} 行为 {retrieval_symptom_ids}"
                )

            if len(candidate_ids) < candidate_k:
                raise ValueError(
                    f"GNN 第 {retrieval_line} 行只有 {len(candidate_ids)} 个候选，"
                    f"少于 candidate_k={candidate_k}"
                )
            selected_ids = candidate_ids[:candidate_k]
            unknown_ids = [
                herb_id for herb_id in selected_ids if herb_id not in herb_mapping
            ]
            if unknown_ids:
                raise ValueError(
                    f"GNN 第 {retrieval_line} 行含未映射的中药 ID: {unknown_ids[:10]}"
                )
            candidate_herbs = [herb_mapping[herb_id] for herb_id in selected_ids]

            symptoms = _required_string_list(context, "symptoms", context_line)
            ground_truth = _required_string_list(
                context, "ground_truth_herbs", context_line
            )
            description = context.get("symptom_description")
            sample_id = context.get("sample_id")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"上下文第 {context_line} 行缺少 symptom_description")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"上下文第 {context_line} 行缺少 sample_id")

            ground_truth_set = set(ground_truth)
            reachable_count = len(ground_truth_set & set(candidate_herbs))
            stats.ground_truth_count += len(ground_truth_set)
            stats.reachable_ground_truth_count += reachable_count
            macro_recalls.append(reachable_count / len(ground_truth_set))

            if reachable_count == 0:
                if unreachable_policy == "error":
                    raise ValueError(
                        f"样本 {sample_id} 的 Top-{candidate_k} 与 GT 无交集"
                    )
                if unreachable_policy == "drop":
                    stats.dropped_unreachable_rows += 1
                    continue

            merged = {
                "sample_id": sample_id.strip(),
                "symptoms": symptoms,
                "symptom_description": description.strip(),
                "candidate_herbs": candidate_herbs,
                "ground_truth_herbs": ground_truth,
            }
            output_handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            stats.written_rows += 1

    if stats.written_rows == 0:
        raise ValueError("对齐后没有可写入样本")
    stats.micro_candidate_recall = (
        stats.reachable_ground_truth_count / stats.ground_truth_count
        if stats.ground_truth_count
        else 0.0
    )
    stats.macro_candidate_recall = (
        sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    )
    return stats


def get_candidate_herbs(candidates_file: Path, can_num: int) -> dict[str, list[str]]:
    """Read symptom-name<TAB>candidate-name lists without last-row-wins joins."""
    mapping = {}
    for number, line in enumerate(
        Path(candidates_file).read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{candidates_file}:{number}: 需要一个制表符")
        key = " ".join(fields[0].split())
        herbs = fields[1].split()[:can_num]
        if not key or len(herbs) < can_num or len(herbs) != len(set(herbs)):
            raise ValueError(f"{candidates_file}:{number}: 症状或候选列表无效")
        if key in mapping and mapping[key] != herbs:
            raise ValueError(f"{candidates_file}:{number}: 相同症状存在冲突候选顺序")
        mapping[key] = herbs
    return mapping


def merge_names_to_jsonl(
    context_jsonl: Path,
    candidates_file: Path,
    output_jsonl: Path,
    candidate_k: int = 50,
    unreachable_policy: str = "drop",
) -> BuildStats:
    """Keep each original-description case, even when normalized symptoms match."""
    if candidate_k <= 0 or unreachable_policy not in {"drop", "keep", "error"}:
        raise ValueError("无效的数据构建参数")
    contexts = list(iter_context_rows(context_jsonl))
    entries = []
    for number, line in enumerate(
        candidates_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{candidates_file}:{number}: 需要一个制表符")
        key, herbs = " ".join(fields[0].split()), fields[1].split()[:candidate_k]
        if not key or len(herbs) < candidate_k or len(herbs) != len(set(herbs)):
            raise ValueError(f"{candidates_file}:{number}: 症状或候选列表无效")
        entries.append((key, herbs))
    by_row = len(entries) == len(contexts)
    mapping = {} if by_row else get_candidate_herbs(candidates_file, candidate_k)
    stats = BuildStats(
        candidate_k=candidate_k,
        alignment_mode="symptom_names_by_row" if by_row else "unique_symptom_lookup",
    )
    rows, recalls, seen = [], [], set()
    for index, (number, context) in enumerate(contexts):
        symptoms = _required_string_list(context, "symptoms", number)
        gt = _required_string_list(context, "ground_truth_herbs", number)
        key = " ".join(symptoms)
        if by_row and entries[index][0] != key:
            raise ValueError(f"上下文第 {number} 行与候选文件的症状名称对齐失败")
        if not by_row and key not in mapping:
            raise ValueError(f"上下文第 {number} 行没有匹配的候选列表: {key}")
        sample_id = context.get("sample_id")
        description = context.get("symptom_description")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"上下文第 {number} 行 sample_id 缺失或重复")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"上下文第 {number} 行缺少 symptom_description")
        seen.add(sample_id)
        candidates = entries[index][1] if by_row else mapping[key]
        reachable = len(set(gt) & set(candidates))
        stats.input_rows += 1
        stats.ground_truth_count += len(set(gt))
        stats.reachable_ground_truth_count += reachable
        recalls.append(reachable / len(set(gt)))
        if not reachable and unreachable_policy == "error":
            raise ValueError(f"{sample_id}: 候选集与 GT 无交集")
        if not reachable and unreachable_policy == "drop":
            stats.dropped_unreachable_rows += 1
            continue
        rows.append(
            {
                "sample_id": sample_id.strip(),
                "symptoms": symptoms,
                "symptom_description": description.strip(),
                "candidate_herbs": candidates,
                "ground_truth_herbs": gt,
            }
        )
    if not rows:
        raise ValueError("对齐后没有可写入样本")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.written_rows = len(rows)
    stats.micro_candidate_recall = (
        stats.reachable_ground_truth_count / stats.ground_truth_count
    )
    stats.macro_candidate_recall = sum(recalls) / len(recalls)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-jsonl", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--retrieval-file", type=Path)
    source.add_argument("--candidate-names-file", type=Path)
    parser.add_argument("--herb-mapping", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument(
        "--unreachable-policy", choices=("drop", "keep", "error"), default="drop"
    )
    parser.add_argument(
        "--training-stage", choices=("stage1", "stage2", "stage3"), default="stage1"
    )
    parser.add_argument("--reference-jsonl", type=Path)
    args = parser.parse_args()
    if args.candidate_names_file:
        stats = merge_names_to_jsonl(
            args.context_jsonl,
            args.candidate_names_file,
            args.output_jsonl,
            args.candidate_k,
            args.unreachable_policy,
        )
    else:
        if args.herb_mapping is None:
            parser.error("--retrieval-file 需要 --herb-mapping")
        stats = merge_to_jsonl(
            args.context_jsonl,
            args.retrieval_file,
            args.herb_mapping,
            args.output_jsonl,
            args.candidate_k,
            args.unreachable_policy,
        )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    convert_jsonl(
        args.output_jsonl,
        args.output_parquet,
        args.candidate_k,
        "keep",
        args.output_k,
        args.training_stage,
        args.reference_jsonl,
    )


if __name__ == "__main__":
    main()
