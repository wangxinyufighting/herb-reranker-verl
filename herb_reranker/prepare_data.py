"""构造原始症状描述输入、中药名称输出的 VERL 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .reward import PROTOCOL, STAGES, names, normalize_stage, validate_reference

SYSTEM_PROMPT = "你是一个拥有丰富经验的中医专家。请根据患者原始症状描述，对候选中药进行相关性重排。只能使用候选中药的原始名称。"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: 必须是 JSON 对象")
                rows.append(row)
    return rows


def input_fingerprint(record: dict[str, Any]) -> str:
    payload = {
        "protocol": PROTOCOL,
        "system": SYSTEM_PROMPT,
        "user": build_user_prompt(
            record["symptoms"],
            record["symptom_description"],
            record["candidate_herbs"],
            record.get("output_k", 20),
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def build_user_prompt(
    symptoms: list[str], description: str, candidates: list[str], output_k: int = 20
) -> str:
    return (
        f"患者症状列表：{'、'.join(symptoms)}\n原始症状描述：{description}\n"
        f"候选中药（初排顺序）：{'、'.join(candidates)}\n\n"
        f"请根据原始症状描述，从这 {len(candidates)} 味候选中选出最相关的 {output_k} 味并排序。\n"
        "【要求】\n1. 可以在 <think> 和 </think> 之间给出不超过200字的简洁说明。\n"
        "2. 最终结果写在 <answer> 和 </answer> 之间，使用 '>' 连接药名。\n"
        f"3. 至少包含 {output_k} 味互不重复的候选中药；建议只输出这 {output_k} 味。\n"
        "4. 不得输出序号、JSON、别名、剂量、省略号或候选以外的药。"
    )


def _convert_record(
    record: dict[str, Any],
    line_no: int,
    min_candidates: int,
    output_k: int = 20,
    training_stage: str = "stage1",
    reference: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    stage = normalize_stage(training_stage)
    sample_id = record.get("sample_id")
    description = record.get("symptom_description")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"第 {line_no} 行缺少 sample_id")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"第 {line_no} 行缺少 symptom_description")
    symptoms = names(record.get("symptoms"), "symptoms")
    candidates = names(record.get("candidate_herbs"), "candidate_herbs")
    gt = names(
        record.get("ground_truth_herbs", record.get("gt_herbs")), "ground_truth_herbs"
    )
    if len(set(candidates)) != len(candidates) or len(set(gt)) != len(gt):
        raise ValueError(f"第 {line_no} 行含重复中药")
    if len(candidates) < min_candidates:
        raise ValueError(f"第 {line_no} 行候选少于 min_candidates={min_candidates}")
    if output_k < 20:
        raise ValueError("output_k 必须至少为20，与 k_max=20 对齐")
    if len(candidates) < output_k:
        raise ValueError(f"第 {line_no} 行候选少于 output_k={output_k}")
    if any("<" in herb or ">" in herb for herb in candidates):
        raise ValueError("候选药名不能包含协议分隔符")
    info = {
        "sample_id": sample_id.strip(),
        "symptoms": symptoms,
        "output_k": output_k,
        "symptom_description": description.strip(),
        "candidate_herbs": candidates,
    }
    fingerprint = input_fingerprint(info)
    ranking, checkpoint = [], ""
    if stage != "stage1":
        if reference is None:
            raise ValueError(f"{sample_id}: 缺少上一阶段 reference")
        if reference.get("input_sha256") != fingerprint:
            raise ValueError(f"{sample_id}: reference 输入指纹不匹配")
        ranking = names(reference.get("ranking"), "reference ranking")
        validate_reference(ranking, candidates, output_k)
        checkpoint = reference.get("checkpoint_id")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"{sample_id}: reference 缺少 checkpoint_id")
    info.update(
        {
            "gnn_candidate_herbs": candidates,
            "candidate_k": len(candidates),
            "output_k": output_k,
            "output_protocol": PROTOCOL,
            "metric_convention": "batch_test_method1_kmax20",
            "training_stage": stage,
            "input_sha256": fingerprint,
            "reference_ranking": ranking,
            "reference_checkpoint": checkpoint,
        }
    )
    row = {
        "data_source": "ptm_herb_rerank",
        "ability": "topk_listwise_reranking",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    symptoms, description.strip(), candidates, output_k
                ),
            },
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {"candidate_herbs": candidates, "gt_herbs": gt},
        },
        "extra_info": info,
    }
    return row, len(set(candidates) & set(gt))


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    min_candidates: int = 50,
    unreachable_policy: str = "drop",
    output_k: int = 20,
    training_stage: str = "stage1",
    reference_jsonl: Path | None = None,
) -> None:
    stage = normalize_stage(training_stage)
    if unreachable_policy not in {"drop", "keep", "error"} or min_candidates <= 0:
        raise ValueError("无效的数据构建参数")
    references = {}
    if reference_jsonl:
        if stage == "stage1":
            raise ValueError("stage1 使用 GNN 参考，不接受 reference-jsonl")
        for item in read_jsonl(reference_jsonl):
            key = item.get("sample_id")
            if not isinstance(key, str) or not key or key in references:
                raise ValueError("reference 的 sample_id 缺失或重复")
            references[key] = item
        if len({item.get("checkpoint_id") for item in references.values()}) != 1:
            raise ValueError("reference 必须来自同一个冻结 checkpoint")
    rows, seen, dropped = [], set(), 0
    for line_no, record in enumerate(read_jsonl(input_path), 1):
        checked, reachable = _convert_record(record, line_no, min_candidates, output_k)
        key = checked["extra_info"]["sample_id"]
        if key in seen:
            raise ValueError(f"sample_id 重复: {key}")
        seen.add(key)
        if not reachable and unreachable_policy == "error":
            raise ValueError(f"{key}: 候选集与 GT 无交集")
        if not reachable and unreachable_policy == "drop":
            dropped += 1
            continue
        if stage != "stage1":
            checked, _ = _convert_record(
                record, line_no, min_candidates, output_k, stage, references.get(key)
            )
        rows.append(checked)
    if set(references) - seen:
        raise ValueError("reference 包含不属于此数据集的 sample_id")
    if not rows:
        raise ValueError("转换后没有可写入的样本")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="snappy")
    print(
        f"已写入 {len(rows)} 条 {stage} 样本到 {output_path}；无可达 GT 丢弃 {dropped} 条。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-candidates", type=int, default=50)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument(
        "--unreachable-policy", choices=("drop", "keep", "error"), default="drop"
    )
    parser.add_argument("--training-stage", choices=STAGES, default="stage1")
    parser.add_argument("--reference-jsonl", type=Path)
    args = parser.parse_args()
    convert_jsonl(
        args.input,
        args.output,
        args.min_candidates,
        args.unreachable_policy,
        args.output_k,
        args.training_stage,
        args.reference_jsonl,
    )


if __name__ == "__main__":
    main()
