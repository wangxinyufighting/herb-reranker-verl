"""验证 Parquet 的药名协议、阶段和冻结参考，拒绝静默混用旧 ID 数据。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .prepare_data import SYSTEM_PROMPT, build_user_prompt, input_fingerprint
from .reward import PROTOCOL, STAGES, names, normalize_stage, validate_reference


def validate_protocol_row(
    row: Mapping[str, Any],
    source: Path,
    row_index: int,
    expected_output_k: int = 20,
    training_stage: str | None = None,
) -> None:
    prefix = f"{source} 第 {row_index} 条"
    info = row.get("extra_info")
    if not isinstance(info, Mapping) or info.get("output_protocol") != PROTOCOL:
        raise ValueError(f"{prefix}: 不是药名协议，请重新构建 Parquet")
    if info.get("metric_convention") != "batch_test_method1_kmax20":
        raise ValueError(f"{prefix}: metric 口径不匹配")
    if expected_output_k < 20 or info.get("output_k") != expected_output_k:
        raise ValueError(f"{prefix}: output_k 与训练配置不匹配")
    stage = normalize_stage(info.get("training_stage"))
    if training_stage is not None and stage != normalize_stage(training_stage):
        raise ValueError(f"{prefix}: training_stage 与训练配置不匹配")
    candidates = names(info.get("candidate_herbs"), "candidate_herbs")
    gnn = names(info.get("gnn_candidate_herbs"), "gnn_candidate_herbs")
    if (
        candidates != gnn
        or len(set(candidates)) != len(candidates)
        or len(candidates) < expected_output_k
    ):
        raise ValueError(f"{prefix}: 候选列表不是唯一药名的 GNN 原始顺序")
    if info.get("input_sha256") != input_fingerprint(dict(info)):
        raise ValueError(f"{prefix}: 输入指纹不匹配")
    if stage != "stage1":
        validate_reference(
            names(info.get("reference_ranking"), "reference_ranking"),
            candidates,
            expected_output_k,
        )
        if not info.get("reference_checkpoint"):
            raise ValueError(f"{prefix}: 缺少 reference_checkpoint")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not all(
        isinstance(message, Mapping) for message in prompt
    ):
        raise ValueError(f"{prefix}: prompt 必须是消息列表")
    text = "\n".join(str(message.get("content", "")) for message in prompt)
    expected_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                info["symptoms"],
                info["symptom_description"],
                candidates,
                expected_output_k,
            ),
        },
    ]
    if prompt != expected_prompt:
        raise ValueError(f"{prefix}: Prompt 与当前构建协议不一致，请重新构建")
    if (
        "原始症状描述：" + str(info["symptom_description"]) not in text
        or "候选中药（初排顺序）：" + "、".join(candidates) not in text
        or "<answer>" not in text
        or "不得输出药名" in text
        or "候选ID" in text
    ):
        raise ValueError(f"{prefix}: Prompt 与药名协议不匹配")


def validate_parquet(
    path: Path, expected_output_k: int = 20, training_stage: str | None = None
) -> int:
    import pyarrow.parquet as pq

    count = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1024, columns=["prompt", "extra_info"]
    ):
        for row in batch.to_pylist():
            count += 1
            validate_protocol_row(row, path, count, expected_output_k, training_stage)
    if not count:
        raise ValueError(f"{path} 不包含任何样本")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-output-k", type=int, default=20)
    parser.add_argument("--training-stage", choices=STAGES)
    args = parser.parse_args()
    for path in args.files:
        count = validate_parquet(path, args.expected_output_k, args.training_stage)
        print(f"协议检查通过: {path} ({count} 条)")


if __name__ == "__main__":
    main()
