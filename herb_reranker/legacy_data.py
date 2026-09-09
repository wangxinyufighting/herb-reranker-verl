"""读取现有旧版 VERL Parquet 的小型适配层。

旧版数据已经是 VERL 可直接消费的 Parquet，不需要经过新的 JSONL/阶段构建流程。
本模块只负责严格读取 ``prompt`` 和 ``reward_model.ground_truth``，不修改数据内容。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator


def iter_parquet_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """按文件顺序读取旧版 Parquet 行，返回 1-based 行号。"""

    import pyarrow.parquet as pq

    required = {"prompt", "reward_model"}
    schema = set(pq.read_schema(path).names)
    missing = required - schema
    if missing:
        raise ValueError(f"{path} 缺少旧版 VERL 字段: {sorted(missing)}")
    columns = ["prompt", "reward_model"]
    if "data_source" in schema:
        columns.append("data_source")
    if "symptoms" in schema:
        columns.append("symptoms")
    row_number = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1024, columns=columns
    ):
        for row in batch.to_pylist():
            row_number += 1
            yield row_number, row
    if row_number == 0:
        raise ValueError(f"{path} 不包含任何样本")


def legacy_ground_truth(row: Mapping[str, Any], source: str = "row") -> dict[str, Any]:
    """提取旧 reward 所需的 candidate_herbs/gt_herbs。"""

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise ValueError(f"{source}: reward_model 必须是对象")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError(f"{source}: reward_model.ground_truth 必须是对象")
    candidates = ground_truth.get("candidate_herbs")
    gt = ground_truth.get("gt_herbs")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{source}: candidate_herbs 必须是非空列表")
    if not isinstance(gt, list):
        raise ValueError(f"{source}: gt_herbs 必须是列表")
    if any(not isinstance(name, str) or not name.strip() for name in candidates):
        raise ValueError(f"{source}: candidate_herbs 含空值或非字符串")
    if any(not isinstance(name, str) or not name.strip() for name in gt):
        raise ValueError(f"{source}: gt_herbs 含空值或非字符串")
    if len(candidates) < 20:
        raise ValueError(f"{source}: candidate_herbs 至少需要 20 味药")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{source}: candidate_herbs 含重复药名")
    return dict(ground_truth)


def legacy_prompt(row: Mapping[str, Any], source: str = "row") -> list[dict[str, str]]:
    """校验并返回旧版 prompt 消息，不重写治疗法提示词。"""

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{source}: prompt 必须是非空消息列表")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(prompt, 1):
        if not isinstance(message, Mapping):
            raise ValueError(f"{source}: prompt 第 {index} 条不是消息对象")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{source}: prompt 第 {index} 条缺少 role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{source}: prompt 第 {index} 条缺少 content")
        messages.append({"role": role, "content": content})
    return messages


def legacy_row_id(row_number: int) -> str:
    """旧数据没有 sample_id，使用稳定的 1-based 行号作为预测键。"""

    return f"row-{row_number}"
