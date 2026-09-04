"""训练前检查 Parquet 是否使用当前 Top-K 候选 ID 协议。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_ID_SCHEME = "deterministic_permuted_v1"


def validate_protocol_row(
    row: Mapping[str, Any], source: Path, row_index: int, expected_output_k: int
) -> None:
    """检查一条 VERL 样本；失败时给出可直接操作的重建提示。"""

    prefix = f"{source} 第 {row_index} 条"
    info = row.get("extra_info")
    if not isinstance(info, Mapping):
        raise ValueError(f"{prefix} 缺少 extra_info")

    if info.get("candidate_id_scheme") != EXPECTED_ID_SCHEME:
        raise ValueError(
            f"{prefix} 不是新版候选 ID 协议；请重新运行 build_train_parquet.sh "
            "和 build_test_parquet.sh"
        )
    if info.get("output_k") != expected_output_k:
        raise ValueError(
            f"{prefix} 的 output_k={info.get('output_k')!r}，"
            f"但训练配置要求 {expected_output_k}"
        )

    candidates = info.get("candidate_herbs")
    gnn_candidates = info.get("gnn_candidate_herbs")
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or isinstance(gnn_candidates, (str, bytes))
        or not isinstance(gnn_candidates, Sequence)
        or len(candidates) != len(gnn_candidates)
        or set(candidates) != set(gnn_candidates)
    ):
        raise ValueError(f"{prefix} 的候选 ID 表与 GNN 原顺序不一致")
    if expected_output_k > len(candidates):
        raise ValueError(
            f"{prefix} 只有 {len(candidates)} 个候选，少于 output_k={expected_output_k}"
        )

    prompt = row.get("prompt")
    if isinstance(prompt, (str, bytes)) or not isinstance(prompt, Sequence):
        raise ValueError(f"{prefix} 的 prompt 不是消息列表")
    prompt_text = "\n".join(
        str(message.get("content", ""))
        for message in prompt
        if isinstance(message, Mapping)
    )
    if "GNN名次" not in prompt_text or f"最相关的 {expected_output_k} 个" not in prompt_text:
        raise ValueError(f"{prefix} 的 Prompt 仍是旧版完整排列协议，请重新构建 Parquet")


def validate_parquet(path: Path, expected_output_k: int) -> int:
    """流式检查一个 Parquet，返回样本数。"""

    import pyarrow.parquet as pq

    row_count = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=1024, columns=["prompt", "extra_info"]
    ):
        for row in batch.to_pylist():
            row_count += 1
            validate_protocol_row(row, path, row_count, expected_output_k)
    if row_count == 0:
        raise ValueError(f"{path} 不包含任何样本")
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-output-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_output_k <= 0:
        raise ValueError("--expected-output-k 必须大于 0")
    for path in args.files:
        count = validate_parquet(path, args.expected_output_k)
        print(f"协议检查通过: {path} ({count} 条, output_k={args.expected_output_k})")


if __name__ == "__main__":
    main()
