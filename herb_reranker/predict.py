"""通过本地 OpenAI-compatible 推理服务生成药名预测和输入指纹。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from .legacy_data import iter_parquet_rows, legacy_prompt, legacy_row_id
from .prepare_data import _convert_record, read_jsonl


def _prediction_rows(data: Path, output_k: int) -> list[dict]:
    """读取 JSONL，或按 schema 读取新/旧版 VERL Parquet。"""

    if data.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        schema = set(pq.read_schema(data).names)
        if "extra_info" in schema:
            # 新协议 Parquet 已经包含 prompt/extra_info，不能误当旧协议按行号读取。
            required = {"prompt", "extra_info"}
            missing = required - schema
            if missing:
                raise ValueError(f"{data} 缺少新协议字段: {sorted(missing)}")
            rows = []
            table = pq.read_table(data, columns=["prompt", "extra_info"])
            for row_number, row in enumerate(table.to_pylist(), 1):
                info = row.get("extra_info")
                if not isinstance(info, dict):
                    raise ValueError(f"{data} 第 {row_number} 条缺少 extra_info")
                sample_id = info.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"{data} 第 {row_number} 条缺少 sample_id")
                rows.append(
                    {
                        "prompt": row["prompt"],
                        "sample_id": sample_id,
                        "extra_info": info,
                    }
                )
            if not rows:
                raise ValueError(f"{data} 不包含任何样本")
            return rows

        rows = []
        for row_number, raw in iter_parquet_rows(data):
            rows.append(
                {
                    "prompt": legacy_prompt(raw, f"{data} 第 {row_number} 条"),
                    "sample_id": legacy_row_id(row_number),
                    "row_index": row_number,
                }
            )
        return rows

    return [
        _convert_record(record, number, output_k, output_k)[0]
        for number, record in enumerate(read_jsonl(data), 1)
    ]


def predict(
    data: Path,
    output: Path,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    output_k: int = 20,
    max_tokens: int = 512,
    workers: int = 1,
) -> int:
    if data.resolve() == output.resolve():
        raise ValueError("预测输出不能覆盖输入数据")
    if workers < 1 or max_tokens < 1:
        raise ValueError("workers 和 max_tokens 必须大于0")
    rows = _prediction_rows(data, output_k)
    ids = [
        row.get("sample_id") or row["extra_info"]["sample_id"] for row in rows
    ]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("数据为空或 sample_id 重复")

    def infer(row: dict) -> dict:
        # Only the prompt is sent. GT, reward metadata, and references stay local.
        payload = {
            "model": model,
            "messages": row["prompt"],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        info = row.get("extra_info", {})
        item = {
            "sample_id": row.get("sample_id") or info["sample_id"],
            "model": model,
            "output": message.get("content") or "",
        }
        if "row_index" in row:
            item["row_index"] = row["row_index"]
        if "input_sha256" in info:
            item["input_sha256"] = info["input_sha256"]
        return item

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for item in pool.map(infer, rows):
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(
        f"已写入 {predict(args.data, args.output, args.model, args.base_url, args.output_k, args.max_tokens, args.workers)} 条预测"
    )


if __name__ == "__main__":
    main()
