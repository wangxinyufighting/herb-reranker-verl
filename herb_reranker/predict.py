"""通过本地 OpenAI-compatible 推理服务生成药名预测和输入指纹。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from .prepare_data import _convert_record, read_jsonl


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
    rows = [
        _convert_record(record, number, output_k, output_k)[0]
        for number, record in enumerate(read_jsonl(data), 1)
    ]
    ids = [row["extra_info"]["sample_id"] for row in rows]
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
        info = row["extra_info"]
        message = result["choices"][0]["message"]
        return {
            "sample_id": info["sample_id"],
            "input_sha256": info["input_sha256"],
            "model": model,
            "output": message.get("content") or "",
        }

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
