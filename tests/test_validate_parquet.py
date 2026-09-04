"""训练数据协议预检的纯逻辑测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from herb_reranker.prepare_data import _convert_record
from herb_reranker.validate_parquet import validate_protocol_row


class ValidateProtocolTest(unittest.TestCase):
    def new_row(self) -> dict[str, object]:
        raw = {
            "sample_id": "case-1",
            "symptoms": ["头痛"],
            "symptom_description": "头痛反复。",
            "candidate_herbs": [f"药{i}" for i in range(1, 51)],
            "ground_truth_herbs": ["药2"],
        }
        row, _ = _convert_record(raw, line_no=1, min_candidates=50, output_k=20)
        return row

    def test_new_protocol_passes(self) -> None:
        validate_protocol_row(self.new_row(), Path("train.parquet"), 1, 20)

    def test_old_protocol_is_rejected(self) -> None:
        row = self.new_row()
        del row["extra_info"]["candidate_id_scheme"]
        with self.assertRaisesRegex(ValueError, "重新运行"):
            validate_protocol_row(row, Path("train.parquet"), 1, 20)

    def test_output_k_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "训练配置要求"):
            validate_protocol_row(self.new_row(), Path("train.parquet"), 1, 10)


if __name__ == "__main__":
    unittest.main()
