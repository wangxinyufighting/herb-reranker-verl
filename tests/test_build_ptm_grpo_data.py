"""PTM 上下文与 GNN 候选对齐测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from herb_reranker.build_ptm_grpo_data import merge_to_jsonl


class BuildPtmGrpoDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mapping = self.root / "herb_mapping.txt"
        self.context = self.root / "test_with_context.jsonl"
        self.retrieval = self.root / "test_top_herbs.txt"
        self.output = self.root / "test.jsonl"

        self.mapping.write_text("药甲 0\n药乙 1\n药丙 2\n药丁 3\n", encoding="utf-8")
        row = {
            "sample_id": "test-000000",
            "symptoms": ["症状甲", "症状乙"],
            "symptom_description": "原始症状描述。",
            "ground_truth_herbs": ["药乙", "候选外药"],
            "symptom_ids": [7, 8],
        }
        self.context.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self.retrieval.write_text("7 8\t0 1 2 3\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_merge_and_top_k(self) -> None:
        stats = merge_to_jsonl(
            self.context,
            self.retrieval,
            self.mapping,
            self.output,
            candidate_k=3,
        )
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["candidate_herbs"], ["药甲", "药乙", "药丙"])
        self.assertEqual(result["symptoms"], ["症状甲", "症状乙"])
        self.assertEqual(result["symptom_description"], "原始症状描述。")
        self.assertEqual(result["ground_truth_herbs"], ["药乙", "候选外药"])
        self.assertEqual(stats.input_rows, 1)
        self.assertEqual(stats.written_rows, 1)
        self.assertAlmostEqual(stats.micro_candidate_recall, 0.5)

    def test_symptom_id_mismatch_raises(self) -> None:
        self.retrieval.write_text("7 9\t0 1 2 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "症状 ID 对齐失败"):
            merge_to_jsonl(
                self.context,
                self.retrieval,
                self.mapping,
                self.output,
                candidate_k=3,
            )

    def test_unknown_herb_id_raises(self) -> None:
        self.retrieval.write_text("7 8\t0 1 99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "未映射的中药 ID"):
            merge_to_jsonl(
                self.context,
                self.retrieval,
                self.mapping,
                self.output,
                candidate_k=3,
            )

    def test_unreachable_sample_is_dropped(self) -> None:
        row = json.loads(self.context.read_text(encoding="utf-8"))
        row["ground_truth_herbs"] = ["候选外药"]
        self.context.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "没有可写入样本"):
            merge_to_jsonl(
                self.context,
                self.retrieval,
                self.mapping,
                self.output,
                candidate_k=3,
                unreachable_policy="drop",
            )


if __name__ == "__main__":
    unittest.main()

