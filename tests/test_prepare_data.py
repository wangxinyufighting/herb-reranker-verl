"""数据转换逻辑测试，不依赖 datasets 或 pyarrow。"""

from __future__ import annotations

import unittest

from herb_reranker.prepare_data import _convert_record


def example_record() -> dict[str, object]:
    return {
        "sample_id": "case-1",
        "symptoms": ["头痛", "乏力"],
        "symptom_description": "头痛伴乏力。",
        "candidate_herbs": [f"候选药{i}" for i in range(1, 21)],
        "ground_truth_herbs": ["候选药1", "不会进入提示词的真实药"],
    }


class PrepareDataTest(unittest.TestCase):
    def test_schema_and_no_ground_truth_leakage(self) -> None:
        row, reachable = _convert_record(example_record(), line_no=1, min_candidates=20)
        self.assertEqual(reachable, 1)
        self.assertEqual(row["data_source"], "ptm_herb_rerank")
        self.assertEqual(row["reward_model"]["ground_truth"]["ground_truth_herbs"][-1], "不会进入提示词的真实药")
        prompt_text = "\n".join(message["content"] for message in row["prompt"])
        self.assertNotIn("不会进入提示词的真实药", prompt_text)

    def test_duplicate_candidates_raise_error(self) -> None:
        record = example_record()
        record["candidate_herbs"] = ["重复药"] * 20
        with self.assertRaisesRegex(ValueError, "含重复中药"):
            _convert_record(record, line_no=1, min_candidates=20)


if __name__ == "__main__":
    unittest.main()
