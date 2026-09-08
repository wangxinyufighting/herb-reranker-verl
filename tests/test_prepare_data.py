"""Data protocol and reference alignment tests."""

import unittest
from herb_reranker.prepare_data import _convert_record


def example_record():
    return {
        "sample_id": "case-1",
        "symptoms": ["头痛", "乏力"],
        "symptom_description": "头痛伴乏力。",
        "candidate_herbs": [f"候选药{i}" for i in range(50)],
        "ground_truth_herbs": ["候选药1", "不会进入提示词的真实药"],
    }


class PrepareDataTest(unittest.TestCase):
    def test_schema_no_gt_or_reference_leakage(self):
        record = example_record()
        row, reachable = _convert_record(record, 1, 50)
        text = "\n".join(message["content"] for message in row["prompt"])
        self.assertEqual(reachable, 1)
        self.assertIn("原始症状描述：头痛伴乏力。", text)
        self.assertIn("<answer>", text)
        self.assertIn("至少包含 20 味", text)
        self.assertNotIn("不会进入提示词的真实药", text)
        self.assertNotIn("治法", text)
        self.assertNotIn("候选ID", text)
        self.assertEqual(
            row["extra_info"]["candidate_herbs"], record["candidate_herbs"]
        )
        self.assertEqual(row["extra_info"]["output_protocol"], "herb_names_v1")

    def test_same_normalized_symptoms_keep_distinct_descriptions(self):
        a = example_record()
        b = dict(a, sample_id="case-2", symptom_description="持续头痛，午后乏力。")
        row_a, _ = _convert_record(a, 1, 50)
        row_b, _ = _convert_record(b, 2, 50)
        self.assertNotEqual(row_a["prompt"], row_b["prompt"])
        self.assertNotEqual(
            row_a["extra_info"]["input_sha256"], row_b["extra_info"]["input_sha256"]
        )

    def test_stage_reference_fingerprint_and_checkpoint(self):
        raw = example_record()
        row, _ = _convert_record(raw, 1, 50)
        ref = {
            "ranking": raw["candidate_herbs"][:20],
            "checkpoint_id": "stage1-selected",
            "input_sha256": row["extra_info"]["input_sha256"],
        }
        stage2, _ = _convert_record(raw, 1, 50, training_stage="stage2", reference=ref)
        self.assertEqual(stage2["prompt"], row["prompt"])
        self.assertEqual(stage2["extra_info"]["reference_ranking"], ref["ranking"])
        self.assertEqual(
            stage2["extra_info"]["reference_checkpoint"], "stage1-selected"
        )
        ref["input_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "指纹"):
            _convert_record(raw, 1, 50, training_stage="stage2", reference=ref)

    def test_reference_required(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            _convert_record(example_record(), 1, 50, training_stage="stage3")

    def test_duplicate_names_and_small_output_rejected(self):
        record = example_record()
        record["candidate_herbs"] = ["重复"] * 50
        with self.assertRaisesRegex(ValueError, "重复"):
            _convert_record(record, 1, 50)
        with self.assertRaisesRegex(ValueError, "output_k"):
            _convert_record(example_record(), 1, 50, output_k=5)

    def test_output_k_cannot_exceed_candidate_count(self):
        with self.assertRaisesRegex(ValueError, "少于 output_k"):
            _convert_record(example_record(), 1, 50, output_k=51)


if __name__ == "__main__":
    unittest.main()
