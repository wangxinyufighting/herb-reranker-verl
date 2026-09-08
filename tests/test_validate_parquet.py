"""Reject stale protocol and stage metadata before launching GPU training."""

import copy
import unittest
from pathlib import Path
from herb_reranker.prepare_data import _convert_record
from herb_reranker.validate_parquet import validate_protocol_row


class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.row, _ = _convert_record(
            {
                "sample_id": "test",
                "symptoms": ["symptom"],
                "symptom_description": "original description",
                "candidate_herbs": [f"h{i}" for i in range(50)],
                "ground_truth_herbs": ["h2"],
            },
            1,
            50,
        )

    def test_current_protocol_passes(self):
        validate_protocol_row(self.row, Path("test"), 1, 20, "stage1")

    def test_old_protocol_and_wrong_stage_rejected(self):
        row = copy.deepcopy(self.row)
        row["extra_info"]["output_protocol"] = "ids"
        with self.assertRaisesRegex(ValueError, "药名协议"):
            validate_protocol_row(row, Path("test"), 1)
        with self.assertRaisesRegex(ValueError, "training_stage"):
            validate_protocol_row(self.row, Path("test"), 1, training_stage="stage2")

    def test_prompt_or_fingerprint_drift_rejected(self):
        for field in ("symptom_description", "input_sha256"):
            row = copy.deepcopy(self.row)
            row["extra_info"][field] = "changed"
            with self.assertRaisesRegex(ValueError, "指纹"):
                validate_protocol_row(row, Path("test"), 1)

    def test_wrong_output_k_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_k"):
            validate_protocol_row(self.row, Path("test"), 1, 50)


if __name__ == "__main__":
    unittest.main()
