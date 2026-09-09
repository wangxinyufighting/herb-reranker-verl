"""Tests for the adapter around the historical reward and Parquet contract."""

import unittest
from pathlib import Path
from unittest.mock import patch

from herb_reranker.legacy_data import (
    legacy_ground_truth,
    legacy_prompt,
    legacy_row_id,
)
from herb_reranker.predict import _prediction_rows


def legacy_row():
    candidates = [f"h{i}" for i in range(50)]
    return {
        "prompt": [{"role": "user", "content": "患者症状：阴疮"}],
        "reward_model": {
            "ground_truth": {
                "candidate_herbs": candidates,
                "gt_herbs": ["h3", "h8"],
                "treatments": ["温经散寒"],
            }
        },
        "symptoms": "阴疮",
        "data_source": "legacy",
    }


class LegacyCompatTest(unittest.TestCase):
    def test_prompt_and_ground_truth_are_preserved(self):
        row = legacy_row()
        self.assertEqual(legacy_prompt(row)[0]["content"], "患者症状：阴疮")
        ground_truth = legacy_ground_truth(row)
        self.assertEqual(ground_truth["candidate_herbs"][3], "h3")
        self.assertEqual(ground_truth["treatments"], ["温经散寒"])

    def test_row_id_is_stable(self):
        self.assertEqual(legacy_row_id(1), "row-1")
        self.assertEqual(legacy_row_id(8661), "row-8661")

    def test_prediction_rows_use_legacy_prompt_without_gt(self):
        with patch(
            "herb_reranker.predict.iter_parquet_rows",
            return_value=iter([(1, legacy_row())]),
        ), patch(
            "pyarrow.parquet.read_schema",
            return_value=type("Schema", (), {"names": ["prompt", "reward_model"]})(),
        ):
            rows = _prediction_rows(Path("fixture.parquet"), 20)
        self.assertEqual(rows[0]["sample_id"], "row-1")
        self.assertEqual(rows[0]["row_index"], 1)
        self.assertEqual(rows[0]["prompt"][0]["role"], "user")
        self.assertNotIn("reward_model", rows[0])

    def test_historical_reward_accepts_name_answer(self):
        import reward as historical_reward

        result = historical_reward.compute_score(
            "<think>按症状排序</think><answer>h3>h8>h0>h1>h2>h4>h5>h6>h7>h9>h10>h11>h12>h13>h14>h15>h16>h17>h18>h19</answer>",
            legacy_ground_truth(legacy_row()),
            "legacy",
            {},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)


if __name__ == "__main__":
    unittest.main()
