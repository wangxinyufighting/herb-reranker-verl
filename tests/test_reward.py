"""Name protocol, legacy metrics, and progressive ordering tests."""

import math
import unittest

from herb_reranker.reward import (
    batch_test_ndcg,
    compute_score,
    ranking_metrics,
    stage_score,
)

C = [f"药{i}" for i in range(50)]
GT = {"candidate_herbs": C, "gt_herbs": [C[1], C[3], C[40], "候选外GT"]}
COPY = C[:20]
ORACLE = [C[1], C[3], C[40]] + [h for h in C if h not in GT["gt_herbs"]][:17]
POOR = [h for h in C if h not in GT["gt_herbs"]][:20]


def answer(ranking):
    return "<think>简洁说明</think><answer>" + ">".join(ranking) + "</answer>"


class RewardTest(unittest.TestCase):
    def score(self, ranking, stage="stage1", reference=None, gt=GT):
        info = {"candidate_herbs": C, "output_k": 20, "training_stage": stage}
        if reference is not None:
            info["reference_ranking"] = reference
        return compute_score("test", answer(ranking), gt, info)

    def test_old_and_verl_call_signatures(self):
        old = compute_score(answer(COPY), GT, data_source="test")
        new = compute_score(
            data_source="test", solution_str=answer(COPY), ground_truth=GT
        )
        self.assertEqual(old, new)
        self.assertEqual(compute_score(answer(COPY), GT, "test"), new)

    def test_copy_gain_and_ceiling(self):
        copied, good, bad = [self.score(r) for r in (COPY, ORACLE, POOR)]
        self.assertEqual(copied["rank_delta"], 0)
        self.assertEqual(copied["quality_reward"], 0)
        self.assertGreater(good["score"], copied["score"])
        self.assertGreater(copied["score"], bad["score"])
        self.assertEqual(good["ceiling_reached"], 1)
        self.assertEqual(good["quality_reward"], 1)
        ceiling_copy = self.score(COPY, gt={"gt_herbs": [C[0], C[1]]})
        self.assertEqual(ceiling_copy["quality_reward"], 1)

    def test_unreachable_is_not_a_perfect_ranking(self):
        result = self.score(COPY, gt={"gt_herbs": ["候选外GT"]})
        self.assertEqual(result["ceiling_reached"], 0)
        self.assertEqual(result["quality_reward"], 0)
        self.assertEqual(result["no_reachable_gt"], 1)

    def test_old_batch_test_ndcg_and_full_gt_recall(self):
        result = self.score(COPY)
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(result["model_ndcg_5"], expected)
        self.assertEqual(result["model_precision_5"], 2 / 5)
        self.assertEqual(result["model_recall_5"], 2 / 4)

    def test_old_ndcg_denominator_can_change_with_deeper_hits(self):
        self.assertEqual(batch_test_ndcg([1, 0, 0, 0, 0, 0], 5), 1)
        self.assertLess(batch_test_ndcg([1, 0, 0, 0, 0, 1], 5), 1)

    def test_no_prefix_repair_or_invalid_perfect_bonus(self):
        for ranking in (
            ORACLE[:3],
            ORACLE[:3] + ["非法药"] * 17,
            ORACLE[:19] + [ORACLE[0]],
        ):
            result = self.score(ranking)
            self.assertEqual(result["valid_output"], 0)
            self.assertEqual(result["ceiling_reached"], 0)
            self.assertLess(result["score"], self.score(COPY)["score"])
        partial = self.score([C[1]])
        self.assertEqual(partial["model_hits_20"], 1)

    def test_extra_valid_names_allowed_like_old_minimum_length(self):
        self.assertEqual(self.score(C)["valid_output"], 1)

    def test_ids_and_multiple_answer_blocks_rejected(self):
        for text in (
            '{"ranking":[1,2,3]}',
            answer(COPY) + answer(ORACLE),
            "<answer></answer>",
        ):
            result = compute_score(text, GT)
            self.assertEqual(result["valid_output"], 0)
            self.assertEqual(result["ceiling_reached"], 0)

    def test_stale_id_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "rebuilt"):
            compute_score("test", answer(COPY), GT, {"candidate_id_scheme": "old"})

    def test_stage2_requires_frozen_reference(self):
        with self.assertRaisesRegex(ValueError, "reference_ranking"):
            self.score(COPY, "stage2")
        with self.assertRaisesRegex(ValueError, "reference_ranking"):
            self.score(COPY, "stage2", ORACLE[:3])
        result = self.score(COPY, "stage2", ORACLE)
        self.assertEqual(result["reference_deficit_5"], 1)
        self.assertEqual(result["reference_hits_5"], 3)

    def test_stage_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "stage"):
            compute_score(
                "test",
                answer(COPY),
                GT,
                {"training_stage": "stage2"},
                training_stage="stage1",
            )

    def test_candidate_external_gt_cannot_be_credited(self):
        ranking = ["候选外GT"] + POOR[:19]
        self.assertEqual(self.score(ranking)["model_hits_20"], 0)

    def test_metrics_dedup_before_cutoff(self):
        metrics = ranking_metrics([C[1], C[1], C[3]], GT["gt_herbs"], C)
        self.assertEqual(metrics["hits_5"], 2)

    def test_no_nan(self):
        for stage in ("stage1", "stage2", "stage3"):
            result = self.score(ORACLE, stage, ORACLE)
            self.assertTrue(all(math.isfinite(value) for value in result.values()))


class LexicographicTest(unittest.TestCase):
    def metrics(self, h5, h10, h20, n):
        return {
            "hits_5": h5,
            "hits_10": h10,
            "hits_20": h20,
            "ndcg_5": n,
            "ndcg_10": n,
            "ndcg_20": n,
        }

    def test_one_hit_always_dominates_ndcg(self):
        ref = self.metrics(0, 0, 0, 0)
        for hits in range(5):
            better = self.metrics(hits + 1, 10, 20, 0)
            worse = self.metrics(hits, 10, 20, 1)
            self.assertGreater(
                stage_score(better, ref, 20, "stage1"),
                stage_score(worse, ref, 20, "stage1"),
            )

    def test_stage2_head_floor_dominates_all_top10_gain(self):
        ref = self.metrics(5, 5, 5, 0)
        maintained = self.metrics(5, 5, 5, 0)
        regression = self.metrics(4, 10, 20, 1)
        self.assertGreater(
            stage_score(maintained, ref, 20, "stage2"),
            stage_score(regression, ref, 20, "stage2"),
        )

    def test_stage2_top10_improves_after_floor_is_met(self):
        ref = self.metrics(3, 5, 5, 0)
        low = self.metrics(3, 5, 5, 1)
        high = self.metrics(3, 6, 6, 0)
        self.assertGreater(
            stage_score(high, ref, 20, "stage2"), stage_score(low, ref, 20, "stage2")
        )

    def test_stage3_both_head_floors_dominate_top20(self):
        ref = self.metrics(5, 10, 10, 0)
        good = self.metrics(5, 10, 10, 0)
        for bad in (self.metrics(4, 10, 20, 1), self.metrics(5, 9, 20, 1)):
            self.assertGreater(
                stage_score(good, ref, 20, "stage3"),
                stage_score(bad, ref, 20, "stage3"),
            )


if __name__ == "__main__":
    unittest.main()
