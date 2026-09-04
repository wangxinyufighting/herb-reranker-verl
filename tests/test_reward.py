"""奖励函数的关键边界测试。"""

from __future__ import annotations

import json
import math
import unittest

from herb_reranker.reward import (
    competence_gate,
    compute_score,
    fixed_joint_rank_reward,
    hierarchical_rank_reward,
)


CANDIDATES = [f"药{i}" for i in range(1, 21)]
GROUND_TRUTH = {"ground_truth_herbs": ["药1", "药2", "药3", "候选外GT"]}
EXTRA_INFO = {"candidate_herbs": CANDIDATES}
INDEX_RANKING = list(range(1, len(CANDIDATES) + 1))


def output(ranking: list[object]) -> str:
    return json.dumps({"ranking": ranking}, ensure_ascii=False)


class HierarchicalRewardTest(unittest.TestCase):
    def test_formula(self) -> None:
        gate_5 = 0.1 + 0.9 * 0.2
        gate_10 = 0.1 + 0.9 * 0.4
        expected = (0.2 + gate_5 * 0.4 + gate_5 * gate_10 * 0.6) / 3.0
        self.assertAlmostEqual(hierarchical_rank_reward(0.2, 0.4, 0.6), expected)

    def test_perfect_ranking_reaches_one(self) -> None:
        self.assertAlmostEqual(hierarchical_rank_reward(1.0, 1.0, 1.0), 1.0)

    def test_epsilon_avoids_zero_reward_dead_zone(self) -> None:
        self.assertEqual(competence_gate(0.0, epsilon=0.1), 0.1)
        self.assertGreater(hierarchical_rank_reward(0.0, 0.5, 0.6), 0.0)

    def test_fixed_joint_reward(self) -> None:
        expected = 0.4 * 0.2 + 0.3 * 0.4 + 0.3 * 0.6
        self.assertAlmostEqual(fixed_joint_rank_reward(0.2, 0.4, 0.6), expected)


class RewardTest(unittest.TestCase):
    def score(self, solution: str, hierarchical: bool = True) -> dict[str, float]:
        return compute_score(
            data_source="ptm_herb_rerank",
            solution_str=solution,
            ground_truth=GROUND_TRUTH,
            extra_info=EXTRA_INFO,
            use_hierarchical_reward=hierarchical,
        )

    def test_perfect_full_permutation_reaches_one(self) -> None:
        result = self.score(output(INDEX_RANKING))
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(result["exact_permutation"], 1.0)

    def test_poor_head_ranking_scores_lower(self) -> None:
        poor = list(range(4, 21)) + [1, 2, 3]
        self.assertLess(
            self.score(output(poor))["score"],
            self.score(output(INDEX_RANKING))["score"],
        )

    def test_gt_outside_candidates_does_not_change_denominator(self) -> None:
        with_unreachable = self.score(output(INDEX_RANKING))
        without_unreachable = compute_score(
            "ptm_herb_rerank",
            output(INDEX_RANKING),
            {"ground_truth_herbs": ["药1", "药2", "药3"]},
            EXTRA_INFO,
            use_hierarchical_reward=True,
        )
        self.assertAlmostEqual(with_unreachable["rank_score"], without_unreachable["rank_score"])
        self.assertEqual(with_unreachable["reachable_gt_count"], 3.0)

    def test_reward_switch_changes_only_rank_aggregation(self) -> None:
        poor = list(range(4, 21)) + [1, 2, 3]
        hierarchical = self.score(output(poor), hierarchical=True)
        fixed = self.score(output(poor), hierarchical=False)
        self.assertEqual(hierarchical["hierarchical_reward_enabled"], 1.0)
        self.assertEqual(fixed["hierarchical_reward_enabled"], 0.0)
        for key in (
            "ndcg_5",
            "ndcg_10",
            "ndcg_15",
            "ndcg_20",
            "format_score",
            "constraint_quality",
        ):
            self.assertEqual(hierarchical[key], fixed[key])
        self.assertNotEqual(hierarchical["rank_score"], fixed["rank_score"])

    def test_invalid_json_gets_zero(self) -> None:
        result = self.score("药1、药2、药3")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["format_score"], 0.0)

    def test_missing_candidates_is_penalized(self) -> None:
        full = self.score(output(INDEX_RANKING))
        partial = self.score(output(INDEX_RANKING[:5]))
        self.assertLess(partial["score"], full["score"])
        self.assertAlmostEqual(partial["candidate_coverage"], 0.25)
        self.assertEqual(partial["exact_permutation"], 0.0)

    def test_duplicate_and_out_of_candidate_are_penalized(self) -> None:
        malformed = INDEX_RANKING[:-1] + [1, 21]
        result = self.score(output(malformed))
        self.assertLess(result["candidate_precision"], 1.0)
        self.assertLess(result["candidate_coverage"], 1.0)
        self.assertEqual(result["exact_permutation"], 0.0)

    def test_non_integer_item_cannot_be_filtered_away(self) -> None:
        malformed = json.dumps({"ranking": INDEX_RANKING + [None]}, ensure_ascii=False)
        result = self.score(malformed)
        self.assertLess(result["candidate_precision"], 1.0)
        self.assertEqual(result["exact_permutation"], 0.0)

    def test_string_indices_are_rejected(self) -> None:
        result = self.score(output([str(index) for index in INDEX_RANKING]))
        self.assertEqual(result["candidate_coverage"], 0.0)
        self.assertAlmostEqual(result["score"], 0.02)

    def test_qwen_think_wrapper_can_be_parsed(self) -> None:
        wrapped = "<think>内部思考</think>\n```json\n" + output(INDEX_RANKING) + "\n```"
        result = self.score(wrapped)
        self.assertAlmostEqual(result["score"], 1.0)

    def test_model_gnn_and_delta_metrics(self) -> None:
        result = self.score(output(INDEX_RANKING))
        for cutoff in (5, 10, 15, 20):
            for metric in ("precision", "recall", "ndcg"):
                self.assertAlmostEqual(
                    result[f"delta_{metric}_{cutoff}"],
                    result[f"model_{metric}_{cutoff}"]
                    - result[f"gnn_{metric}_{cutoff}"],
                )

        self.assertAlmostEqual(result["model_precision_5"], 3 / 5)
        # Recall 使用完整 GT 作分母，候选集未召回的 GT 仍计入分母。
        self.assertAlmostEqual(result["model_recall_5"], 3 / 4)
        # 测试 NDCG 使用完整 GT；reward 的候选条件化 NDCG 仍为 1。
        expected_ndcg = (
            1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        ) / sum(1.0 / math.log2(rank + 1) for rank in range(1, 5))
        self.assertAlmostEqual(result["model_ndcg_5"], expected_ndcg)
        self.assertAlmostEqual(result["ndcg_5"], 1.0)
        self.assertAlmostEqual(result["delta_ndcg_5"], 0.0)

    def test_invalid_output_does_not_inherit_gnn_metrics(self) -> None:
        result = self.score("非法输出")
        self.assertEqual(result["model_ndcg_5"], 0.0)
        self.assertGreater(result["gnn_ndcg_5"], 0.0)
        self.assertAlmostEqual(
            result["delta_ndcg_5"], -result["gnn_ndcg_5"]
        )


if __name__ == "__main__":
    unittest.main()
