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
    relative_improvement_reward,
)


CANDIDATES = [f"药{i}" for i in range(1, 51)]
GROUND_TRUTH = {"ground_truth_herbs": ["药2", "药4", "药40", "候选外GT"]}
EXTRA_INFO = {
    "candidate_herbs": CANDIDATES,
    "gnn_candidate_herbs": CANDIDATES,
    "output_k": 20,
}
COPY_RANKING = list(range(1, 21))
ORACLE_RANKING = [2, 4, 40] + [
    index for index in range(1, 51) if index not in {2, 4, 40}
][:17]
POOR_RANKING = [
    index for index in range(1, 51) if index not in {2, 4, 40}
][:20]


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

    def test_relative_improvement_is_zero_centered(self) -> None:
        self.assertEqual(relative_improvement_reward(0.4, 0.4), 0.0)
        self.assertAlmostEqual(relative_improvement_reward(1.0, 0.4), 1.0)
        self.assertAlmostEqual(relative_improvement_reward(0.0, 0.4), -1.0)
        self.assertGreater(relative_improvement_reward(0.6, 0.4), 0.0)
        self.assertLess(relative_improvement_reward(0.2, 0.4), 0.0)


class RewardTest(unittest.TestCase):
    def score(self, solution: str, hierarchical: bool = True) -> dict[str, float]:
        return compute_score(
            data_source="ptm_herb_rerank",
            solution_str=solution,
            ground_truth=GROUND_TRUTH,
            extra_info=EXTRA_INFO,
            use_hierarchical_reward=hierarchical,
        )

    def test_copy_is_valid_but_has_zero_quality_improvement(self) -> None:
        result = self.score(output(COPY_RANKING))
        self.assertEqual(result["valid_output"], 1.0)
        self.assertEqual(result["exact_topk"], 1.0)
        self.assertEqual(result["exact_permutation"], 0.0)
        self.assertAlmostEqual(result["relative_improvement"], 0.0)
        self.assertAlmostEqual(result["score"], 0.05)
        for cutoff in (5, 10, 15, 20):
            self.assertEqual(result[f"copy_ratio_{cutoff}"], 1.0)
            self.assertEqual(result[f"exact_copy_{cutoff}"], 1.0)

    def test_better_copy_and_worse_rankings_are_ordered(self) -> None:
        better = self.score(output(ORACLE_RANKING))
        copied = self.score(output(COPY_RANKING))
        worse = self.score(output(POOR_RANKING))
        self.assertGreater(better["score"], copied["score"])
        self.assertGreater(copied["score"], worse["score"])
        self.assertGreater(better["relative_improvement"], 0.0)
        self.assertLess(worse["relative_improvement"], 0.0)
        self.assertGreater(better["anti_copy_bonus"], 0.0)
        self.assertEqual(worse["anti_copy_bonus"], 0.0)

    def test_gt_outside_candidates_does_not_change_reward_rank_score(self) -> None:
        with_unreachable = self.score(output(ORACLE_RANKING))
        without_unreachable = compute_score(
            "ptm_herb_rerank",
            output(ORACLE_RANKING),
            {"ground_truth_herbs": ["药2", "药4", "药40"]},
            EXTRA_INFO,
            use_hierarchical_reward=True,
        )
        self.assertAlmostEqual(
            with_unreachable["rank_score"], without_unreachable["rank_score"]
        )
        self.assertEqual(with_unreachable["reachable_gt_count"], 3.0)

    def test_reward_switch_changes_only_rank_aggregation(self) -> None:
        hierarchical = self.score(output(ORACLE_RANKING), hierarchical=True)
        fixed = self.score(output(ORACLE_RANKING), hierarchical=False)
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
        self.assertNotEqual(hierarchical["gnn_rank_score"], fixed["gnn_rank_score"])

    def test_invalid_json_is_below_every_valid_ranking(self) -> None:
        invalid = self.score("药1、药2、药3")
        worst_valid = self.score(output(POOR_RANKING))
        self.assertLess(invalid["score"], worst_valid["score"])
        self.assertEqual(invalid["format_score"], 0.0)
        self.assertEqual(invalid["valid_output"], 0.0)

    def test_missing_candidates_is_invalid(self) -> None:
        full = self.score(output(COPY_RANKING))
        partial = self.score(output(COPY_RANKING[:5]))
        self.assertLess(partial["score"], full["score"])
        self.assertAlmostEqual(partial["required_output_coverage"], 0.25)
        self.assertAlmostEqual(partial["candidate_coverage"], 0.10)
        self.assertEqual(partial["missing_index_count"], 15.0)
        self.assertEqual(partial["valid_output"], 0.0)

    def test_duplicate_and_out_of_candidate_are_invalid(self) -> None:
        malformed = COPY_RANKING[:-2] + [1, 51]
        result = self.score(output(malformed))
        self.assertEqual(result["valid_output"], 0.0)
        self.assertEqual(result["duplicate_index_count"], 1.0)
        self.assertEqual(result["invalid_index_count"], 1.0)
        self.assertEqual(result["missing_index_count"], 2.0)

    def test_more_than_output_k_is_invalid(self) -> None:
        result = self.score(output(list(range(1, 22))))
        self.assertEqual(result["valid_output"], 0.0)
        self.assertEqual(result["extra_index_count"], 1.0)

    def test_string_indices_are_rejected(self) -> None:
        result = self.score(output([str(index) for index in COPY_RANKING]))
        self.assertEqual(result["required_output_coverage"], 0.0)
        self.assertEqual(result["invalid_index_count"], 20.0)
        self.assertEqual(result["valid_output"], 0.0)

    def test_qwen_think_wrapper_can_be_parsed(self) -> None:
        wrapped = "<think>内部思考</think>\n```json\n" + output(COPY_RANKING) + "\n```"
        result = self.score(wrapped)
        self.assertEqual(result["valid_output"], 1.0)
        self.assertAlmostEqual(result["score"], 0.05)

    def test_separate_id_and_gnn_orders(self) -> None:
        # ID 1/2/3 分别指向药3/药1/药2；GNN baseline 仍是药1/药2/药3。
        result = compute_score(
            data_source="ptm_herb_rerank",
            solution_str=output([2, 3]),
            ground_truth={"ground_truth_herbs": ["药1", "药2"]},
            extra_info={
                "candidate_herbs": ["药3", "药1", "药2"],
                "gnn_candidate_herbs": ["药1", "药2", "药3"],
                "output_k": 2,
            },
        )
        self.assertEqual(result["copy_ratio_output"], 1.0)
        self.assertAlmostEqual(result["relative_improvement"], 0.0)

    def test_model_gnn_and_delta_metrics(self) -> None:
        result = self.score(output(COPY_RANKING))
        for cutoff in (5, 10, 15, 20):
            for metric in ("precision", "recall", "ndcg"):
                self.assertAlmostEqual(
                    result[f"delta_{metric}_{cutoff}"],
                    result[f"model_{metric}_{cutoff}"]
                    - result[f"gnn_{metric}_{cutoff}"],
                )
                self.assertAlmostEqual(
                    result[f"headroom_{metric}_{cutoff}"],
                    result[f"oracle_{metric}_{cutoff}"]
                    - result[f"gnn_{metric}_{cutoff}"],
                )

        self.assertAlmostEqual(result["model_precision_5"], 2 / 5)
        # Recall 使用完整 GT 作分母，候选集未召回的 GT 仍计入分母。
        self.assertAlmostEqual(result["model_recall_5"], 2 / 4)
        expected_ndcg = (
            1.0 / math.log2(3) + 1.0 / math.log2(5)
        ) / sum(1.0 / math.log2(rank + 1) for rank in range(1, 5))
        self.assertAlmostEqual(result["model_ndcg_5"], expected_ndcg)
        self.assertEqual(result["delta_ndcg_5"], 0.0)

    def test_invalid_output_does_not_inherit_gnn_metrics(self) -> None:
        result = self.score("非法输出")
        self.assertEqual(result["model_ndcg_5"], 0.0)
        self.assertGreater(result["gnn_ndcg_5"], 0.0)
        self.assertAlmostEqual(result["delta_ndcg_5"], -result["gnn_ndcg_5"])

    def test_oracle_exposes_retriever_headroom(self) -> None:
        result = self.score(output(ORACLE_RANKING))
        for cutoff in (5, 10, 15, 20):
            for metric in ("precision", "recall", "ndcg"):
                self.assertAlmostEqual(
                    result[f"model_{metric}_{cutoff}"],
                    result[f"oracle_{metric}_{cutoff}"],
                )
                self.assertAlmostEqual(
                    result[f"delta_{metric}_{cutoff}"],
                    result[f"headroom_{metric}_{cutoff}"],
                )
                self.assertAlmostEqual(result[f"remaining_gap_{metric}_{cutoff}"], 0.0)

        self.assertGreater(result["rank_delta"], 0.0)
        self.assertLess(result["copy_ratio_20"], 1.0)


if __name__ == "__main__":
    unittest.main()
