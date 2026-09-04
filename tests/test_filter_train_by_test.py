"""测试测试分布感知的训练子集筛选逻辑。"""

from __future__ import annotations

import unittest

from herb_reranker.filter_train_by_test import select_train_indices


def row(sample_id: str, *symptoms: str, ground_truth: str = "任意GT") -> dict:
    """构造最小 VERL 行；GT 仅用于验证筛选器不会读取它。"""

    return {
        "extra_info": {"sample_id": sample_id, "symptoms": list(symptoms)},
        "reward_model": {"ground_truth": ground_truth},
    }


class FilterTrainByTestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            row("a-1", "症状甲"),
            row("a-2", "症状甲"),
            row("a-3", "症状甲"),
            row("b-1", "症状乙"),
            row("ab-1", "症状甲", "症状乙"),
            row("c-1", "症状丙"),
        ]

    def test_exact_keeps_only_complete_symptom_match(self) -> None:
        test = [row("test", "症状甲", ground_truth="秘密测试标签")]
        indices, stats = select_train_indices(self.train, test, mode="exact")
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(stats.selected_rows, 3)
        self.assertFalse(stats.used_test_labels)

    def test_matched_caps_rows_per_test_case(self) -> None:
        test = [row("test", "症状甲")]
        indices, stats = select_train_indices(
            self.train,
            test,
            mode="matched",
            train_per_test=2,
            seed=7,
        )
        self.assertEqual(len(indices), 2)
        self.assertTrue(all(index in {0, 1, 2} for index in indices))
        self.assertEqual(stats.exact_selected_rows, 2)

    def test_nearest_fallback_requires_shared_symptom(self) -> None:
        test = [
            row("partial", "症状乙", "训练中不存在的症状"),
            row("uncovered", "完全无关症状"),
        ]
        indices, stats = select_train_indices(
            self.train,
            test,
            mode="matched",
            train_per_test=1,
            seed=7,
        )
        self.assertEqual(len(indices), 1)
        self.assertIn(indices[0], {3, 4})
        self.assertEqual(stats.fallback_covered_test_rows, 1)
        self.assertEqual(stats.uncovered_test_rows, 1)

    def test_selection_is_reproducible(self) -> None:
        test = [row("test", "症状甲")]
        first, _ = select_train_indices(
            self.train, test, mode="matched", train_per_test=2, seed=42
        )
        second, _ = select_train_indices(
            self.train, test, mode="matched", train_per_test=2, seed=42
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
