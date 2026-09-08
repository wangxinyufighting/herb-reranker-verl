"""Integration tests for staged data and frozen model predictions."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow.parquet as pq
from herb_reranker.prepare_data import _convert_record, convert_jsonl, read_jsonl
from herb_reranker.evaluate import evaluate
from herb_reranker.predict import predict
from herb_reranker.reward import compute_score
from herb_reranker.validate_parquet import validate_parquet
from herb_reranker.build_ptm_grpo_data import merge_names_to_jsonl
from herb_reranker.split_data import split_records


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.pred, self.ref, self.parquet = [
            self.root / name
            for name in ("data.jsonl", "pred.jsonl", "ref.jsonl", "data.parquet")
        ]
        self.herbs = [f"h{i}" for i in range(50)]
        self.rows = [
            {
                "sample_id": str(i),
                "symptoms": ["same"],
                "symptom_description": f"original text {i}",
                "candidate_herbs": self.herbs,
                "ground_truth_herbs": gt,
            }
            for i, gt in enumerate((["h1", "unreachable"], ["unreachable"]))
        ]
        self.write(self.data, self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def predictions(self):
        return [
            {
                "sample_id": row["sample_id"],
                "ranking": self.herbs[:20],
                "input_sha256": _convert_record(row, 1, 50)[0]["extra_info"][
                    "input_sha256"
                ],
            }
            for row in self.rows
        ]

    def test_validation_split_is_stable_and_grouped(self):
        rows = self.rows + [dict(self.rows[0], sample_id="duplicate-input")]
        a, b = split_records(rows, 0.5)
        self.assertEqual((a, b), split_records(rows, 0.5))
        self.assertFalse(
            {row["symptom_description"] for row in a}
            & {row["symptom_description"] for row in b}
        )
        self.assertEqual(len(a) + len(b), 3)

    def test_train_drop_test_keep(self):
        convert_jsonl(self.data, self.parquet)
        self.assertEqual(validate_parquet(self.parquet), 1)
        convert_jsonl(self.data, self.parquet, unreachable_policy="keep")
        self.assertEqual(validate_parquet(self.parquet), 2)

    def test_metrics_and_all_stages_roundtrip(self):
        self.write(self.pred, self.predictions())
        metrics = evaluate(
            self.data, self.pred, reference_output=self.ref, checkpoint_id="frozen"
        )
        self.assertEqual(metrics["model_recall_5"], 0.25)
        self.assertEqual(metrics["no_reachable_gt"], 0.5)
        for stage in ("stage2", "stage3"):
            convert_jsonl(
                self.data,
                self.parquet,
                unreachable_policy="keep",
                training_stage=stage,
                reference_jsonl=self.ref,
            )
            self.assertEqual(validate_parquet(self.parquet, training_stage=stage), 2)
            for row in pq.read_table(self.parquet).to_pylist():
                result = compute_score(
                    "test",
                    "<answer>" + ">".join(self.herbs[:20]) + "</answer>",
                    row["reward_model"]["ground_truth"],
                    row["extra_info"],
                )
                self.assertEqual(result["valid_output"], 1)

    def test_stale_missing_and_invalid_predictions_fail(self):
        for modification in ("fingerprint", "short", "missing"):
            items = self.predictions()
            if modification == "fingerprint":
                items[0]["input_sha256"] = "stale"
            elif modification == "short":
                items[0]["ranking"] = self.herbs[:2]
            else:
                items.pop()
            self.write(self.pred, items)
            with self.assertRaises(ValueError):
                evaluate(
                    self.data,
                    self.pred,
                    reference_output=self.ref,
                    checkpoint_id="frozen",
                )
            self.assertFalse(self.ref.exists())

    def test_name_join_preserves_cases_and_rejects_conflicting_candidates(self):
        source = self.root / "names.txt"
        source.write_text(
            "same\t" + " ".join(self.herbs) + "\n"
            "same\t" + " ".join(self.herbs) + "\n"
        )
        output = self.root / "merged.jsonl"
        stats = merge_names_to_jsonl(
            self.data, source, output, unreachable_policy="keep"
        )
        self.assertEqual(stats.written_rows, 2)
        self.assertNotEqual(
            read_jsonl(output)[0]["symptom_description"],
            read_jsonl(output)[1]["symptom_description"],
        )
        source.write_text(
            "same\t" + " ".join(self.herbs) + "\n"
            "same\t" + " ".join(reversed(self.herbs)) + "\n"
        )
        merge_names_to_jsonl(self.data, source, output, unreachable_policy="keep")
        self.assertNotEqual(
            read_jsonl(output)[0]["candidate_herbs"],
            read_jsonl(output)[1]["candidate_herbs"],
        )
        source.write_text(source.read_text() + "same\t" + " ".join(self.herbs) + "\n")
        with self.assertRaisesRegex(ValueError, "冲突"):
            merge_names_to_jsonl(self.data, source, output)

    def test_filtered_candidate_subsequence_preserves_duplicate_cases(self):
        source = self.root / "names.txt"
        first = "same\t" + " ".join(self.herbs)
        second = "same\t" + " ".join(reversed(self.herbs))
        source.write_text(first + "\n" + second + "\n")
        missing = dict(self.rows[0], sample_id="missing", symptoms=["missing"])
        last = dict(self.rows[1], ground_truth_herbs=["h1"])
        self.write(self.data, [self.rows[0], missing, last])
        output = self.root / "merged.jsonl"
        stats = merge_names_to_jsonl(self.data, source, output)
        self.assertEqual(stats.input_rows, 3)
        self.assertEqual(stats.written_rows, 2)
        self.assertEqual(stats.dropped_missing_candidate_rows, 1)
        self.assertEqual(stats.alignment_mode, "symptom_names_subsequence")
        self.assertEqual([row["sample_id"] for row in read_jsonl(output)], ["0", "1"])
        self.assertEqual(
            read_jsonl(output)[1]["candidate_herbs"], list(reversed(self.herbs))
        )
        with self.assertRaisesRegex(ValueError, "缺少"):
            merge_names_to_jsonl(self.data, source, output, unreachable_policy="keep")

    def test_partial_missing_duplicate_case_is_not_guessed(self):
        source = self.root / "names.txt"
        source.write_text(
            "same\t"
            + " ".join(self.herbs)
            + "\n"
            + "same\t"
            + " ".join(reversed(self.herbs))
            + "\n"
        )
        self.write(self.data, self.rows + [dict(self.rows[0], sample_id="third")])
        with self.assertRaisesRegex(ValueError, "冲突"):
            merge_names_to_jsonl(self.data, source, self.root / "merged.jsonl")

    def test_predict_sends_no_gt_and_exports_fingerprint(self):
        sent = []

        def respond(request, timeout):
            sent.append(json.loads(request.data))
            output = "<answer>" + ">".join(self.herbs[:20]) + "</answer>"
            return io.BytesIO(
                json.dumps({"choices": [{"message": {"content": output}}]}).encode()
            )

        with patch("herb_reranker.predict.urlopen", side_effect=respond):
            self.assertEqual(predict(self.data, self.pred, "frozen"), 2)
        self.assertNotIn("unreachable", json.dumps(sent))
        self.assertNotIn("ground_truth", json.dumps(sent))
        evaluate(
            self.data, self.pred, reference_output=self.ref, checkpoint_id="frozen"
        )
        self.assertEqual(len(read_jsonl(self.ref)), 2)


if __name__ == "__main__":
    unittest.main()
