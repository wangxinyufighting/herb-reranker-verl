"""Name-based reward derived from the original rerank_reward.

Use metric.py's batch_test NDCG (method=1, k_max=20). Stage objectives are
lexicographic: Top-5, then Top-10 under a Top-5 floor, then Top-20 under both
floors. The floors come from frozen previous-checkpoint predictions, not GT.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

CUTOFFS = (5, 10, 20)
STAGES = ("stage1", "stage2", "stage3")
PROTOCOL = "herb_names_v1"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def normalize_stage(value: Any) -> str:
    stage = str(value).strip().lower()
    stage = {"1": "stage1", "2": "stage2", "3": "stage3"}.get(stage, stage)
    if stage not in STAGES:
        raise ValueError(f"training_stage must be one of {STAGES}, got {value!r}")
    return stage


def names(value: Any, label: str, allow_empty: bool = False) -> list[str]:
    # Parquet/VERL may supply numpy arrays, so do not require list specifically.
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a list of herb names")
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a list of herb names") from exc
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{label} contains an empty or non-string name")
    result = [item.strip() for item in result]
    if not result and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return result


def dedup_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def dcg_at_k(relevance: list[float], k: int) -> float:
    return sum(
        value / math.log2(index + 2) for index, value in enumerate(relevance[:k])
    )


def batch_test_ndcg(relevance: list[float], k: int, k_max: int = 20) -> float:
    relevance = relevance[:k_max]
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return dcg_at_k(relevance, k) / ideal if ideal else 0.0


def ranking_metrics(
    ranking: list[str], gt: list[str], candidates: list[str]
) -> dict[str, float]:
    """Dedup before truncation; full GT denominator for recall, as in old reward."""
    gt_set = set(gt)
    reachable = gt_set & set(candidates)
    relevance = [
        float(name in reachable) for name in dedup_preserve_order(ranking)[:20]
    ]
    result: dict[str, float] = {}
    for k in CUTOFFS:
        hits = sum(relevance[:k])
        precision = hits / k
        recall = hits / len(gt_set) if gt_set else 0.0
        result.update(
            {
                f"hits_{k}": hits,
                f"precision_{k}": precision,
                f"recall_{k}": recall,
                f"f1_{k}": 2 * precision * recall / (precision + recall)
                if hits
                else 0.0,
                f"ndcg_{k}": batch_test_ndcg(relevance, k),
            }
        )
    return result


def stage_score(
    metrics: Mapping[str, float],
    reference: Mapping[str, float],
    reachable: int,
    stage: str,
) -> float:
    """Mixed-radix scalarization: a one-hit priority change dominates the tail.

    Starting NDCG in [0, 1/2] makes every lower-priority tail strictly < 1.
    Radices are derived from attainable hit counts, not fitted metric weights.
    """
    if not reachable:
        return 0.0
    target = {"stage1": 5, "stage2": 10, "stage3": 20}[stage]
    components = []
    for k in CUTOFFS:
        if k >= target:
            break
        limit = min(k, reachable)
        deficit = max(0.0, reference[f"hits_{k}"] - metrics[f"hits_{k}"])
        components.append((limit - deficit, limit))
    components.append((metrics[f"hits_{target}"], min(target, reachable)))
    encoded = metrics[f"ndcg_{target}"] / 2.0
    for value, maximum in reversed(components):
        encoded = (value + encoded) / (maximum + 1.0)
    return encoded


def parse_answer(solution: str) -> tuple[list[str], bool, int]:
    """Do not repair, map IDs, accept aliases, or fill the tail using GNN/GT."""
    matches = list(ANSWER_RE.finditer(solution))
    think = THINK_RE.search(solution)
    if len(matches) != 1:
        return [], False, len(think.group(1).strip()) if think else 0
    body = matches[0].group(1).strip()
    ranking = [part.strip() for part in body.split(">")]
    clean = bool(body) and all(ranking) and not any("<" in name for name in ranking)
    return ranking if body else [], clean, len(think.group(1).strip()) if think else 0


def validate_reference(
    reference: list[str], candidates: list[str], output_k: int
) -> None:
    if (
        len(reference) < output_k
        or len(set(reference)) != len(reference)
        or not set(reference) <= set(candidates)
    ):
        raise ValueError(
            "reference_ranking must contain at least output_k unique candidate names"
        )


def compute_score(
    *args: Any,
    data_source: str = "",
    solution_str: Any = None,
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Accept VERL and original reward call conventions, including keyword calls."""
    if args:
        if len(args) in (2, 3) and isinstance(args[1], Mapping):
            solution_str, ground_truth = args[:2]
            if len(args) == 3:
                data_source = args[2]
        elif len(args) in (3, 4):
            data_source, solution_str, ground_truth = args[:3]
            if len(args) == 4:
                extra_info = args[3]
        else:
            raise TypeError(
                "Expected (solution, gt) or (data_source, solution, gt, extra_info)"
            )
    if not isinstance(ground_truth, Mapping):
        raise ValueError("ground_truth must contain gt_herbs or ground_truth_herbs")
    info = dict(extra_info or {})
    if (
        info.get("candidate_id_scheme")
        or info.get("output_protocol", PROTOCOL) != PROTOCOL
    ):
        raise ValueError("Old ID-protocol data must be rebuilt for the name reward")
    candidates = names(
        info.get("candidate_herbs", ground_truth.get("candidate_herbs")),
        "candidate_herbs",
    )
    gnn = names(info.get("gnn_candidate_herbs", candidates), "gnn_candidate_herbs")
    if (
        len(set(candidates)) != len(candidates)
        or len(set(gnn)) != len(gnn)
        or set(gnn) != set(candidates)
    ):
        raise ValueError(
            "candidate_herbs and gnn_candidate_herbs must be unique and have the same names"
        )
    gt = names(
        ground_truth.get("ground_truth_herbs", ground_truth.get("gt_herbs")),
        "ground_truth_herbs",
        allow_empty=True,
    )
    stage = normalize_stage(
        kwargs.get("training_stage", info.get("training_stage", "stage1"))
    )
    if info.get("training_stage") and normalize_stage(info["training_stage"]) != stage:
        raise ValueError(
            "reward stage disagrees with Parquet training_stage; rebuild stage data"
        )
    output_k = int(kwargs.get("output_k", info.get("output_k", 20)))
    if info.get("output_k", output_k) != output_k:
        raise ValueError("reward output_k disagrees with Parquet")
    if output_k < 20 or output_k > len(candidates):
        raise ValueError(
            "output_k must be between 20 and candidate count (metric k_max=20)"
        )
    if stage == "stage1":
        reference = gnn[:output_k]
    else:
        # Silent fallback to GNN would stop protecting the preceding checkpoint.
        reference = names(info.get("reference_ranking"), "reference_ranking")
        validate_reference(reference, candidates, output_k)
    text = str(solution_str or "")
    ranking, parse_ok, think_chars = parse_answer(text)
    unique = dedup_preserve_order(ranking)
    invalid = sum(name not in set(candidates) for name in ranking)
    duplicates = len(ranking) - len(unique)
    shortfall = max(0, output_k - len(set(unique) & set(candidates)))
    valid = parse_ok and not invalid and not duplicates and not shortfall

    reachable_set = set(gt) & set(candidates)
    ideal = [name for name in gnn if name in reachable_set] + [
        name for name in gnn if name not in reachable_set
    ]
    metrics = {
        "model": ranking_metrics(ranking, gt, candidates),
        "gnn": ranking_metrics(gnn, gt, candidates),
        "reference": ranking_metrics(reference, gt, candidates),
        "oracle": ranking_metrics(ideal, gt, candidates),
    }
    scores = {
        key: stage_score(value, metrics["reference"], len(reachable_set), stage)
        for key, value in metrics.items()
    }
    delta = scores["model"] - scores["gnn"]
    ceiling = bool(
        valid
        and reachable_set
        and math.isclose(scores["model"], scores["oracle"], abs_tol=1e-12, rel_tol=0.0)
    )
    quality = 1.0 if ceiling else delta
    # Invalid/short answers retain the observed ranking diagnostics, but cannot
    # win a perfect-ranking bonus or gain from dropping most required names.
    if not valid:
        quality = min(0.0, quality)
    format_score = float(valid)
    penalty = min(1.0, 0.03 * (invalid + duplicates + shortfall))
    if not parse_ok:
        penalty = max(penalty, 0.1)
    format_weight = float(kwargs.get("format_weight", 0.1))
    if not 0 <= format_weight <= 0.2:
        raise ValueError("format_weight must be in [0, 0.2]")
    # Unlike the old negative-only scaling, this monotone transform cannot
    # reverse lexicographic preferences between two complete legal answers.
    reward = quality + format_weight * format_score - penalty
    result = {
        "score": reward,
        "quality_reward": quality,
        "rank_score": scores["model"],
        "gnn_rank_score": scores["gnn"],
        "reference_rank_score": scores["reference"],
        "oracle_rank_score": scores["oracle"],
        "rank_delta": delta,
        "format_score": format_score,
        "format_penalty": penalty,
        "valid_output": float(valid),
        "ceiling_reached": float(ceiling),
        "gnn_at_ceiling": float(
            bool(reachable_set)
            and math.isclose(
                scores["gnn"], scores["oracle"], abs_tol=1e-12, rel_tol=0.0
            )
        ),
        "reachable_gt_count": float(len(reachable_set)),
        "no_reachable_gt": float(not reachable_set),
        "duplicate_name_count": float(duplicates),
        "invalid_name_count": float(invalid),
        "shortfall_count": float(shortfall),
        "output_count": float(len(unique)),
        "think_chars": float(think_chars),
        "think_over_budget": float(think_chars > 200),
        "training_stage": float(STAGES.index(stage) + 1),
    }
    for k in CUTOFFS:
        for key in ("hits", "precision", "recall", "f1", "ndcg"):
            for prefix, values in metrics.items():
                result[f"{prefix}_{key}_{k}"] = values[f"{key}_{k}"]
            result[f"{key}_{k}"] = metrics["model"][f"{key}_{k}"]
            result[f"delta_{key}_{k}"] = (
                metrics["model"][f"{key}_{k}"] - metrics["gnn"][f"{key}_{k}"]
            )
        result[f"reference_deficit_{k}"] = max(
            0.0, metrics["reference"][f"hits_{k}"] - metrics["model"][f"hits_{k}"]
        )
    return result
