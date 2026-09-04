"""VERL 自定义奖励函数。

本文件刻意不导入 VERL：VERL 会按文件路径动态加载 ``compute_score``，而纯标准库
实现便于在普通 Python 环境中单独测试。奖励只评价候选列表内部的相对排序，不把
召回器未提供的真实中药归咎于 reranker。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把浮点数截断到闭区间。"""

    return max(lower, min(upper, value))


def _as_bool(value: Any) -> bool:
    """安全解析 Hydra/环境变量传入的布尔值，避免字符串 ``"false"`` 被当成真。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"无法解析布尔值: {value!r}")
    return bool(value)


def competence_gate(score: float, epsilon: float = 0.1) -> float:
    """根据已达到的前缀质量，连续控制下一层排序目标的强度。

    ``epsilon`` 是最小开放程度。即使 Top-5 暂时没有命中，Top-10/20 仍保留微弱
    信号，从而避免 GRPO 组内所有排序奖励都为零。
    """

    eps = _clip(float(epsilon))
    return eps + (1.0 - eps) * _clip(float(score))


def hierarchical_rank_reward(
    ndcg_5: float,
    ndcg_10: float,
    ndcg_20: float,
    epsilon: float = 0.1,
) -> float:
    """能力门控的层级排序奖励。

    Top-5 始终直接优化；Top-10 由 Top-5 质量软激活；Top-20 只有在 Top-5 和
    Top-10 都较好时才充分激活。该奖励不依赖 epoch 或 global step，因此可在一次
    VERL 训练中完成样本级自节奏学习。
    """

    score_5 = _clip(float(ndcg_5))
    score_10 = _clip(float(ndcg_10))
    score_20 = _clip(float(ndcg_20))
    gate_5 = competence_gate(score_5, epsilon)
    gate_10 = competence_gate(score_10, epsilon)
    return (score_5 + gate_5 * score_10 + gate_5 * gate_10 * score_20) / 3.0


def fixed_joint_rank_reward(
    ndcg_5: float,
    ndcg_10: float,
    ndcg_20: float,
    weight_5: float = 0.4,
    weight_10: float = 0.3,
    weight_20: float = 0.3,
) -> float:
    """不使用能力门控时的固定联合奖励，用作严格消融对照。"""

    weights = [
        max(0.0, float(weight_5)),
        max(0.0, float(weight_10)),
        max(0.0, float(weight_20)),
    ]
    normalizer = sum(weights)
    if normalizer == 0.0:
        weights, normalizer = [0.4, 0.3, 0.3], 1.0
    scores = [_clip(float(ndcg_5)), _clip(float(ndcg_10)), _clip(float(ndcg_20))]
    return sum(weight * score for weight, score in zip(weights, scores)) / normalizer


def _string_list(value: Any) -> list[str]:
    """把列表型字段规范化为去除首尾空白的字符串列表。

    字符串本身不能被当作字符序列展开；非列表值直接视为空，奖励函数因此不会
    因单条脏样本抛异常而中断整个分布式训练任务。
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _extract_ground_truth(ground_truth: Any) -> list[str]:
    """兼容字典、列表以及 JSON 字符串形式的 ground truth。"""

    value = ground_truth
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Mapping):
        value = value.get("ground_truth_herbs", value.get("herbs", []))
    return _string_list(value)


def _extract_json_ranking(solution: Any) -> tuple[list[Any], float, float]:
    """从模型输出中提取 ``ranking`` 数组。

    返回值依次是原始数组、JSON 是否可解析、ranking 是否为列表。扫描 JSON 对象而
    不是简单正则，是为了兼容 Markdown 代码块以及 Qwen 偶尔残留的思考文本。
    输出协议本身仍是严格 JSON；兼容解析仅用于避免无关包装掩盖真实排序质量。
    """

    if not isinstance(solution, str):
        return [], 0.0, 0.0

    text = solution.strip()
    if "</think>" in text:
        # Qwen3 若没有完全遵守 /no_think，答案通常位于最后一个闭合标签之后。
        text = text.rsplit("</think>", maxsplit=1)[-1].strip()

    decoder = json.JSONDecoder()
    saw_json_object = False
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, Mapping):
            continue
        saw_json_object = True
        if "ranking" in obj:
            ranking = obj["ranking"]
            if isinstance(ranking, list):
                return ranking, 1.0, 1.0
            return [], 1.0, 0.0

    return [], float(saw_json_object), 0.0


def _ndcg_at_k(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    """计算二元相关性的 NDCG@cutoff；没有可达 GT 时返回 0。"""

    if not relevant or cutoff <= 0:
        return 0.0

    dcg = 0.0
    for rank, herb in enumerate(ranking[:cutoff], start=1):
        if herb in relevant:
            dcg += 1.0 / math.log2(rank + 1.0)

    ideal_hits = min(cutoff, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """计算一个生成结果的规则奖励，签名与 VERL 自定义奖励接口一致。

    可通过 ``reward.custom_reward_function.reward_kwargs`` 传入：
    - ``use_hierarchical_reward``：是否启用能力门控，默认启用；
    - ``hierarchical_epsilon``：能力门控的最小开放程度，默认 0.1；
    - ``fixed_weight_5/10/20``：关闭门控后的固定联合权重；
    - ``rank_weight``：排序项权重，默认 0.95；
    - ``format_weight``：格式项权重，默认 0.05。

    ``data_source`` 目前不参与计算，但必须保留在签名中供 VERL 调用。
    """

    del data_source  # 明确说明该参数仅用于满足 VERL 接口。
    info = extra_info if isinstance(extra_info, Mapping) else {}
    candidates = _string_list(info.get("candidate_herbs", []))
    targets = _extract_ground_truth(ground_truth)

    # 数据预处理会保证候选唯一；这里再次去重是为了让在线训练面对脏数据时保持稳定。
    candidates = list(dict.fromkeys(candidates))
    candidate_set = set(candidates)
    relevant = set(targets) & candidate_set

    raw_ranking, json_ok, list_ok = _extract_json_ranking(solution_str)
    raw_output_count = len(raw_ranking)
    output_items = [item.strip() for item in raw_ranking if isinstance(item, str) and item.strip()]

    # 只保留候选内中药的第一次出现；随后按 GNN 原顺序补齐遗漏项。
    # 补齐只用于定义稳定的排序指标，constraint_quality 会单独惩罚遗漏、重复与越界。
    seen: set[str] = set()
    valid_unique: list[str] = []
    for herb in output_items:
        if herb in candidate_set and herb not in seen:
            seen.add(herb)
            valid_unique.append(herb)
    completed_ranking = valid_unique + [herb for herb in candidates if herb not in seen]

    candidate_count = len(candidates)
    # 分母使用原始数组长度，使 null、数字和空字符串同样受到惩罚，不能在过滤后“消失”。
    candidate_precision = len(valid_unique) / max(raw_output_count, 1)
    candidate_coverage = len(valid_unique) / candidate_count if candidate_count else 0.0
    constraint_quality = candidate_precision * candidate_coverage

    exact_permutation = float(
        bool(candidates)
        and list_ok == 1.0
        and raw_output_count == candidate_count
        and len(valid_unique) == candidate_count
    )

    ndcg = {
        cutoff: _ndcg_at_k(completed_ranking, relevant, cutoff)
        for cutoff in (5, 10, 20)
    }
    use_hierarchical = _as_bool(kwargs.get("use_hierarchical_reward", True))
    epsilon = _clip(float(kwargs.get("hierarchical_epsilon", 0.1)))
    gate_5 = competence_gate(ndcg[5], epsilon)
    gate_10 = competence_gate(ndcg[10], epsilon)

    if use_hierarchical:
        rank_score = hierarchical_rank_reward(
            ndcg_5=ndcg[5],
            ndcg_10=ndcg[10],
            ndcg_20=ndcg[20],
            epsilon=epsilon,
        )
    else:
        rank_score = fixed_joint_rank_reward(
            ndcg_5=ndcg[5],
            ndcg_10=ndcg[10],
            ndcg_20=ndcg[20],
            weight_5=float(kwargs.get("fixed_weight_5", 0.4)),
            weight_10=float(kwargs.get("fixed_weight_10", 0.3)),
            weight_20=float(kwargs.get("fixed_weight_20", 0.3)),
        )

    # 采用分级格式奖励，避免训练早期所有输出都不合法而导致组内奖励完全相同。
    format_score = (
        0.20 * json_ok
        + 0.20 * list_ok
        + 0.30 * candidate_precision
        + 0.30 * candidate_coverage
    )

    rank_weight = _clip(float(kwargs.get("rank_weight", 0.95)))
    format_weight = _clip(float(kwargs.get("format_weight", 0.05)))
    normalizer = rank_weight + format_weight
    if normalizer == 0.0:
        rank_weight, format_weight, normalizer = 0.95, 0.05, 1.0
    rank_weight /= normalizer
    format_weight /= normalizer

    total_score = rank_weight * rank_score * constraint_quality + format_weight * format_score

    # score 是 VERL 使用的主奖励，其余标量会进入 reward_extra_info 便于监控和消融。
    return {
        "score": float(total_score),
        "rank_score": float(rank_score),
        "ndcg_5": float(ndcg[5]),
        "ndcg_10": float(ndcg[10]),
        "ndcg_20": float(ndcg[20]),
        "gate_5": float(gate_5),
        "gate_10": float(gate_10),
        "hierarchical_reward_enabled": float(use_hierarchical),
        "format_score": float(format_score),
        "candidate_precision": float(candidate_precision),
        "candidate_coverage": float(candidate_coverage),
        "constraint_quality": float(constraint_quality),
        "exact_permutation": float(exact_permutation),
        "reachable_gt_count": float(len(relevant)),
    }
