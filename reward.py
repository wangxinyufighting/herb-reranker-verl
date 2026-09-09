from __future__ import annotations

import os
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union, Set, Optional
import numpy as np
from collections import defaultdict
from collections.abc import Mapping, Sequence

def dedup_preserve_order(xs: Sequence[int]) -> List[int]:
    """
    去重并保持原始顺序
    用于去除预测列表中的重复ID，同时保持排名顺序
    
    Args:
        xs: 可能包含重复的ID序列
    
    Returns:
        去重后的ID列表（保持首次出现顺序）
        
    Examples:
        [1, 2, 1, 3, 2] -> [1, 2, 3]
    """
    seen = set()
    out: List[int] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def dcg_at_k(r: Sequence[float], k: int, method: int = 1) -> float:
    """
    Match utils/batch_test.py:dcg_at_k behavior (method=1 default).
    计算DCG@k (Discounted Cumulative Gain at k)
    衡量排序质量的指标，考虑位置折扣
    
    DCG公式:
    - method=0: DCG = r[0] + Σ(i=1..k) r[i] / log2(i+1)
    - method=1: DCG = Σ(i=0..k-1) r[i] / log2(i+2)  [默认，更常用]
    
    Args:
        r: 相关性分数列表 (1.0表示相关, 0.0表示不相关)
        k: 考虑前k个结果
        method: 计算方法 (0或1)
    
    Returns:
        DCG@k值
    
    Examples:
        dcg_at_k([1, 1, 0, 1], 3, method=1)
        = 1/log2(2) + 1/log2(3) + 0/log2(4)
        = 1 + 0.63 + 0 = 1.63
    """
    rr = list(r)[:k]
    if not rr:
        return 0.0
    if method == 0:
        # r[0] + sum_{i=1..} r[i] / log2(i+1)
        out = float(rr[0])
        for i in range(1, len(rr)):
            out += float(rr[i]) / math.log2(i + 1)
        return out
    if method == 1:
        # sum_{i=0..} r[i] / log2(i+2)
        out = 0.0
        for i, rel in enumerate(rr):
            out += float(rel) / math.log2(i + 2)
        return out
    raise ValueError("method must be 0 or 1")


def ndcg_at_k(r: Sequence[float], k: int, method: int = 1) -> float:
    """Match utils/batch_test.py:ndcg_at_k behavior."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


# Reward target setup: align with eval_herb_predictions.py Oracle metrics.
# KS: Tuple[int, ...] = (5, 10, 20)
# KS: Tuple[int, ...] = (5, 10, 20, 30)
KS: Tuple[int, ...] = (5, 10, 20, 30)
KMAX: int = 20

# Reward stage switch:
# - composite: 0.7*(0.7*NDCG@5 + 0.2*NDCG@10 + 0.1*NDCG@20) + 0.3*(0.7*Recall@5 + 0.2*Recall@10 + 0.1*Recall@20)
# - focus5/focus10/focus20: legacy single-k focused stages
DEFAULT_REWARD_STAGE = "stage1"

# Domain reward configuration
DOMAIN_REWARD_KS: Tuple[int, ...] = (5, 10, 20)  # K values for domain reward computation
DEFAULT_BETA: float = 1.0  # Penalty coefficient for negative compatibility
DOMAIN_PAIR_SCORE: float = 0.05

# Preprocessed compatibility data files (in the same directory as this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# Reward Profiles
# ═══════════════════════════════════════════════════════════════════

REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "stage1_domain_50.0": {
        "domain_reward_weight": 0.5,  # 50% domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "stage1_domain_100.0": {
        "domain_reward_weight": 1.0,  # 100% domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "stage1_domain_20.0": {
        "domain_reward_weight": 0.2,  # 20% domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "stage1_domain_1.0": {
        "domain_reward_weight": 1.0,  # Weight for domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "stage1_domain_0.5": {
        "domain_reward_weight": 0.5,  # Weight for domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "only5_n0_f10": {
        "domain_reward_weight": 0.0,  # Weight for domain reward component
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.0,   # 0.3 * 0.7
            "f1_5": 1.0, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.0,  # 0.3 * 0.2
            "f1_10": 0.0, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.0,  # 0.3 * 0.1
            "f1_20": 0.0, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },    

    "only5_n5_f5": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.5,   # 0.3 * 0.7
            "f1_5": 0.5, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.0,  # 0.3 * 0.2
            "f1_10": 0.0, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.0,  # 0.3 * 0.1
            "f1_20": 0.0, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "only5_n3_f7": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.7,   # 0.3 * 0.7
            "f1_5": 0.3, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.0,  # 0.3 * 0.2
            "f1_10": 0.0, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.0,  # 0.3 * 0.1
            "f1_20": 0.0, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "only15": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.0,   # 0.3 * 0.7
            "f1_5": 0.0, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.0,  # 0.3 * 0.2
            "f1_10": 0.0, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.3,  # 0.3 * 0.1
            "f1_20": 0.7, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "only10": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.0,   # 0.3 * 0.7
            "f1_5": 0.0, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.3,  # 0.3 * 0.2
            "f1_10": 0.7, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.0,  # 0.3 * 0.1
            "f1_20": 0.0, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "only5": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.3,   # 0.3 * 0.7
            "f1_5": 0.7, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.0,  # 0.3 * 0.2
            "f1_10": 0.0, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.0,  # 0.3 * 0.1
            "f1_20": 0.0, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "stage1": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    "stage2": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.06,  # 0.3 * 0.2
            "f1_5": 0.14, # 0.7 * 0.2

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.12,  # 0.3 * 0.4
            "f1_10": 0.28, # 0.7 * 0.4

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.12,  # 0.3 * 0.4
            "f1_20": 0.28, # 0.7 * 0.4

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "composite_hierarchical": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.10,   # Maintain k=5 optimization
            "f1_5": 0.10,

            "p10": 0.0,
            "r10": 0.10,  # Add explicit recall@10 to strengthen signal
            "n10": 0.15,  # Maintain k=10 optimization
            "f1_10": 0.15,

            "p20": 0.1,  # Add precision@20 to balance recall
            "r20": 0.25,  # Strong recall@20 signal
            "n20": 0.15,  # Maintain NDCG@20
            "f1_20": 0.0, # Remove F1@20 since we have explicit P and R

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.05,  # Reduce format weight to give more room for metrics
        "min_output_items": 20,
    },
    "compositev2": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.21,   # 0.3 * 0.7
            "f1_5": 0.49, # 0.7 * 0.7

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.06,  # 0.3 * 0.2
            "f1_10": 0.14, # 0.7 * 0.2

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.03,  # 0.3 * 0.1
            "f1_20": 0.07, # 0.7 * 0.1

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
    
    "compositev3": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.06,  # 0.3 * 0.2
            "f1_5": 0.14, # 0.7 * 0.2

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.12,  # 0.3 * 0.4
            "f1_10": 0.28, # 0.7 * 0.4

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.12,  # 0.3 * 0.4
            "f1_20": 0.28, # 0.7 * 0.4

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },

    "composite_all": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,   
            "n5": 0.3 * 0.5,   
            "f1_5": 0.3 * 0.5,

            "p10": 0.0,
            "r10": 0.0,  
            "n10": 0.3 * 0.5, 
            "f1_10": 0.3 * 0.5,

            "p20": 0.0,
            "r20": 0.0,  
            "n20": 0.3 * 0.5,  
            "f1_20": 0.3 * 0.5,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },

    "composite3": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.21,   # 0.3 * 0.7
            "n5": 0.49,   # 0.7 * 0.7
            "f1_5": 0.0,

            "p10": 0.0,
            "r10": 0.06,  # 0.3 * 0.2
            "n10": 0.14,  # 0.7 * 0.2
            "f1_10": 0.0,

            "p20": 0.0,
            "r20": 0.03,  # 0.3 * 0.1
            "n20": 0.07,  # 0.7 * 0.1
            "f1_20": 0.0,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },

    "focus5": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.3,
            "f1_5": 0.7,

            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.0,
            "f1_10": 0.0,

            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 0.0,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 5,
    },

    "focus10": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.0,
            "f1_5": 0.0,

            "p10": 0.2,
            "r10": 0.5,
            "n10": 0.3,
            "f1_10": 0.0,

            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 0.0,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 10,
    },

    "focus20": {
        "domain_reward_weight": 0.0,
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.0,
            "f1_5": 0.0,

            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.0,
            "f1_10": 0.0,

            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.2,
            "f1_20": 0.8,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },

    # ─────────────────────────────────────────────────────────
    # Domain-aware profiles (with compatibility reward)
    # ─────────────────────────────────────────────────────────
    "stage1_with_domain": {
        "domain_reward_weight": 0.15,  # 15% weight for domain compatibility
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.21,
            "f1_5": 0.49,

            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.06,
            "f1_10": 0.14,

            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.03,
            "f1_20": 0.07,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },

    "only5_with_domain": {
        "domain_reward_weight": 0.2,  # Higher weight for focused top-5 optimization
        "metric_weights": {
            "p5": 0.0,
            "r5": 0.0,
            "n5": 0.3,
            "f1_5": 0.7,

            "p10": 0.0,
            "r10": 0.0,
            "n10": 0.0,
            "f1_10": 0.0,

            "p20": 0.0,
            "r20": 0.0,
            "n20": 0.0,
            "f1_20": 0.0,

            "p30": 0.0,
            "r30": 0.0,
            "n30": 0.0,
            "f1_30": 0.0,
        },
        "format_weight": 0.1,
        "min_output_items": 20,
    },
}


def _get_reward_profile() -> Dict[str, Any]:
    stage = os.getenv("TCM_REWARD_STAGE", DEFAULT_REWARD_STAGE).strip().lower()
    if stage not in REWARD_PROFILES:
        stage = DEFAULT_REWARD_STAGE
    return REWARD_PROFILES[stage]


def extract_content(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_ranking_list(solution_str: str) -> List[str]:
    answer_content = extract_content(solution_str, "answer")
    if not answer_content:
        return []
    items = [part.strip() for part in answer_content.split(">")]
    return [x for x in items if x]


def check_response_format(solution_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.search(pattern, solution_str.strip(), re.DOTALL | re.IGNORECASE) else 0.0


def check_ranking_format(solution_str: str) -> float:
    answer_content = extract_content(solution_str, "answer")
    if not answer_content:
        return 0.0
    ranking_pattern = re.compile(r"^([^\s>]+(\s*>\s*[^\s>]+)*)$")
    return 1.0 if ranking_pattern.match(answer_content.strip()) else 0.0


def get_theoretical_optimal_score(gt_herbs: List[str], candidate_herbs: List[str], metric_weights: Dict[str, float] = None) -> float:
    if metric_weights is None:
        metric_weights = _get_reward_profile()["metric_weights"]

    gt_herbs_set = []
    for i in gt_herbs:
        if i not in gt_herbs_set:
            gt_herbs_set.append(i)

    ideal_ranking = [h for h in gt_herbs_set if h in candidate_herbs]
    irrelevant_in_order = [h for h in candidate_herbs if h not in ideal_ranking]
    ideal_ranking += irrelevant_in_order
    
    metrics = _oracle_metrics_for_ranking(ideal_ranking, gt_herbs)
    best_score = sum(w * metrics.get(m, 0.0) for m, w in metric_weights.items())
      
    return best_score


def _oracle_metrics_for_ranking(ranking: Sequence[str], gt_herbs: Sequence[str]) -> Dict[str, float]:
    ranking = dedup_preserve_order(list(ranking))
    topk_max = ranking[:KMAX]

    metrics = {}
    for key in ["p", "r", "n", "f1_"]:
        for k in KS:
            metrics[f"{key}{k}"] = 0.0

    for k in KS:
        topk = ranking[:k]
        gt_set = set(gt_herbs)
        hits = sum(1 for h in topk if h in gt_set)
        p = hits / k
        r = hits / max(1, len(gt_set))
        
        r_max = [1.0 if h in gt_set else 0.0 for h in topk_max]
        n = ndcg_at_k(r_max, k, method=1)

        metrics[f"p{k}"] = p
        metrics[f"r{k}"] = r
        metrics[f"n{k}"] = n
        
        divisor = (p + r)
        metrics[f"f1_{k}"] = (2 * p * r / divisor) if divisor > 0 else 0.0

    return metrics
    

def compute_score(
    solution_str: str,
    ground_truth: Union[Dict[str, Any], List[Any]],
    data_source: str = "",
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    
    extra_info = extra_info if isinstance(extra_info, Mapping) else {}
    
    profile = _get_reward_profile()
    
    candidate_herbs = ground_truth["candidate_herbs"]
    gt_herbs = ground_truth["gt_herbs"]
    
    best_score = get_theoretical_optimal_score(gt_herbs, candidate_herbs, profile["metric_weights"])
    
    base_metrics = _oracle_metrics_for_ranking(candidate_herbs, gt_herbs)
    base_score = sum(w * base_metrics[m] for m, w in profile["metric_weights"].items())

    # ── 解析模型输出 ───────────────────────────────────────────
    pred_raw   = extract_ranking_list(solution_str)
    pred_dedup = dedup_preserve_order(pred_raw)
    
    # 提前计算 pred_metrics，因为后续监控指标一定需要它，防止 UnboundLocalError
    pred_metrics = _oracle_metrics_for_ranking(pred_dedup, gt_herbs)
    pred_score = sum(w * pred_metrics[m] for m, w in profile["metric_weights"].items())
    
    # ── 约束惩罚 ──────────────────────────────────────────────
    dup_penalty = 0
    invalid_penalty = 0
    shortfall_penalty = 0
    copy_penalty = 0

    if not pred_raw:
        penalty = 1.0
    else:
        duplicate_count = max(0, len(pred_raw) - len(pred_dedup))
        invalid_count   = sum(1 for h in pred_dedup if h not in candidate_herbs)
        shortfall       = max(0, profile["min_output_items"] - len(pred_dedup))

        dup_penalty =       min(1, 0.03 * duplicate_count) if duplicate_count > 0 else 0.0
        invalid_penalty =   min(1, 0.03 * invalid_count)
        shortfall_penalty = min(1, 0.03 * shortfall)

        # Copy penalty: 只在"相似且质量没提升"时惩罚
        if abs(pred_score - best_score) < 1e-6:
            copy_penalty = 0.0
        else:
            k_min = min(len(pred_raw), len(candidate_herbs), KMAX)
            base_top_k = candidate_herbs[:k_min]
            pred_top_k = pred_raw[:k_min]

            pos_match_count = sum(1 for p, b in zip(pred_top_k, base_top_k) if p == b)
            pos_similarity = pos_match_count / k_min if k_min > 0 else 0.0

            baseline_quality = base_score / max(best_score, 0.01)

            # 根据baseline质量动态调整阈值
            if baseline_quality > 0.9:
                copy_threshold = 0.95
            elif baseline_quality > 0.7:
                copy_threshold = 0.85
            else:
                copy_threshold = 0.75

            # 只在"高度相似且质量没提升"时惩罚
            if pos_similarity > copy_threshold and pred_score <= base_score:
                copy_penalty = 0.5 * (pos_similarity - copy_threshold) / (1 - copy_threshold)
            else:
                copy_penalty = 0.0

        penalty = (
            dup_penalty
            + invalid_penalty
            + shortfall_penalty
            + copy_penalty
        )

    # ── 格式奖励 ──────────────────────────────────────────────
    format_reward = (
        0.5 * check_response_format(solution_str)
        + 0.5 * check_ranking_format(solution_str)
    )

    # ── 质量奖励：相对提升 ────────────────────────────────────
    if abs(pred_score - best_score) < 1e-6:
        quality_reward = 1.0
    else:
        quality_reward = pred_score - base_score

    # ── 领域奖励：中药配伍兼容性 ──────────────────────────────
    domain_reward_by_k = {}
    base_domain_reward_by_k = {}
    domain_reward_raw_total = 0.0
    base_domain_reward_total = 0.0
    domain_reward_total = 0.0

    # ── 汇总 ──────────────────────────────────────────────────
    penalty = min(penalty, 0.5)  # 总惩罚上限0.5分，防止过度惩罚导致训练不稳定

    if quality_reward < 0:
        # 负奖励时：减轻惩罚强度，增大组内差异，加快学习
        final = quality_reward * 0.5 - penalty * 0.3
    else:
        # 正奖励时：正常处理
        final = (
            quality_reward
            + profile["format_weight"] * format_reward
            + profile["domain_reward_weight"] * domain_reward_total
            - penalty
        )

    # ── 监控指标 ──────────────────────────────────────────────
    extra_info = {}
    for k in KS:
        for m in ["p", "r", "n"]:
            key = f"{m}{k}"
            extra_info[f"rerank_{key}"]  = pred_metrics[key]
            extra_info[f"prerank_{key}"] = base_metrics[key]
            extra_info[f"diff_{key}"]    = pred_metrics[key] - base_metrics[key]

    extra_info['format_reward'] = format_reward
    extra_info['quality_reward'] = quality_reward

    extra_info['invalid_penalty'] = invalid_penalty
    extra_info['shortfall_penalty'] = shortfall_penalty
    extra_info['dup_penalty'] = dup_penalty
    extra_info['copy_penalty'] = copy_penalty
    extra_info['penalty'] = penalty

    # Domain reward metrics
    extra_info['domain_reward_total'] = domain_reward_total
    extra_info['domain_reward_raw_total'] = domain_reward_raw_total
    extra_info['base_domain_reward_total'] = base_domain_reward_total
    for k in DOMAIN_REWARD_KS:
        extra_info[f'domain_reward_{k}'] = domain_reward_by_k.get(k, 0.0)
        extra_info[f'base_domain_reward_{k}'] = base_domain_reward_by_k.get(k, 0.0)


    return {
        "score": final,
        **extra_info,
    }
