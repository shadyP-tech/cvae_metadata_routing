from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional


def ordinal_magnification_similarity(query_mag: int, expert_mag: int, tau: float) -> float:
    return math.exp(-abs(query_mag - expert_mag) / tau)


def categorical_exact_similarity(query_id: int, expert_id: int) -> float:
    return 1.0 if int(query_id) == int(expert_id) else 0.0


def matrix_similarity(
    query_id: int,
    expert_id: int,
    similarity_matrix: Dict[str, Dict[str, float]],
) -> float:
    q = str(int(query_id))
    e = str(int(expert_id))
    if q in similarity_matrix and e in similarity_matrix[q]:
        return float(similarity_matrix[q][e])
    return 0.0


def compute_similarity(
    query_meta: dict,
    expert_meta: dict,
    strategy: str,
    tau: float,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    query_mag = int(query_meta["magnification"])
    expert_mag = int(expert_meta["magnification"])
    # Lazy import avoids import cycles while keeping strategies module as the main call site.
    from src.routing.registry import resolve_strategy

    strategy_fn = resolve_strategy(strategy)
    return strategy_fn(query_mag, expert_mag, float(tau), similarity_matrix)


def softmax(values: Iterable[float], temperature: float = 1.0) -> List[float]:
    vals = [v / max(temperature, 1e-8) for v in values]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]
