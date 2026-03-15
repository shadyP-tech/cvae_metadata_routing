from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.routing.strategies import compute_similarity, softmax


def route_hard(
    query_meta: dict,
    experts_meta: List[dict],
    strategy: str,
    tau: float,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[int, List[float]]:
    sims = [
        compute_similarity(
            query_meta,
            em,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        for em in experts_meta
    ]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return best_idx, sims


def route_soft(
    query_meta: dict,
    experts_meta: List[dict],
    strategy: str,
    tau: float,
    temperature: float,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[float]:
    sims = [
        compute_similarity(
            query_meta,
            em,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        for em in experts_meta
    ]
    return softmax(sims, temperature=temperature)


def uniform_sampling_weights(num_experts: int) -> List[float]:
    if num_experts <= 0:
        return []
    w = 1.0 / num_experts
    return [w for _ in range(num_experts)]


def equal_weight_scoring_weights(num_experts: int) -> List[float]:
    return uniform_sampling_weights(num_experts)


def confusion_update(matrix: Dict[str, Dict[str, int]], true_domain: str, pred_domain: str) -> None:
    if true_domain not in matrix:
        matrix[true_domain] = {}
    matrix[true_domain][pred_domain] = matrix[true_domain].get(pred_domain, 0) + 1
