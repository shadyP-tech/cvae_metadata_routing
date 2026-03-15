from __future__ import annotations

from typing import Callable, Dict

from src.routing.strategies import (
    categorical_exact_similarity,
    matrix_similarity,
    ordinal_magnification_similarity,
)

StrategyFn = Callable[[int, int, float, Dict[str, Dict[str, float]] | None], float]


def _ordinal(query_id: int, expert_id: int, tau: float, similarity_matrix: Dict[str, Dict[str, float]] | None) -> float:
    _ = similarity_matrix
    return ordinal_magnification_similarity(query_mag=query_id, expert_mag=expert_id, tau=tau)


def _categorical(query_id: int, expert_id: int, tau: float, similarity_matrix: Dict[str, Dict[str, float]] | None) -> float:
    _ = tau
    _ = similarity_matrix
    return categorical_exact_similarity(query_id=query_id, expert_id=expert_id)


def _matrix(query_id: int, expert_id: int, tau: float, similarity_matrix: Dict[str, Dict[str, float]] | None) -> float:
    _ = tau
    if similarity_matrix is None:
        raise ValueError("site_similarity_matrix strategy requires similarity_matrix in config")
    return matrix_similarity(query_id=query_id, expert_id=expert_id, similarity_matrix=similarity_matrix)


STRATEGY_REGISTRY: Dict[str, StrategyFn] = {
    "ordinal_magnification": _ordinal,
    "categorical_exact": _categorical,
    "site_similarity_matrix": _matrix,
}


def resolve_strategy(name: str) -> StrategyFn:
    fn = STRATEGY_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown routing strategy: {name}. Available: {sorted(STRATEGY_REGISTRY)}")
    return fn
