from __future__ import annotations

from typing import Iterable, List

import torch


def reconstruction_mse(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((recon - x) ** 2).mean(dim=1)


def mean_and_variance(values: Iterable[float]) -> dict:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "var": 0.0}
    t = torch.tensor(vals, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "var": float(t.var(unbiased=False).item())}


def selection_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def average_rank_desc(values: List[float]) -> List[float]:
    """Return average ranks (1=best) for descending values, with tie handling."""
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    mx = tx.mean()
    my = ty.mean()
    vx = tx - mx
    vy = ty - my
    denom = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy)).item()
    if denom <= 1e-12:
        return 0.0
    return float(torch.sum(vx * vy).item() / denom)


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = average_rank_desc(x)
    ry = average_rank_desc(y)
    return pearson_corr(rx, ry)
