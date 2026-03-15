from __future__ import annotations

from typing import Any, Dict


REQUIRED_TOP_LEVEL = ["seed", "data", "features", "model", "training", "routing"]


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dictionary.")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    split = cfg.get("data", {}).get("split")
    if not isinstance(split, dict):
        raise ValueError("data.split must be a dictionary containing train/val/test ratios.")

    for key in ["train", "val", "test"]:
        if key not in split:
            raise ValueError(f"data.split must include '{key}'.")

    train = float(split["train"])
    val = float(split["val"])
    test = float(split["test"])
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"data.split ratios must sum to 1.0, got {total:.6f}")

    if int(cfg["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be > 0")
    if int(cfg["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be > 0")

    magnifications = cfg.get("data", {}).get("magnifications", [])
    if not isinstance(magnifications, list) or not magnifications:
        raise ValueError("data.magnifications must be a non-empty list")
    for m in magnifications:
        if int(m) < 0:
            raise ValueError(f"data.magnifications must contain only non-negative integers, got: {m}")

    routing_strategy = str(cfg.get("routing", {}).get("strategy", "")).strip()
    if not routing_strategy:
        raise ValueError("routing.strategy must be provided")
    from src.routing.registry import STRATEGY_REGISTRY

    if routing_strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"routing.strategy must be one of {sorted(STRATEGY_REGISTRY)}, got: {routing_strategy}"
        )

    tracking = cfg.get("tracking")
    if tracking is not None:
        if not isinstance(tracking, dict):
            raise ValueError("tracking must be a dictionary when provided")

        backend = str(tracking.get("backend", "wandb")).strip().lower()
        if backend not in {"wandb"}:
            raise ValueError(f"tracking.backend must be one of ['wandb'], got: {backend}")

        tags = tracking.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError("tracking.tags must be a list when provided")

    return cfg
