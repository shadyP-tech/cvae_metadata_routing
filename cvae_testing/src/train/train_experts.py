from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from src.torch_utils import safe_torch_load
from src.train.train_utils import run_training


def _filter_by_domain(payload: Dict[str, object], domain: int):
    metadata = payload["metadata"]
    embeddings = payload["embeddings"]
    idxs = [i for i, m in enumerate(metadata) if int(m["magnification"]) == domain]
    if not idxs:
        return torch.empty((0, embeddings.shape[1]))
    return embeddings[idxs]


def train_domain_experts(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: list[int],
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from_dir: Path | None = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")
    input_dim = int(train_payload["embeddings"].shape[1])

    output: Dict[str, str] = {}
    for domain in domains:
        train_x = _filter_by_domain(train_payload, domain)
        val_x = _filter_by_domain(val_payload, domain)
        if train_x.numel() == 0 or val_x.numel() == 0:
            continue
        result = run_training(
            train_embeddings=train_x,
            val_embeddings=val_x,
            out_dir=out_dir,
            model_name=f"expert_{domain}x",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            resume_from=(resume_from_dir / f"expert_{domain}x.pt") if resume_from_dir is not None else None,
        )
        output[f"{domain}x"] = str(result.checkpoint_path)

    with (out_dir / "expert_checkpoints.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output
