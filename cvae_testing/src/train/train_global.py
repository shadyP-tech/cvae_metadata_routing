from __future__ import annotations

from pathlib import Path

import torch

from src.torch_utils import safe_torch_load
from src.train.train_utils import run_training


def train_global_model(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")

    input_dim = int(train_payload["embeddings"].shape[1])
    result = run_training(
        train_embeddings=train_payload["embeddings"],
        val_embeddings=val_payload["embeddings"],
        out_dir=out_dir,
        model_name="global_cvae",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        resume_from=resume_from,
    )
    return result.checkpoint_path
