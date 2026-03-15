from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.eval.metrics import mean_and_variance
from src.models.cvae_expert import CVAEExpert, elbo_components


def _load_model(checkpoint: Path, input_dim: int, hidden_dim: int, latent_dim: int, device: torch.device):
    model = CVAEExpert(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def compute_expert_domain_matrix(
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
) -> Dict[str, object]:
    payload = torch.load(test_cache, map_location="cpu")
    x = payload["embeddings"]
    meta = payload["metadata"]
    input_dim = int(x.shape[1])

    domains = sorted(set(int(m["magnification"]) for m in meta))
    by_domain_indices = {d: [i for i, m in enumerate(meta) if int(m["magnification"]) == d] for d in domains}

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x.to(device)

    matrix: Dict[str, Dict[str, float]] = {}
    confidence: Dict[str, Dict[str, dict]] = {}

    for expert_domain, ckpt in expert_checkpoints.items():
        model = _load_model(Path(ckpt), input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
        matrix[expert_domain] = {}
        confidence[expert_domain] = {}

        with torch.no_grad():
            for d in domains:
                idxs = by_domain_indices[d]
                if not idxs:
                    continue
                xs = x[idxs]
                recon, mu, logvar = model(xs)
                rec, kl = elbo_components(recon, xs, mu, logvar)
                nelbo = rec + kl
                matrix[expert_domain][f"{d}x"] = float(rec.mean().item())
                confidence[expert_domain][f"{d}x"] = mean_and_variance(nelbo.tolist())

    return {
        "reconstruction_matrix": matrix,
        "confidence": confidence,
    }
