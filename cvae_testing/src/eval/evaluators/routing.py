from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Optional

import torch

from src.eval.metrics import selection_accuracy
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.router import (
    confusion_update,
    equal_weight_scoring_weights,
    route_hard,
    route_soft,
)


def _load_model(checkpoint: Path, input_dim: int, hidden_dim: int, latent_dim: int, device: torch.device):
    model = CVAEExpert(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _score_model(model: CVAEExpert, x: torch.Tensor) -> torch.Tensor:
    recon, mu, logvar = model(x)
    rec, kl = elbo_components(recon, x, mu, logvar)
    return rec + kl


def _reconstruction_only(model: CVAEExpert, x: torch.Tensor) -> torch.Tensor:
    recon, mu, logvar = model(x)
    rec, _ = elbo_components(recon, x, mu, logvar)
    return rec


def evaluate_routing(
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    global_checkpoint: Path,
    hidden_dim: int,
    latent_dim: int,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    rng = random.Random(seed)
    payload = torch.load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    meta = payload["metadata"]

    input_dim = int(x_cpu.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x_cpu.to(device)

    expert_names = sorted(expert_checkpoints.keys())
    expert_mags = [int(name.replace("expert_", "").replace("x", "")) if "expert_" in name else int(name.replace("x", "")) for name in expert_names]

    expert_models = [
        _load_model(Path(expert_checkpoints[name]), input_dim, hidden_dim, latent_dim, device)
        for name in expert_names
    ]
    global_model = _load_model(global_checkpoint, input_dim, hidden_dim, latent_dim, device)

    with torch.no_grad():
        all_scores = []
        all_recon = []
        for m in expert_models:
            all_scores.append(_score_model(m, x))
            all_recon.append(_reconstruction_only(m, x))
        # [num_experts, num_samples]
        expert_scores = torch.stack(all_scores, dim=0)
        expert_recon = torch.stack(all_recon, dim=0)
        global_scores = _score_model(global_model, x)
        global_recon = _reconstruction_only(global_model, x)

    hard_scores = []
    soft_scores = []
    random_scores = []
    uniform_sampling_scores = []
    equal_scores = []
    oracle_scores = []
    hard_recon = []
    soft_recon = []
    random_recon = []
    uniform_sampling_recon = []
    equal_recon = []
    oracle_recon = []
    global_baseline_scores = global_scores.tolist()
    global_baseline_recon = global_recon.tolist()

    true_domains: List[str] = []
    routed_domains: List[str] = []
    confusion: Dict[str, Dict[str, int]] = {}

    experts_meta = [{"magnification": m} for m in expert_mags]

    fixed_random_idx = rng.randrange(len(expert_models))

    for i, sample_meta in enumerate(meta):
        query_meta = {"magnification": int(sample_meta["magnification"])}
        true_domain = f"{query_meta['magnification']}x"
        true_domains.append(true_domain)

        hard_idx, _ = route_hard(
            query_meta,
            experts_meta,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        routed_domain = f"{expert_mags[hard_idx]}x"
        routed_domains.append(routed_domain)
        confusion_update(confusion, true_domain=true_domain, pred_domain=routed_domain)
        hard_scores.append(float(expert_scores[hard_idx, i].item()))
        hard_recon.append(float(expert_recon[hard_idx, i].item()))

        soft_w = route_soft(
            query_meta,
            experts_meta,
            strategy=strategy,
            tau=tau,
            temperature=temperature,
            similarity_matrix=similarity_matrix,
        )
        soft_scores.append(float(sum(soft_w[j] * expert_scores[j, i].item() for j in range(len(expert_models)))))
        soft_recon.append(float(sum(soft_w[j] * expert_recon[j, i].item() for j in range(len(expert_models)))))

        random_scores.append(float(expert_scores[fixed_random_idx, i].item()))
        random_recon.append(float(expert_recon[fixed_random_idx, i].item()))

        sampled_idx = rng.randrange(len(expert_models))
        uniform_sampling_scores.append(float(expert_scores[sampled_idx, i].item()))
        uniform_sampling_recon.append(float(expert_recon[sampled_idx, i].item()))

        eq_w = equal_weight_scoring_weights(len(expert_models))
        equal_scores.append(float(sum(eq_w[j] * expert_scores[j, i].item() for j in range(len(expert_models)))))
        equal_recon.append(float(sum(eq_w[j] * expert_recon[j, i].item() for j in range(len(expert_models)))))

        oracle_idx = expert_mags.index(query_meta["magnification"]) if query_meta["magnification"] in expert_mags else hard_idx
        oracle_scores.append(float(expert_scores[oracle_idx, i].item()))
        oracle_recon.append(float(expert_recon[oracle_idx, i].item()))

    results = {
        "metrics": {
            "hard_metadata_routing_nelbo": float(torch.tensor(hard_scores).mean().item()),
            "soft_metadata_routing_nelbo": float(torch.tensor(soft_scores).mean().item()),
            "random_expert_nelbo": float(torch.tensor(random_scores).mean().item()),
            "uniform_sampling_nelbo": float(torch.tensor(uniform_sampling_scores).mean().item()),
            "equal_weight_scoring_nelbo": float(torch.tensor(equal_scores).mean().item()),
            "global_cvae_nelbo": float(torch.tensor(global_baseline_scores).mean().item()),
            "oracle_expert_nelbo": float(torch.tensor(oracle_scores).mean().item()),
            "hard_metadata_routing_recon": float(torch.tensor(hard_recon).mean().item()),
            "soft_metadata_routing_recon": float(torch.tensor(soft_recon).mean().item()),
            "random_expert_recon": float(torch.tensor(random_recon).mean().item()),
            "uniform_sampling_recon": float(torch.tensor(uniform_sampling_recon).mean().item()),
            "equal_weight_scoring_recon": float(torch.tensor(equal_recon).mean().item()),
            "global_cvae_recon": float(torch.tensor(global_baseline_recon).mean().item()),
            "oracle_expert_recon": float(torch.tensor(oracle_recon).mean().item()),
            "routing_selection_accuracy": selection_accuracy(true_domains, routed_domains),
        },
        "routing": {
            "confusion_matrix": confusion,
            "true_domains": true_domains,
            "routed_domains": routed_domains,
            "random_expert_choice": f"{expert_mags[fixed_random_idx]}x",
        },
    }
    return results
