from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import torch

from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.models.projection_head import ProjectionHead
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load


@dataclass
class RoutingStats:
    spearman_similarity_vs_neg_nelbo: float
    top1_agreement_with_best_expert: float
    mean_rank_of_metadata_selected_expert: float


class HybridExpertBank:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        payload = safe_torch_load(checkpoint, map_location=device)
        self.variant = str(payload["variant"])
        self.domains = [int(d) for d in payload["domains"]]
        self.input_dim = int(payload["input_dim"])
        self.projection_dim = int(payload["projection_dim"])
        self.head_hidden_dim = int(payload["head_hidden_dim"])
        self.cvae_hidden_dim = int(payload["cvae_hidden_dim"])
        self.latent_dim = int(payload["latent_dim"])
        self.device = device

        self.shared_head: ProjectionHead | None = None
        self.shared_cvae: CVAEExpert | None = None
        self.heads: Dict[int, ProjectionHead] = {}
        self.cvaes: Dict[int, CVAEExpert] = {}

        if payload.get("shared_head") is not None:
            self.shared_head = ProjectionHead(self.input_dim, self.projection_dim, self.head_hidden_dim).to(device)
            self.shared_head.load_state_dict(payload["shared_head"])
            self.shared_head.eval()

        if payload.get("shared_cvae") is not None:
            self.shared_cvae = CVAEExpert(self.projection_dim, self.cvae_hidden_dim, self.latent_dim).to(device)
            self.shared_cvae.load_state_dict(payload["shared_cvae"])
            self.shared_cvae.eval()

        for k, state in payload.get("heads", {}).items():
            d = int(k)
            m = ProjectionHead(self.input_dim, self.projection_dim, self.head_hidden_dim).to(device)
            m.load_state_dict(state)
            m.eval()
            self.heads[d] = m

        for k, state in payload.get("cvaes", {}).items():
            d = int(k)
            m = CVAEExpert(self.projection_dim, self.cvae_hidden_dim, self.latent_dim).to(device)
            m.load_state_dict(state)
            m.eval()
            self.cvaes[d] = m

    def _head_for_domain(self, domain: int) -> ProjectionHead:
        if self.shared_head is not None:
            return self.shared_head
        return self.heads[int(domain)]

    def _cvae_for_domain(self, domain: int) -> CVAEExpert:
        if self.shared_cvae is not None:
            return self.shared_cvae
        return self.cvaes[int(domain)]

    def score_domain_nelbo(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(expert_domain)
        cvae = self._cvae_for_domain(expert_domain)
        proj = head(x)
        recon, mu, logvar = cvae(proj)
        rec, kl = elbo_components(recon, proj, mu, logvar)
        return rec + kl

    def score_domain_recon(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(expert_domain)
        cvae = self._cvae_for_domain(expert_domain)
        proj = head(x)
        recon, mu, logvar = cvae(proj)
        rec, _ = elbo_components(recon, proj, mu, logvar)
        return rec

    def project(self, domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(domain)
        return head(x)

    def generate_from_reference(self, domain: int, x_ref: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
        head = self._head_for_domain(domain)
        cvae = self._cvae_for_domain(domain)
        if x_ref.shape[0] == 0 or n_samples <= 0:
            return torch.empty((0, self.projection_dim), device=self.device)
        rng = random.Random(seed)
        with torch.no_grad():
            proj_ref = head(x_ref)
            mu, logvar = cvae.encode(proj_ref)
            idxs = [rng.randrange(int(mu.shape[0])) for _ in range(int(n_samples))]
            mu_s = mu[idxs]
            logvar_s = logvar[idxs]
            z = cvae.reparameterize(mu_s, logvar_s)
            return cvae.decode(z)


def _payload_by_domain(payload: Dict[str, object]) -> Dict[int, List[int]]:
    meta = payload["metadata"]
    domains = sorted(set(int(m["magnification"]) for m in meta))
    return {d: [i for i, m in enumerate(meta) if int(m["magnification"]) == d] for d in domains}


def _hard_domain_route(
    target_domain: int,
    expert_domains: List[int],
    strategy: str,
    tau: float,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[int, List[float]]:
    sims = [
        compute_similarity(
            {"magnification": int(target_domain)},
            {"magnification": int(d)},
            strategy=strategy,
            tau=float(tau),
            similarity_matrix=similarity_matrix,
        )
        for d in expert_domains
    ]
    best_idx = max(range(len(expert_domains)), key=lambda i: sims[i])
    return best_idx, sims


def _rank_desc(values: List[float]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    rank = [0] * len(values)
    for r, idx in enumerate(order, start=1):
        rank[idx] = r
    return rank


def _score_domains_batched(
    bank: HybridExpertBank,
    expert_domains: List[int],
    x_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_scores: List[torch.Tensor] = []
    all_recon: List[torch.Tensor] = []

    with torch.no_grad():
        for ed in expert_domains:
            score_batches: List[torch.Tensor] = []
            recon_batches: List[torch.Tensor] = []
            for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[i : i + int(batch_size)].to(device)
                score_batches.append(bank.score_domain_nelbo(ed, xb).cpu())
                recon_batches.append(bank.score_domain_recon(ed, xb).cpu())
            all_scores.append(torch.cat(score_batches, dim=0) if score_batches else torch.empty((0,), dtype=torch.float32))
            all_recon.append(torch.cat(recon_batches, dim=0) if recon_batches else torch.empty((0,), dtype=torch.float32))

    return torch.stack(all_scores, dim=0), torch.stack(all_recon, dim=0)


def compute_hybrid_matrices_and_routing(
    test_cache: Path,
    hybrid_checkpoint: Path,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    by_domain = _payload_by_domain(payload)
    domains = sorted(by_domain.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = HybridExpertBank(hybrid_checkpoint, device=device)
    expert_domains = [d for d in domains if d in bank.domains]
    if not expert_domains:
        raise RuntimeError("No overlapping expert domains found for hybrid routing evaluation.")

    compatibility_nelbo: Dict[str, Dict[str, float]] = {}
    compatibility_recon: Dict[str, Dict[str, float]] = {}

    # Dense scoring matrix [expert_idx, sample_idx], computed in batches to cap peak GPU memory.
    score_tensor, recon_tensor = _score_domains_batched(
        bank=bank,
        expert_domains=expert_domains,
        x_cpu=x_cpu,
        device=device,
        batch_size=2048,
    )

    for e_i, ed in enumerate(expert_domains):
        row_n = {}
        row_r = {}
        for td in domains:
            idxs = by_domain[td]
            if not idxs:
                continue
            row_n[f"{td}x"] = float(score_tensor[e_i, idxs].mean().item())
            row_r[f"{td}x"] = float(recon_tensor[e_i, idxs].mean().item())
        compatibility_nelbo[f"{ed}x"] = row_n
        compatibility_recon[f"{ed}x"] = row_r

    routing_matrix_counts = {f"{ed}x": {f"{td}x": 0 for td in domains} for ed in expert_domains}

    metadata_scores = []
    oracle_scores = []
    random_scores = []
    uniform_scores = []
    true_domains = []
    metadata_selected_domains = []
    oracle_selected_domains = []

    rng = random.Random(seed)
    fixed_random_idx = rng.randrange(len(expert_domains))

    domain_stats: List[RoutingStats] = []

    for td in domains:
        idxs = by_domain[td]
        if not idxs:
            continue

        hard_idx, sims = _hard_domain_route(
            target_domain=td,
            expert_domains=expert_domains,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        utilities = []
        for e_i in range(len(expert_domains)):
            nelbo_mean = float(score_tensor[e_i, idxs].mean().item())
            utilities.append(-nelbo_mean)

        best_idx = max(range(len(expert_domains)), key=lambda i: utilities[i])
        top1_agree = 1.0 if best_idx == hard_idx else 0.0
        ranks = _rank_desc(utilities)
        selected_rank = float(ranks[hard_idx])
        rho = spearman_corr(sims, utilities)
        domain_stats.append(
            RoutingStats(
                spearman_similarity_vs_neg_nelbo=rho,
                top1_agreement_with_best_expert=top1_agree,
                mean_rank_of_metadata_selected_expert=selected_rank,
            )
        )

        for idx in idxs:
            sample_scores = [float(score_tensor[e_i, idx].item()) for e_i in range(len(expert_domains))]
            oracle_idx = min(range(len(expert_domains)), key=lambda i: sample_scores[i])
            sampled_idx = rng.randrange(len(expert_domains))

            metadata_scores.append(sample_scores[hard_idx])
            oracle_scores.append(sample_scores[oracle_idx])
            random_scores.append(sample_scores[fixed_random_idx])
            uniform_scores.append(sample_scores[sampled_idx])

            true_domains.append(f"{td}x")
            metadata_selected_domains.append(f"{expert_domains[hard_idx]}x")
            oracle_selected_domains.append(f"{expert_domains[oracle_idx]}x")
            routing_matrix_counts[f"{expert_domains[hard_idx]}x"][f"{td}x"] += 1

    routing_matrix = {}
    for e in expert_domains:
        row = {}
        for td in domains:
            denom = max(len(by_domain[td]), 1)
            row[f"{td}x"] = routing_matrix_counts[f"{e}x"][f"{td}x"] / denom
        routing_matrix[f"{e}x"] = row

    if domain_stats:
        stats = {
            "spearman_similarity_vs_neg_nelbo": float(
                sum(s.spearman_similarity_vs_neg_nelbo for s in domain_stats) / len(domain_stats)
            ),
            "top1_agreement_with_best_expert": float(
                sum(s.top1_agreement_with_best_expert for s in domain_stats) / len(domain_stats)
            ),
            "mean_rank_of_metadata_selected_expert": float(
                sum(s.mean_rank_of_metadata_selected_expert for s in domain_stats) / len(domain_stats)
            ),
        }
    else:
        stats = {
            "spearman_similarity_vs_neg_nelbo": 0.0,
            "top1_agreement_with_best_expert": 0.0,
            "mean_rank_of_metadata_selected_expert": 0.0,
        }

    return {
        "compatibility_matrix": {
            "nelbo": compatibility_nelbo,
            "reconstruction": compatibility_recon,
        },
        "routing_matrix": routing_matrix,
        "routing_statistics": stats,
        "routing_metrics": {
            "metadata_routing_nelbo": float(torch.tensor(metadata_scores).mean().item()) if metadata_scores else 0.0,
            "oracle_routing_nelbo": float(torch.tensor(oracle_scores).mean().item()) if oracle_scores else 0.0,
            "random_routing_nelbo": float(torch.tensor(random_scores).mean().item()) if random_scores else 0.0,
            "uniform_routing_nelbo": float(torch.tensor(uniform_scores).mean().item()) if uniform_scores else 0.0,
            "metadata_to_oracle_gap": (
                float(torch.tensor(metadata_scores).mean().item() - torch.tensor(oracle_scores).mean().item())
                if metadata_scores and oracle_scores
                else 0.0
            ),
        },
        "routing_assignments": {
            "true_domains": true_domains,
            "metadata_selected_domains": metadata_selected_domains,
            "oracle_selected_domains": oracle_selected_domains,
        },
    }


def _domain_labels(payload: Dict[str, object], idxs: List[int]) -> torch.Tensor:
    return torch.tensor([int(payload["metadata"][i]["label"]) for i in idxs], dtype=torch.long)


def _allocate_class_counts(labels: torch.Tensor, total: int) -> Dict[int, int]:
    classes = [0, 1]
    if labels.numel() == 0 or total <= 0:
        return {0: 0, 1: 0}
    counts = {c: int((labels == c).sum().item()) for c in classes}
    n = int(labels.numel())
    target = {c: int(round(total * counts[c] / max(n, 1))) for c in classes}
    diff = total - sum(target.values())
    # Assign remainder to majority class.
    major = max(classes, key=lambda c: counts[c])
    target[major] += diff
    return target


def _train_eval_logreg(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> Dict[str, float]:
    try:
        import importlib

        linear_model = importlib.import_module("sklearn.linear_model")
        sk_metrics = importlib.import_module("sklearn.metrics")
        LogisticRegression = getattr(linear_model, "LogisticRegression")
        balanced_accuracy_score = getattr(sk_metrics, "balanced_accuracy_score")
        accuracy_score = getattr(sk_metrics, "accuracy_score")
        roc_auc_score = getattr(sk_metrics, "roc_auc_score")
    except Exception:
        return {"auroc": 0.0, "balanced_accuracy": 0.0}

    if x_train.shape[0] < 2 or x_test.shape[0] < 2:
        return {"auroc": 0.0, "balanced_accuracy": 0.0}

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train.cpu().numpy(), y_train.cpu().numpy())
    xte_np = x_test.cpu().numpy()
    yte_np = y_test.cpu().numpy()
    probs = clf.predict_proba(xte_np)[:, 1]
    preds = clf.predict(xte_np)

    # AUROC is undefined with single-class targets; use neutral 0.5 for reporting stability.
    if len(set(yte_np.tolist())) < 2:
        auroc = 0.5
        bacc = float(accuracy_score(yte_np, preds))
    else:
        auroc = float(roc_auc_score(yte_np, probs))
        bacc = float(balanced_accuracy_score(yte_np, preds))

    return {
        "auroc": auroc,
        "balanced_accuracy": bacc,
    }


def evaluate_downstream_utility(
    train_cache: Path,
    test_cache: Path,
    hybrid_checkpoint: Path,
    pooled_checkpoint: Path,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    budget_multipliers: List[float],
) -> Dict[str, object]:
    _ = temperature
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    test_payload = safe_torch_load(test_cache, map_location="cpu")
    train_by_domain = _payload_by_domain(train_payload)
    test_by_domain = _payload_by_domain(test_payload)

    domains = sorted(train_by_domain.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    routed_bank = HybridExpertBank(hybrid_checkpoint, device=device)
    pooled_bank = HybridExpertBank(pooled_checkpoint, device=device)

    x_train_cpu = train_payload["embeddings"]
    x_test_cpu = test_payload["embeddings"]

    out: Dict[str, object] = {}

    for budget in budget_multipliers:
        budget_key = f"budget_{budget:.1f}x"
        out[budget_key] = {}
        rng = random.Random(seed + int(budget * 1000))

        for d in domains:
            train_idxs = train_by_domain.get(d, [])
            test_idxs = test_by_domain.get(d, [])
            if not train_idxs or not test_idxs:
                continue

            # Domain-level top-1 routing.
            expert_domains = [ed for ed in routed_bank.domains if ed in domains]
            hard_idx, _ = _hard_domain_route(
                target_domain=d,
                expert_domains=expert_domains,
                strategy=strategy,
                tau=tau,
                similarity_matrix=None,
            )
            routed_domain = expert_domains[hard_idx]
            random_domain = expert_domains[rng.randrange(len(expert_domains))]

            xtr_local = x_train_cpu[train_idxs].to(device)
            ytr_local = _domain_labels(train_payload, train_idxs).to(device)
            xte_local = x_test_cpu[test_idxs].to(device)
            yte_local = _domain_labels(test_payload, test_idxs).to(device)

            # Use routed projection head as the common representation frame for this domain.
            with torch.no_grad():
                xtr_real_proj = routed_bank.project(routed_domain, xtr_local)
                xte_real_proj = routed_bank.project(routed_domain, xte_local)

            n_syn = int(round(float(budget) * int(xtr_local.shape[0])))
            class_targets = _allocate_class_counts(ytr_local.cpu(), n_syn)

            def _make_condition(bank: HybridExpertBank, gen_domain: int, cond_seed: int):
                synth_chunks = []
                label_chunks = []
                for cls, n_cls in class_targets.items():
                    if n_cls <= 0:
                        continue
                    class_mask = ytr_local == int(cls)
                    if not torch.any(class_mask):
                        continue
                    ref = xtr_local[class_mask]
                    x_syn = bank.generate_from_reference(gen_domain, ref, n_cls, seed=cond_seed + int(cls))
                    y_syn = torch.full((int(x_syn.shape[0]),), int(cls), dtype=torch.long, device=device)
                    synth_chunks.append(x_syn)
                    label_chunks.append(y_syn)
                if not synth_chunks:
                    return torch.empty((0, xtr_real_proj.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)
                return torch.cat(synth_chunks, dim=0), torch.cat(label_chunks, dim=0)

            x_syn_routed, y_syn_routed = _make_condition(routed_bank, routed_domain, seed + 11)
            x_syn_random, y_syn_random = _make_condition(routed_bank, random_domain, seed + 23)
            x_syn_pooled, y_syn_pooled = _make_condition(pooled_bank, d, seed + 37)

            conditions = {
                "real_only": (xtr_real_proj, ytr_local),
                "real_plus_routed_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_routed], dim=0),
                    torch.cat([ytr_local, y_syn_routed], dim=0),
                ),
                "real_plus_random_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_random], dim=0),
                    torch.cat([ytr_local, y_syn_random], dim=0),
                ),
                "real_plus_pooled_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_pooled], dim=0),
                    torch.cat([ytr_local, y_syn_pooled], dim=0),
                ),
            }

            domain_res = {
                "synthetic_budget": n_syn,
                "class_targets": {str(k): int(v) for k, v in class_targets.items()},
                "routed_domain": f"{routed_domain}x",
                "random_domain": f"{random_domain}x",
                "metrics": {},
            }
            for name, (xtr_c, ytr_c) in conditions.items():
                domain_res["metrics"][name] = _train_eval_logreg(
                    x_train=xtr_c,
                    y_train=ytr_c,
                    x_test=xte_real_proj,
                    y_test=yte_local,
                    seed=seed,
                )
            out[budget_key][f"{d}x"] = domain_res

    return out


def evaluate_global_baselines(
    test_cache: Path,
    legacy_global_checkpoint: Path,
    legacy_hidden_dim: int,
    legacy_latent_dim: int,
    pooled_checkpoint: Path,
) -> Dict[str, float]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x_cpu.to(device)

    # Legacy baseline: global CVAE directly on shared foundation embeddings.
    legacy = CVAEExpert(
        input_dim=int(x.shape[1]),
        hidden_dim=int(legacy_hidden_dim),
        latent_dim=int(legacy_latent_dim),
    ).to(device)
    legacy.load_state_dict(safe_torch_load(legacy_global_checkpoint, map_location=device))
    legacy.eval()

    with torch.no_grad():
        recon, mu, logvar = legacy(x)
        rec, kl = elbo_components(recon, x, mu, logvar)
        legacy_nelbo = float((rec + kl).mean().item())

    # Matched pooled hybrid baseline: shared projection head + shared CVAE.
    pooled_bank = HybridExpertBank(pooled_checkpoint, device=device)
    pooled_domain = int(pooled_bank.domains[0]) if pooled_bank.domains else 0
    with torch.no_grad():
        pooled_scores = pooled_bank.score_domain_nelbo(pooled_domain, x)
        pooled_nelbo = float(pooled_scores.mean().item())

    return {
        "legacy_global_nelbo": legacy_nelbo,
        "hybrid_pooled_global_nelbo": pooled_nelbo,
    }
