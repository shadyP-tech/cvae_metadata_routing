from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.engine.contracts import RunContext
from src.eval.evaluators import compute_expert_domain_matrix
from src.eval.evaluators.latent_compatibility import (
    compute_distance_utility_correlation,
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    compute_metric_utility_correlation,
    distance_to_similarity,
    evaluate_routing_alignment,
    load_embeddings_with_domains,
    matrix_to_domain_dict,
    maybe_project_latent_2d,
    plot_composite_figure,
    plot_distance_vs_utility,
    plot_latent_map,
    plot_matrix_heatmap,
    verify_similarity_matrix,
)
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.train_experts import train_domain_experts


def _validate_latent_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("latent_compatibility", {})
    allowed_metrics = {"centroid", "wasserstein", "gaussian_kl"}
    metrics = [str(m) for m in block.get("metrics", ["centroid", "wasserstein", "gaussian_kl"])]
    if not metrics:
        raise ValueError("latent_compatibility.metrics must not be empty")
    unknown = sorted(set(metrics) - allowed_metrics)
    if unknown:
        raise ValueError(f"latent_compatibility.metrics contains unsupported values: {unknown}")

    similarity_transform = str(block.get("similarity_transform", "exp_neg")).strip()
    if similarity_transform != "exp_neg":
        raise ValueError("latent_compatibility.similarity_transform must be 'exp_neg'")

    splits = [str(s) for s in block.get("splits", ["test"])]
    allowed_splits = {"train", "val", "test"}
    if not splits or any(s not in allowed_splits for s in splits):
        raise ValueError(f"latent_compatibility.splits must be in {sorted(allowed_splits)}")

    verification = block.get("verification", {})
    wasserstein = block.get("wasserstein", {})
    similarity = block.get("similarity", {})
    empirical = block.get("empirical_utility", {})

    out = {
        "metrics": metrics,
        "splits": splits,
        "similarity_transform": similarity_transform,
        "min_samples_per_domain": int(block.get("min_samples_per_domain", 50)),
        "covariance_regularization_lambda": float(block.get("covariance_regularization_lambda", 1e-4)),
        "evaluation_metrics": [
            str(m) for m in block.get("evaluation_metrics", ["top1_agreement", "mean_rank", "spearman_with_utility"])
        ],
        "verification": {
            "symmetry_atol": float(verification.get("symmetry_atol", 1e-6)),
            "symmetry_rtol": float(verification.get("symmetry_rtol", 1e-5)),
            "diag_opt_tol": float(verification.get("diag_opt_tol", 1e-6)),
        },
        "wasserstein": {
            "eigenvalue_floor": float(wasserstein.get("eigenvalue_floor", 1e-10)),
        },
        "similarity": {
            "scale_floor": float(similarity.get("scale_floor", 1e-8)),
            "scale_policy": str(similarity.get("scale_policy", block.get("scale_policy", "median_off_diagonal"))),
        },
        "umap": {
            "max_points": int(block.get("umap", {}).get("max_points", 5000)),
        },
        "composite_metric": str(block.get("composite_metric", "wasserstein")),
        "empirical_utility": {
            "enabled": bool(empirical.get("enabled", True)),
        },
    }

    if out["similarity"]["scale_policy"] != "median_off_diagonal":
        raise ValueError("latent_compatibility similarity scale policy must be 'median_off_diagonal'")
    if out["min_samples_per_domain"] <= 0:
        raise ValueError("latent_compatibility.min_samples_per_domain must be > 0")
    if out["covariance_regularization_lambda"] <= 0:
        raise ValueError("latent_compatibility.covariance_regularization_lambda must be > 0")
    if out["wasserstein"]["eigenvalue_floor"] <= 0:
        raise ValueError("latent_compatibility.wasserstein.eigenvalue_floor must be > 0")
    if out["similarity"]["scale_floor"] <= 0:
        raise ValueError("latent_compatibility.similarity.scale_floor must be > 0")
    if out["umap"]["max_points"] <= 0:
        raise ValueError("latent_compatibility.umap.max_points must be > 0")

    return out


def _to_jsonable_gaussian_stats(
    domain_order: List[int],
    stats: Dict[int, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for d in domain_order:
        ds = stats[d]
        payload[f"{d}x"] = {
            "n_samples": int(ds.n_samples),
            "used_diagonal_covariance": bool(ds.used_diagonal_covariance),
            "mean": ds.mean.astype(float).tolist(),
            "covariance_diagonal": np.diag(ds.covariance).astype(float).tolist(),
        }
    return payload


def _utility_matrix_from_expert_matrix(domain_order: List[int], expert_matrix_report: Dict[str, Any]) -> np.ndarray:
    confidence = expert_matrix_report.get("confidence", {})
    matrix = np.zeros((len(domain_order), len(domain_order)), dtype=np.float64)
    for i, query_domain in enumerate(domain_order):
        for j, expert_domain in enumerate(domain_order):
            expert_key = f"{expert_domain}x"
            query_key = f"{query_domain}x"
            mean_nelbo = float(confidence.get(expert_key, {}).get(query_key, {}).get("mean", 0.0))
            matrix[i, j] = -mean_nelbo
    return matrix


def _write_metric_correlation_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "corr_with_utility", "corr_distance_with_utility"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class LatentCompatibilityExperiment(BaseExperiment):
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        latent_cfg = _validate_latent_config(cfg)
        # Base runner contributes 5 stages before experiment.run() is called.
        # This experiment contributes 7 stages, plus 2 when empirical utility is enabled.
        steps = 12
        if bool(latent_cfg["empirical_utility"]["enabled"]):
            steps += 2
        return steps

    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        _ = global_ckpt
        latent_cfg = _validate_latent_config(cfg)
        progress.advance("latent compatibility config validated")

        embeddings, sample_domains, _ = load_embeddings_with_domains(
            cache_paths=cache_paths,
            splits=latent_cfg["splits"],
        )
        progress.advance("latent embeddings loaded")

        domain_order, gaussian_stats, gaussian_warnings = compute_domain_gaussian_stats(
            embeddings=embeddings,
            domains=sample_domains,
            covariance_regularization_lambda=float(latent_cfg["covariance_regularization_lambda"]),
            min_samples_per_domain=int(latent_cfg["min_samples_per_domain"]),
        )
        progress.advance("domain gaussian summaries computed")

        distance_mats = compute_distance_matrices(
            domain_order=domain_order,
            stats=gaussian_stats,
            eigenvalue_floor=float(latent_cfg["wasserstein"]["eigenvalue_floor"]),
        )

        distance_name_map = {
            "centroid": "distance_centroid.npy",
            "wasserstein": "distance_wasserstein.npy",
            "gaussian_kl": "distance_kl_sym.npy",
        }
        for metric_name, distance in distance_mats.items():
            np.save(run_ctx.reports_dir / distance_name_map[metric_name], distance)
        progress.advance("distance matrices computed")

        similarity_name_map = {
            "centroid": "similarity_centroid.npy",
            "wasserstein": "similarity_wasserstein.npy",
            "gaussian_kl": "similarity_kl_sym.npy",
        }

        similarity_mats: Dict[str, np.ndarray] = {}
        scale_by_metric: Dict[str, float] = {}
        verification_by_metric: Dict[str, Any] = {}
        routing_by_metric: Dict[str, Any] = {}

        for metric_name in latent_cfg["metrics"]:
            sim, scale = distance_to_similarity(
                distance=distance_mats[metric_name],
                scale_floor=float(latent_cfg["similarity"]["scale_floor"]),
            )
            similarity_mats[metric_name] = sim
            scale_by_metric[metric_name] = float(scale)
            np.save(run_ctx.reports_dir / similarity_name_map[metric_name], sim)

            verification_by_metric[metric_name] = verify_similarity_matrix(
                matrix=sim,
                atol=float(latent_cfg["verification"]["symmetry_atol"]),
                rtol=float(latent_cfg["verification"]["symmetry_rtol"]),
                diag_opt_tol=float(latent_cfg["verification"]["diag_opt_tol"]),
                symmetric_expected=True,
            )

            routing_by_metric[metric_name] = evaluate_routing_alignment(
                domain_order=domain_order,
                similarity_matrix=sim,
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                similarity_lookup_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            )
        progress.advance("similarity matrices, verification, and routing computed")

        utility_matrix: np.ndarray | None = None
        utility_report: Dict[str, Any] | None = None
        if bool(latent_cfg["empirical_utility"]["enabled"]):
            experts = train_domain_experts(
                train_cache=cache_paths["train"],
                val_cache=cache_paths["val"],
                out_dir=run_ctx.checkpoints_dir,
                domains=[int(d) for d in cfg["data"]["magnifications"]],
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                latent_dim=int(cfg["model"]["latent_dim"]),
                lr=float(cfg["training"]["learning_rate"]),
                epochs=int(cfg["training"]["epochs"]),
                patience=int(cfg["training"]["patience"]),
                batch_size=int(cfg["training"]["batch_size"]),
                resume_from_dir=resume_checkpoints_dir,
            )
            progress.advance("empirical utility experts trained")

            utility_report = compute_expert_domain_matrix(
                test_cache=cache_paths["test"],
                expert_checkpoints=experts,
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                latent_dim=int(cfg["model"]["latent_dim"]),
            )
            utility_matrix = _utility_matrix_from_expert_matrix(domain_order, utility_report)
            with (run_ctx.reports_dir / "expert_utility_matrix.json").open("w", encoding="utf-8") as f:
                json.dump(utility_report, f, indent=2)
            progress.advance("empirical utility matrix computed")

        metric_corr_rows: List[Dict[str, Any]] = []
        for metric_name in latent_cfg["metrics"]:
            corr = 0.0
            corr_dist = 0.0
            if utility_matrix is not None:
                corr = compute_metric_utility_correlation(similarity_mats[metric_name], utility_matrix)
                corr_dist = compute_distance_utility_correlation(
                    distance_matrix=distance_mats[metric_name],
                    utility_matrix=utility_matrix,
                    off_diagonal_only=True,
                )
            routing_by_metric[metric_name]["spearman_with_utility"] = float(corr)
            routing_by_metric[metric_name]["spearman_distance_with_utility"] = float(corr_dist)
            metric_corr_rows.append(
                {
                    "metric": metric_name,
                    "corr_with_utility": f"{corr:.6f}",
                    "corr_distance_with_utility": f"{corr_dist:.6f}",
                }
            )

        _write_metric_correlation_csv(run_ctx.reports_dir / "metric_correlation_table.csv", metric_corr_rows)

        for metric_name in latent_cfg["metrics"]:
            fname = {
                "centroid": "compatibility_heatmap_centroid.png",
                "wasserstein": "compatibility_heatmap_wasserstein.png",
                "gaussian_kl": "compatibility_heatmap_kl.png",
            }[metric_name]
            plot_matrix_heatmap(
                matrix=similarity_mats[metric_name],
                domain_order=domain_order,
                title=f"Latent Compatibility ({metric_name})",
                out_path=run_ctx.plots_dir / fname,
            )

        projection, sample_idxs, reducer_info = maybe_project_latent_2d(
            embeddings=embeddings,
            seed=int(cfg["seed"]),
            max_points=int(latent_cfg["umap"]["max_points"]),
        )
        sampled_domains = sample_domains[sample_idxs] if sample_idxs.size > 0 else np.empty((0,), dtype=np.int64)

        plot_latent_map(
            coords=projection,
            sample_domains=sampled_domains,
            domain_order=domain_order,
            out_path=run_ctx.plots_dir / "latent_umap.png",
            title=f"Latent Domain Map ({reducer_info.get('method', 'unknown')})",
        )

        if utility_matrix is not None:
            plot_matrix_heatmap(
                matrix=utility_matrix,
                domain_order=domain_order,
                title="Empirical Expert Utility (negative NELBO)",
                out_path=run_ctx.plots_dir / "expert_utility_heatmap.png",
                cmap="magma",
            )

            for metric_name in latent_cfg["metrics"]:
                scatter_name = {
                    "centroid": "distance_vs_utility_centroid.png",
                    "wasserstein": "distance_vs_utility_wasserstein.png",
                    "gaussian_kl": "distance_vs_utility_kl.png",
                }[metric_name]
                plot_distance_vs_utility(
                    distance_matrix=distance_mats[metric_name],
                    utility_matrix=utility_matrix,
                    domain_order=domain_order,
                    out_path=run_ctx.plots_dir / scatter_name,
                    title=f"Distance vs Utility ({metric_name})",
                    add_regression=True,
                    color_by_query=True,
                )

        composite_metric = str(latent_cfg["composite_metric"])
        if composite_metric not in similarity_mats:
            composite_metric = latent_cfg["metrics"][0]
        plot_composite_figure(
            coords=projection,
            sample_domains=sampled_domains,
            domain_order=domain_order,
            compatibility_matrix=similarity_mats[composite_metric],
            utility_matrix=utility_matrix,
            distance_matrix=distance_mats[composite_metric] if utility_matrix is not None else None,
            out_path=run_ctx.plots_dir / "latent_compatibility_composite.png",
        )
        progress.advance("plots generated")

        gaussian_json = {
            "domain_order": [f"{d}x" for d in domain_order],
            "covariance_regularization_lambda": float(latent_cfg["covariance_regularization_lambda"]),
            "min_samples_per_domain": int(latent_cfg["min_samples_per_domain"]),
            "warnings": gaussian_warnings,
            "domains": _to_jsonable_gaussian_stats(domain_order, gaussian_stats),
        }
        with (run_ctx.reports_dir / "latent_gaussian_stats.json").open("w", encoding="utf-8") as f:
            json.dump(gaussian_json, f, indent=2)

        routing_payload = {
            "domain_order": [f"{d}x" for d in domain_order],
            "scales": scale_by_metric,
            "verification": verification_by_metric,
            "routing": routing_by_metric,
            "reducer": reducer_info,
            "distance_matrices": {
                k: matrix_to_domain_dict(domain_order, v) for k, v in distance_mats.items()
            },
            "similarity_matrices": {
                k: matrix_to_domain_dict(domain_order, v) for k, v in similarity_mats.items()
            },
        }
        with (run_ctx.reports_dir / "routing_agreement.json").open("w", encoding="utf-8") as f:
            json.dump(routing_payload, f, indent=2)

        with (run_ctx.reports_dir / "report.md").open("w", encoding="utf-8") as f:
            f.write("# Latent Compatibility Report\n\n")
            f.write(f"- domains: {', '.join(f'{d}x' for d in domain_order)}\n")
            f.write(f"- splits: {', '.join(latent_cfg['splits'])}\n")
            f.write(f"- covariance_regularization_lambda: {float(latent_cfg['covariance_regularization_lambda']):.1e}\n")
            f.write(f"- eigenvalue_floor: {float(latent_cfg['wasserstein']['eigenvalue_floor']):.1e}\n")
            f.write(f"- scale_floor: {float(latent_cfg['similarity']['scale_floor']):.1e}\n")
            f.write("\n## Routing Agreement\n\n")
            for metric_name in latent_cfg["metrics"]:
                r = routing_by_metric[metric_name]
                f.write(f"- {metric_name}: top1={float(r['top1_agreement']):.4f}, ")
                f.write(f"mean_rank={float(r['mean_rank']):.4f}, ")
                f.write(f"spearman_with_utility={float(r.get('spearman_with_utility', 0.0)):.4f}, ")
                f.write(f"spearman_distance_with_utility={float(r.get('spearman_distance_with_utility', 0.0)):.4f}\n")

            f.write("\n## Verification\n\n")
            for metric_name in latent_cfg["metrics"]:
                v = verification_by_metric[metric_name]
                f.write(
                    f"- {metric_name}: finite_ok={v['finite_ok']}, symmetry_ok={v['symmetry_ok']}, diagonal_optimality_ok={v['diagonal_optimality_ok']}\n"
                )

            if gaussian_warnings:
                f.write("\n## Warnings\n\n")
                for warning in gaussian_warnings:
                    f.write(f"- {warning}\n")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="latent_compatibility",
            payload={
                "routing_artifact": "routing_agreement.json",
                "gaussian_stats_artifact": "latent_gaussian_stats.json",
                "correlation_artifact": "metric_correlation_table.csv",
                "report_artifact": "report.md",
            },
        )
        progress.advance("reports written")
        progress.close()

        print("Latent compatibility experiment complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
        print("Plots:", run_ctx.plots_dir)
        print("Latest run pointer:", run_ctx.latest_file)
