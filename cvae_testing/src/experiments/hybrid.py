from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext
from src.eval.evaluators import compute_hybrid_matrices_and_routing, evaluate_downstream_utility, evaluate_global_baselines
from src.eval.reporting.hybrid_compact import write_hybrid_compact_reports
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.hybrid_ablation import train_hybrid_pooled_baseline, train_hybrid_variant


def _validate_hybrid_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hybrid_cfg = cfg.get("hybrid", {})
    variants = [str(v) for v in hybrid_cfg.get("variants", ["A", "B", "C"])]
    budgets = [float(v) for v in hybrid_cfg.get("synthetic_budgets", [1.0, 0.5])]

    allowed_variants = {"A", "B", "C"}
    if not variants:
        raise ValueError("hybrid.variants must not be empty")
    unknown = sorted(set(variants) - allowed_variants)
    if unknown:
        raise ValueError(f"hybrid.variants contains unsupported values: {unknown}")

    if any(b <= 0 for b in budgets):
        raise ValueError("hybrid.synthetic_budgets must contain only positive values")

    budget_set = {round(b, 6) for b in budgets}
    if 1.0 not in budget_set or 0.5 not in budget_set:
        raise ValueError(
            "hybrid.synthetic_budgets must include the locked protocol budgets 1.0 and 0.5"
        )

    return {
        "variants": variants,
        "budgets": budgets,
        "projection_dim": int(hybrid_cfg.get("projection_dim", cfg["features"]["embedding_dim"])),
        "head_hidden_dim": int(hybrid_cfg.get("head_hidden_dim", 256)),
        "cvae_hidden_dim": int(hybrid_cfg.get("cvae_hidden_dim", cfg["model"]["hidden_dim"])),
    }


class HybridAblationExperiment(BaseExperiment):
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        variants = [str(v) for v in cfg.get("hybrid", {}).get("variants", ["A", "B", "C"])]
        # Base runner contributes 5 steps; hybrid contributes 5 + 3 per variant.
        return 10 + (3 * len(variants))

    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        hybrid_validated = _validate_hybrid_config(cfg)
        progress.advance("hybrid config validated")
        domains = [int(d) for d in cfg["data"]["magnifications"]]
        projection_dim = hybrid_validated["projection_dim"]
        head_hidden_dim = hybrid_validated["head_hidden_dim"]
        cvae_hidden_dim = hybrid_validated["cvae_hidden_dim"]
        variants = hybrid_validated["variants"]
        budgets = hybrid_validated["budgets"]

        pooled_info = train_hybrid_pooled_baseline(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            domains=domains,
            projection_dim=projection_dim,
            head_hidden_dim=head_hidden_dim,
            cvae_hidden_dim=cvae_hidden_dim,
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            seed=int(cfg["seed"]),
            model_name="hybrid_pooled_baseline",
            resume_from=(resume_checkpoints_dir / "hybrid_pooled_baseline.pt") if resume_checkpoints_dir is not None else None,
        )
        progress.advance("hybrid pooled baseline trained")

        hybrid_results: Dict[str, object] = {
            "mode": "hybrid_ablation",
            "variants": {},
            "legacy_global_checkpoint": str(global_ckpt),
            "hybrid_pooled_checkpoint": str(pooled_info["checkpoint"]),
            "synthetic_budgets": budgets,
        }

        baseline_metrics = evaluate_global_baselines(
            test_cache=cache_paths["test"],
            legacy_global_checkpoint=Path(global_ckpt),
            legacy_hidden_dim=int(cfg["model"]["hidden_dim"]),
            legacy_latent_dim=int(cfg["model"]["latent_dim"]),
            pooled_checkpoint=Path(pooled_info["checkpoint"]),
        )
        hybrid_results["global_baselines"] = baseline_metrics
        progress.advance("global baseline metrics computed")

        for variant in variants:
            model_name = f"hybrid_variant_{variant}"
            info = train_hybrid_variant(
                train_cache=cache_paths["train"],
                val_cache=cache_paths["val"],
                out_dir=run_ctx.checkpoints_dir,
                domains=domains,
                projection_dim=projection_dim,
                head_hidden_dim=head_hidden_dim,
                cvae_hidden_dim=cvae_hidden_dim,
                latent_dim=int(cfg["model"]["latent_dim"]),
                lr=float(cfg["training"]["learning_rate"]),
                epochs=int(cfg["training"]["epochs"]),
                patience=int(cfg["training"]["patience"]),
                batch_size=int(cfg["training"]["batch_size"]),
                seed=int(cfg["seed"]),
                variant=variant,
                model_name=model_name,
                resume_from=(resume_checkpoints_dir / f"{model_name}.pt") if resume_checkpoints_dir is not None else None,
            )
            progress.advance(f"variant {variant} trained")

            matrix_and_routing = compute_hybrid_matrices_and_routing(
                test_cache=cache_paths["test"],
                hybrid_checkpoint=Path(info["checkpoint"]),
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                temperature=float(cfg["routing"]["temperature"]),
                seed=int(cfg["seed"]),
                similarity_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            )
            progress.advance(f"variant {variant} routing and matrix evaluated")

            downstream = evaluate_downstream_utility(
                train_cache=cache_paths["train"],
                test_cache=cache_paths["test"],
                hybrid_checkpoint=Path(info["checkpoint"]),
                pooled_checkpoint=Path(pooled_info["checkpoint"]),
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                temperature=float(cfg["routing"]["temperature"]),
                seed=int(cfg["seed"]),
                budget_multipliers=budgets,
            )
            progress.advance(f"variant {variant} downstream utility evaluated")

            hybrid_results["variants"][variant] = {
                "checkpoint": info["checkpoint"],
                "history": info["history"],
                "compatibility_matrix": matrix_and_routing["compatibility_matrix"],
                "routing_matrix": matrix_and_routing["routing_matrix"],
                "routing_statistics": matrix_and_routing["routing_statistics"],
                "routing_metrics": matrix_and_routing["routing_metrics"],
                "downstream_utility": downstream,
                "global_baselines": baseline_metrics,
            }

            with (run_ctx.reports_dir / f"hybrid_variant_{variant}_report.json").open("w", encoding="utf-8") as f:
                json.dump(hybrid_results["variants"][variant], f, indent=2)

        with (run_ctx.reports_dir / "hybrid_summary.json").open("w", encoding="utf-8") as f:
            json.dump(hybrid_results, f, indent=2)
        progress.advance("hybrid summary written")

        write_hybrid_compact_reports(run_ctx.reports_dir, hybrid_results)
        progress.advance("hybrid compact reports written")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="hybrid_ablation",
            payload={
                "global_baselines": hybrid_results.get("global_baselines", {}),
                "variants": hybrid_results.get("variants", {}),
                "summary_artifact": "hybrid_summary.json",
                "compact_artifact": "hybrid_variant_comparison.csv",
            },
        )

        progress.close()

        print("Hybrid ablation experiment complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
        print("Latest run pointer:", run_ctx.latest_file)
