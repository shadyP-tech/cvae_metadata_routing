from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict


def _mean(xs: list[float]) -> float:
    clean = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(clean) / len(clean) if clean else 0.0


def write_hybrid_compact_reports(reports_dir: Path, hybrid_results: Dict[str, object]) -> None:
    rows = []
    variants = hybrid_results.get("variants", {})
    global_baselines = hybrid_results.get("global_baselines", {})

    for variant_name, payload in variants.items():
        routing_stats = payload.get("routing_statistics", {})
        routing_metrics = payload.get("routing_metrics", {})
        downstream = payload.get("downstream_utility", {})

        for budget_key, by_domain in downstream.items():
            auroc_real = []
            auroc_random = []
            auroc_pooled = []
            auroc_routed = []
            bacc_real = []
            bacc_random = []
            bacc_pooled = []
            bacc_routed = []

            for _, domain_payload in by_domain.items():
                m = domain_payload.get("metrics", {})
                auroc_real.append(float(m.get("real_only", {}).get("auroc", 0.0)))
                auroc_random.append(float(m.get("real_plus_random_synthetic", {}).get("auroc", 0.0)))
                auroc_pooled.append(float(m.get("real_plus_pooled_synthetic", {}).get("auroc", 0.0)))
                auroc_routed.append(float(m.get("real_plus_routed_synthetic", {}).get("auroc", 0.0)))

                bacc_real.append(float(m.get("real_only", {}).get("balanced_accuracy", 0.0)))
                bacc_random.append(float(m.get("real_plus_random_synthetic", {}).get("balanced_accuracy", 0.0)))
                bacc_pooled.append(float(m.get("real_plus_pooled_synthetic", {}).get("balanced_accuracy", 0.0)))
                bacc_routed.append(float(m.get("real_plus_routed_synthetic", {}).get("balanced_accuracy", 0.0)))

            row = {
                "variant": str(variant_name),
                "budget": str(budget_key),
                "metadata_nelbo": float(routing_metrics.get("metadata_routing_nelbo", 0.0)),
                "oracle_nelbo": float(routing_metrics.get("oracle_routing_nelbo", 0.0)),
                "metadata_to_oracle_gap": float(routing_metrics.get("metadata_to_oracle_gap", 0.0)),
                "spearman_similarity_vs_neg_nelbo": float(
                    routing_stats.get("spearman_similarity_vs_neg_nelbo", 0.0)
                ),
                "top1_agreement_with_best_expert": float(
                    routing_stats.get("top1_agreement_with_best_expert", 0.0)
                ),
                "mean_rank_metadata_selected": float(
                    routing_stats.get("mean_rank_of_metadata_selected_expert", 0.0)
                ),
                "legacy_global_nelbo": float(global_baselines.get("legacy_global_nelbo", 0.0)),
                "hybrid_pooled_global_nelbo": float(global_baselines.get("hybrid_pooled_global_nelbo", 0.0)),
                "auroc_real_only": _mean(auroc_real),
                "auroc_real_plus_random": _mean(auroc_random),
                "auroc_real_plus_pooled": _mean(auroc_pooled),
                "auroc_real_plus_routed": _mean(auroc_routed),
                "auroc_delta_routed_vs_real": _mean(auroc_routed) - _mean(auroc_real),
                "auroc_delta_routed_vs_random": _mean(auroc_routed) - _mean(auroc_random),
                "auroc_delta_routed_vs_pooled": _mean(auroc_routed) - _mean(auroc_pooled),
                "bacc_real_only": _mean(bacc_real),
                "bacc_real_plus_random": _mean(bacc_random),
                "bacc_real_plus_pooled": _mean(bacc_pooled),
                "bacc_real_plus_routed": _mean(bacc_routed),
                "bacc_delta_routed_vs_real": _mean(bacc_routed) - _mean(bacc_real),
                "bacc_delta_routed_vs_random": _mean(bacc_routed) - _mean(bacc_random),
                "bacc_delta_routed_vs_pooled": _mean(bacc_routed) - _mean(bacc_pooled),
            }
            rows.append(row)

    if not rows:
        return

    csv_path = reports_dir / "hybrid_variant_comparison.csv"
    md_path = reports_dir / "hybrid_variant_comparison.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Hybrid Ablation Compact Comparison\\n\\n")
        f.write("| variant | budget | metadata_nelbo | oracle_nelbo | metadata-oracle | spearman | top1 | mean_rank | auroc routed-real | auroc routed-random | auroc routed-pooled | bacc routed-real | bacc routed-random | bacc routed-pooled |\\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\\n")
        for r in rows:
            f.write(
                f"| {r['variant']} | {r['budget']} | {r['metadata_nelbo']:.2f} | {r['oracle_nelbo']:.2f} | "
                f"{r['metadata_to_oracle_gap']:.2f} | {r['spearman_similarity_vs_neg_nelbo']:.3f} | "
                f"{r['top1_agreement_with_best_expert']:.3f} | {r['mean_rank_metadata_selected']:.2f} | "
                f"{r['auroc_delta_routed_vs_real']:.4f} | {r['auroc_delta_routed_vs_random']:.4f} | "
                f"{r['auroc_delta_routed_vs_pooled']:.4f} | {r['bacc_delta_routed_vs_real']:.4f} | "
                f"{r['bacc_delta_routed_vs_random']:.4f} | {r['bacc_delta_routed_vs_pooled']:.4f} |\\n"
            )

        f.write("\\n")
        f.write("## Global Baselines\\n\\n")
        f.write(
            f"- legacy_global_nelbo: {float(global_baselines.get('legacy_global_nelbo', 0.0)):.2f}\\n"
        )
        f.write(
            f"- hybrid_pooled_global_nelbo: {float(global_baselines.get('hybrid_pooled_global_nelbo', 0.0)):.2f}\\n"
        )
