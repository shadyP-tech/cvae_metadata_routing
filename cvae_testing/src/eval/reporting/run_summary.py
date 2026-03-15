from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def write_run_summary(reports_dir: Path, mode: str, payload: Dict[str, Any]) -> None:
    summary = {
        "mode": mode,
        "artifacts": payload,
    }

    with (reports_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (reports_dir / "run_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Run Summary\n\n")
        f.write(f"- mode: {mode}\n")

        if mode == "legacy_routed_cvae":
            metrics = payload.get("routing_metrics", {})
            f.write("\n## Routing Metrics\n\n")
            f.write(f"- hard_metadata_routing_nelbo: {float(metrics.get('hard_metadata_routing_nelbo', 0.0)):.4f}\n")
            f.write(f"- global_cvae_nelbo: {float(metrics.get('global_cvae_nelbo', 0.0)):.4f}\n")
            f.write(f"- oracle_expert_nelbo: {float(metrics.get('oracle_expert_nelbo', 0.0)):.4f}\n")
            f.write(f"- routing_selection_accuracy: {float(metrics.get('routing_selection_accuracy', 0.0)):.4f}\n")

        elif mode == "hybrid_ablation":
            baselines = payload.get("global_baselines", {})
            variants = payload.get("variants", {})
            f.write("\n## Hybrid Baselines\n\n")
            f.write(f"- legacy_global_nelbo: {float(baselines.get('legacy_global_nelbo', 0.0)):.4f}\n")
            f.write(f"- hybrid_pooled_global_nelbo: {float(baselines.get('hybrid_pooled_global_nelbo', 0.0)):.4f}\n")
            f.write(f"- n_variants: {len(variants)}\n")
