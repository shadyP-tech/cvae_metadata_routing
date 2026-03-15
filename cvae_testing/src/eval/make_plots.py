from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_latest_run(project_root: Path) -> Path:
    outputs_root = project_root / "outputs"
    if not outputs_root.exists():
        raise FileNotFoundError("No outputs directory found")

    latest_candidates = sorted(outputs_root.glob("*/*/latest.txt"))
    if not latest_candidates:
        raise FileNotFoundError("No latest.txt run pointers found")

    latest_ptr = max(latest_candidates, key=lambda p: p.stat().st_mtime)
    run_id = latest_ptr.read_text(encoding="utf-8").strip()
    return latest_ptr.parent / run_id


def _plot_bar(values: Dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values.keys())
    nums = [values[k] for k in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, nums)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    # Annotate bar values for quick readability.
    for bar, val in zip(bars, nums):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_matrix(
    matrix: Dict[str, Dict[str, float]],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = sorted(matrix.keys(), key=lambda s: int(s.replace("expert_", "").replace("x", "")))
    col_set = set()
    for row in matrix.values():
        col_set.update(row.keys())
    col_labels = sorted(col_set, key=lambda s: int(s.replace("x", "")))

    data: List[List[float]] = []
    for r in row_labels:
        data.append([float(matrix[r].get(c, float("nan"))) for c in col_labels])

    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.xlabel("Domain")
    plt.ylabel("Expert")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i][j]
            if val == val:
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_relative_improvement(metrics: Dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_methods = {
        "hard": metrics["hard_metadata_routing_nelbo"],
        "soft": metrics["soft_metadata_routing_nelbo"],
        "oracle": metrics["oracle_expert_nelbo"],
        "global": metrics["global_cvae_nelbo"],
    }
    random_baseline = metrics["random_expert_nelbo"]
    global_baseline = metrics["global_cvae_nelbo"]

    def improvement(base: float, val: float) -> float:
        return (base - val) / base * 100.0

    labels = list(target_methods.keys())
    vs_random = [improvement(random_baseline, target_methods[k]) for k in labels]
    vs_global = [improvement(global_baseline, target_methods[k]) for k in labels]

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], vs_random, width=width, label="vs random")
    plt.bar([i + width / 2 for i in x], vs_global, width=width, label="vs global")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Relative Improvement (%)")
    plt.title("NELBO Relative Improvement (Lower Is Better)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_split_sizes(cache_report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    sizes = [int(cache_report[s]["num_samples"]) for s in splits]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(splits, sizes)
    plt.ylabel("Num Samples")
    plt.title("Split Sizes")
    plt.grid(axis="y", alpha=0.25)
    for b, s in zip(bars, sizes):
        plt.text(b.get_x() + b.get_width() / 2, s, str(s), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_patient_overlap(leakage_report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlaps = leakage_report.get("patient_overlap", {})
    keys = ["train_val", "train_test", "val_test"]
    counts = [len(overlaps.get(k, [])) for k in keys]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(keys, counts)
    plt.ylabel("Num Overlapping Patients")
    plt.title("Patient Overlap Across Splits")
    plt.grid(axis="y", alpha=0.25)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _flatten_confusion(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for true_domain, preds in confusion.items():
        out[true_domain] = {k: float(v) for k, v in preds.items()}
    return out


def generate_plots_from_reports(reports_dir: Path, out_dir: Path) -> Path:
    routing = _load_json(reports_dir / "routing_results.json")
    expert = _load_json(reports_dir / "expert_matrix.json")
    leakage = _load_json(reports_dir / "leakage_report.json")
    cache = _load_json(reports_dir / "cache_report.json")

    metrics = routing["metrics"]

    _plot_bar(
        {
            "hard": metrics["hard_metadata_routing_nelbo"],
            "soft": metrics["soft_metadata_routing_nelbo"],
            "oracle": metrics["oracle_expert_nelbo"],
            "global": metrics["global_cvae_nelbo"],
            "random": metrics["random_expert_nelbo"],
            "uniform": metrics["uniform_sampling_nelbo"],
            "equal": metrics["equal_weight_scoring_nelbo"],
        },
        title="Method Comparison by NELBO",
        ylabel="NELBO (lower is better)",
        out_path=out_dir / "baseline_nelbo_bar.png",
    )

    _plot_bar(
        {
            "hard": metrics["hard_metadata_routing_recon"],
            "soft": metrics["soft_metadata_routing_recon"],
            "oracle": metrics["oracle_expert_recon"],
            "global": metrics["global_cvae_recon"],
            "random": metrics["random_expert_recon"],
            "uniform": metrics["uniform_sampling_recon"],
            "equal": metrics["equal_weight_scoring_recon"],
        },
        title="Method Comparison by Reconstruction",
        ylabel="Reconstruction Score (lower is better)",
        out_path=out_dir / "baseline_reconstruction_bar.png",
    )

    _plot_matrix(
        _flatten_confusion(routing["routing"].get("confusion_matrix", {})),
        title="Routing Confusion Matrix (True Domain vs Routed Expert)",
        out_path=out_dir / "routing_confusion_heatmap.png",
        cmap="Blues",
    )

    _plot_matrix(
        expert["reconstruction_matrix"],
        title="Expert-Domain Reconstruction Matrix",
        out_path=out_dir / "expert_domain_reconstruction_heatmap.png",
    )

    mean_matrix = {
        expert_name: {
            domain_name: float(stats["mean"])
            for domain_name, stats in domain_stats.items()
        }
        for expert_name, domain_stats in expert["confidence"].items()
    }
    var_matrix = {
        expert_name: {
            domain_name: float(stats["var"])
            for domain_name, stats in domain_stats.items()
        }
        for expert_name, domain_stats in expert["confidence"].items()
    }

    _plot_matrix(
        mean_matrix,
        title="Expert Confidence Mean ELBO",
        out_path=out_dir / "expert_confidence_mean_heatmap.png",
        cmap="magma",
    )
    _plot_matrix(
        var_matrix,
        title="Expert Confidence ELBO Variance",
        out_path=out_dir / "expert_confidence_variance_heatmap.png",
        cmap="magma",
    )

    _plot_relative_improvement(metrics, out_dir / "relative_improvement_nelbo.png")
    _plot_split_sizes(cache, out_dir / "split_sizes.png")
    _plot_patient_overlap(leakage, out_dir / "patient_overlap.png")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots from routing/expert reports.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory under outputs/<dataset>/<experiment>/<run_id>",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory containing cache_report.json, leakage_report.json, expert_matrix.json, routing_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated plots",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir
    if run_dir is None and args.reports_dir is None:
        try:
            run_dir = _resolve_latest_run(project_root)
        except FileNotFoundError:
            run_dir = None

    if run_dir is not None:
        reports_dir = run_dir / "reports"
        out_dir = args.out_dir or (run_dir / "plots")
    else:
        reports_dir = args.reports_dir or (project_root / "artifacts" / "reports")
        out_dir = args.out_dir or Path("results/plots")

    generated_dir = generate_plots_from_reports(reports_dir=reports_dir, out_dir=out_dir)
    print("Saved plots to", generated_dir)


if __name__ == "__main__":
    main()
