from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_reconstruction_vs_magnification(recon_matrix: Dict[str, Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    domains = sorted({int(d[:-1]) for row in recon_matrix.values() for d in row.keys()})

    plt.figure(figsize=(8, 5))
    for expert_name, scores in sorted(recon_matrix.items()):
        ys = [scores.get(f"{d}x", float("nan")) for d in domains]
        plt.plot(domains, ys, marker="o", label=expert_name)

    plt.xlabel("Magnification")
    plt.ylabel("Reconstruction MSE (lower is better)")
    plt.title("Expert Reconstruction Error vs Magnification")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
