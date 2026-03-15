from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol


@dataclass(frozen=True)
class RunContext:
    output_root: Path
    run_root: Path
    reports_dir: Path
    embeddings_dir: Path
    checkpoints_dir: Path
    manifests_dir: Path
    plots_dir: Path
    latest_file: Path


class Dataset(Protocol):
    def get_splits(self) -> Dict[str, Any]:
        ...

    def get_metadata(self) -> Dict[str, Any]:
        ...


class ModelBundle(Protocol):
    def as_dict(self) -> Dict[str, Any]:
        ...


class Trainer(Protocol):
    def train(self) -> Dict[str, Any]:
        ...


class Evaluator(Protocol):
    def evaluate(self) -> Dict[str, Any]:
        ...


class Experiment(Protocol):
    def run(self) -> Dict[str, Any]:
        ...
