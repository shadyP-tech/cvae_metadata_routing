from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from src.app.bootstrap import resolve_config_path
from src.data.datasets.breakhis import prepare_breakhis_records
from src.data.datasets.camelyon17 import prepare_camelyon17_records


def _prepare_breakhis(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    root = resolve_config_path(project_root, str(cfg["data"]["root"]))
    return prepare_breakhis_records(
        root=root,
        extensions=cfg["data"]["image_extensions"],
        split=cfg["data"]["split"],
        cap_per_domain=int(cfg["data"]["max_samples_per_domain"]),
        seed=int(cfg["seed"]),
        require_patient_ids=bool(cfg["data"]["require_patient_ids"]),
    )


def _prepare_camelyon17(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    root = resolve_config_path(project_root, str(cfg["data"]["root"]))
    return prepare_camelyon17_records(
        root=root,
        extensions=cfg["data"]["image_extensions"],
        split=cfg["data"]["split"],
        cap_per_domain=int(cfg["data"]["max_samples_per_domain"]),
        seed=int(cfg["seed"]),
        require_patient_ids=bool(cfg["data"]["require_patient_ids"]),
        domain_field=str(cfg["data"].get("domain_field", "center")),
        metadata_file=str(cfg["data"].get("metadata_file", "metadata.csv")),
        use_metadata_split=bool(cfg["data"].get("use_metadata_split", False)),
    )


DATASET_REGISTRY = {
    "breakhis": _prepare_breakhis,
    "camelyon17": _prepare_camelyon17,
}


def prepare_dataset_records(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    dataset_type = str(cfg.get("data", {}).get("dataset_type", "breakhis")).lower()
    adapter = DATASET_REGISTRY.get(dataset_type)
    if adapter is None:
        raise ValueError(f"Unsupported data.dataset_type: {dataset_type}. Available: {sorted(DATASET_REGISTRY)}")
    return adapter(project_root, cfg)
