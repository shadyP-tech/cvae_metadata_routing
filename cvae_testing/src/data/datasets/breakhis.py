from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

from src.data.shared_split import image_level_split_indices, split_groups


MAG_REGEX = re.compile(r"(?<!\d)(40|100|200|400)(?:\s*[xX])?(?!\d)")


@dataclass(frozen=True)
class BreakHisRecord:
    sample_id: str
    image_path: str
    label: int
    label_name: str
    magnification: int
    domain_name: str
    patient_id: Optional[str]
    split: str = ""


def _label_from_path(path: Path) -> Tuple[int, str]:
    lowered = str(path).lower()
    if "benign" in lowered or re.search(r"(^|[_-])b([_-]|$)", path.name.lower()):
        return 0, "benign"
    if "malignant" in lowered or re.search(r"(^|[_-])m([_-]|$)", path.name.lower()):
        return 1, "malignant"
    raise ValueError(f"Could not infer class label from path: {path}")


def _magnification_from_path(path: Path) -> int:
    for text in [path.name, *path.parts[::-1]]:
        match = MAG_REGEX.search(text)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer magnification from path: {path}")


def _patient_id_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    # Common BreakHis naming often looks like:
    # SOB_B_A-14-22549G-40-001 or similar variants.
    match = re.match(r"(.+?)-(40|100|200|400)-\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback: remove trailing patch index if present.
    match = re.match(r"(.+?)-\d+$", stem)
    if match:
        return match.group(1)
    return None


def discover_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {e.lower() for e in extensions}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set])


def build_records(root: Path, extensions: Iterable[str]) -> List[BreakHisRecord]:
    records: List[BreakHisRecord] = []
    for image_path in discover_images(root, extensions):
        label, label_name = _label_from_path(image_path)
        magnification = _magnification_from_path(image_path)
        patient_id = _patient_id_from_filename(image_path)
        sample_id = image_path.stem
        records.append(
            BreakHisRecord(
                sample_id=sample_id,
                image_path=str(image_path),
                label=label,
                label_name=label_name,
                magnification=magnification,
                domain_name=f"{magnification}x",
                patient_id=patient_id,
            )
        )
    return records


def _assign_split_groupwise(
    records: List[BreakHisRecord],
    split: Dict[str, float],
    seed: int,
    require_patient_ids: bool,
) -> Tuple[List[BreakHisRecord], Dict[str, str]]:
    rng = random.Random(seed)
    limitations: Dict[str, str] = {}
    out: List[BreakHisRecord] = []

    with_patient = [r for r in records if r.patient_id is not None]
    without_patient = [r for r in records if r.patient_id is None]

    if require_patient_ids and without_patient:
        raise ValueError("Missing patient IDs while require_patient_ids=true.")

    # Primary path: split once globally at patient level to avoid cross-domain leakage.
    by_patient: Dict[str, List[BreakHisRecord]] = {}
    for rec in with_patient:
        by_patient.setdefault(rec.patient_id or "", []).append(rec)

    if by_patient:
        # Patient-level stratification by label keeps a coarse benign/malignant balance.
        patient_label: Dict[str, int] = {}
        for pid, recs in by_patient.items():
            patient_label[pid] = recs[0].label

        benign_ids = [pid for pid, lbl in patient_label.items() if lbl == 0]
        malignant_ids = [pid for pid, lbl in patient_label.items() if lbl == 1]

        benign_split = split_groups(benign_ids, split, rng)
        malignant_split = split_groups(malignant_ids, split, rng)

        patient_to_split: Dict[str, str] = {}
        for split_name in ["train", "val", "test"]:
            for pid in benign_split[split_name] + malignant_split[split_name]:
                patient_to_split[pid] = split_name

        for pid, recs in by_patient.items():
            split_name = patient_to_split[pid]
            for rec in recs:
                out.append(BreakHisRecord(**{**rec.__dict__, "split": split_name}))

    # Fallback path for records where patient ID cannot be parsed.
    if without_patient:
        limitations["global"] = "Some patient IDs unavailable; image-level split used for those samples."
        split_map = image_level_split_indices(len(without_patient), split=split, rng=rng)
        for split_name, split_idxs in split_map.items():
            for i in split_idxs:
                rec = without_patient[i]
                out.append(BreakHisRecord(**{**rec.__dict__, "split": split_name}))

    return out, limitations


def cap_samples_per_domain(
    records: List[BreakHisRecord],
    cap_per_domain: int,
    seed: int,
) -> List[BreakHisRecord]:
    rng = random.Random(seed)
    by_domain: Dict[int, List[BreakHisRecord]] = {}
    for rec in records:
        by_domain.setdefault(rec.magnification, []).append(rec)

    capped: List[BreakHisRecord] = []
    for _, domain_records in by_domain.items():
        by_split: Dict[str, List[BreakHisRecord]] = {"train": [], "val": [], "test": []}
        for rec in domain_records:
            by_split[rec.split].append(rec)

        domain_capped: List[BreakHisRecord] = []
        for split_name, split_records in by_split.items():
            split_cap = int(round(cap_per_domain * (0.70 if split_name == "train" else 0.15)))
            split_cap = min(split_cap, len(split_records))
            rng.shuffle(split_records)
            domain_capped.extend(split_records[:split_cap])

        if len(domain_capped) < cap_per_domain:
            remaining = [r for r in domain_records if r not in domain_capped]
            rng.shuffle(remaining)
            domain_capped.extend(remaining[: cap_per_domain - len(domain_capped)])

        capped.extend(domain_capped[:cap_per_domain])
    return capped


def leakage_report(records: List[BreakHisRecord]) -> Dict[str, object]:
    seen_paths = set()
    dup_paths = []
    for rec in records:
        if rec.image_path in seen_paths:
            dup_paths.append(rec.image_path)
        seen_paths.add(rec.image_path)

    patients_by_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for rec in records:
        if rec.patient_id is not None:
            patients_by_split[rec.split].add(rec.patient_id)

    overlaps = {
        "train_val": sorted(patients_by_split["train"].intersection(patients_by_split["val"])),
        "train_test": sorted(patients_by_split["train"].intersection(patients_by_split["test"])),
        "val_test": sorted(patients_by_split["val"].intersection(patients_by_split["test"])),
    }
    return {
        "duplicate_paths": dup_paths,
        "patient_overlap": overlaps,
    }


def write_manifest(records: List[BreakHisRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("sample_id,image_path,label,label_name,magnification,domain,patient_id,split\n")
        for rec in records:
            f.write(
                f"{rec.sample_id},{rec.image_path},{rec.label},{rec.label_name},{rec.magnification},"
                f"{rec.domain_name},{rec.patient_id or ''},{rec.split}\n"
            )


def prepare_breakhis_records(
    root: Path,
    extensions: Iterable[str],
    split: Dict[str, float],
    cap_per_domain: int,
    seed: int,
    require_patient_ids: bool,
) -> Tuple[List[BreakHisRecord], Dict[str, object]]:
    records = build_records(root, extensions)
    split_records, limitations = _assign_split_groupwise(
        records=records,
        split=split,
        seed=seed,
        require_patient_ids=require_patient_ids,
    )
    capped = cap_samples_per_domain(split_records, cap_per_domain=cap_per_domain, seed=seed)
    report = leakage_report(capped)
    report["limitations"] = limitations
    return capped, report
