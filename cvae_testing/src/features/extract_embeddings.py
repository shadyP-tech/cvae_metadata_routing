from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.data.datasets.breakhis import BreakHisRecord


class RecordImageDataset(Dataset):
    def __init__(self, records: List[BreakHisRecord], image_size: int) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        return self.transform(image), rec


def _build_resnet18() -> torch.nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    records = [b[1] for b in batch]
    return imgs, records


def extract_and_cache_embeddings(
    records: List[BreakHisRecord],
    cache_dir: Path,
    image_size: int,
    batch_size: int,
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: cache_dir / f"{split}.pt" for split in ["train", "val", "test"]
    }
    if all(p.exists() for p in paths.values()):
        # Do not silently reuse stale empty caches.
        reusable = True
        expected_by_split = {
            split: sorted([r.image_path for r in records if r.split == split])
            for split in ["train", "val", "test"]
        }
        for split, p in paths.items():
            payload = torch.load(p, map_location="cpu")
            if int(payload["embeddings"].shape[0]) <= 0:
                reusable = False
                break
            cached_paths = sorted([m["image_path"] for m in payload["metadata"]])
            if cached_paths != expected_by_split[split]:
                reusable = False
                break
        if reusable:
            return paths

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = _build_resnet18().to(device)

    by_split = {
        split: [r for r in records if r.split == split] for split in ["train", "val", "test"]
    }

    for split, split_records in by_split.items():
        ds = RecordImageDataset(split_records, image_size=image_size)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

        all_embeddings = []
        all_meta = []
        with torch.no_grad():
            for x, batch_records in dl:
                x = x.to(device)
                feats = model(x).squeeze(-1).squeeze(-1).cpu()
                all_embeddings.append(feats)
                all_meta.extend([asdict(r) for r in batch_records])

        embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 512))
        payload = {"embeddings": embeddings, "metadata": all_meta}
        torch.save(payload, paths[split])

        with (cache_dir / f"{split}_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2)

    return paths


def validate_embedding_cache(cache_paths: Dict[str, Path], expected_dim: int = 512) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for split, path in cache_paths.items():
        payload = torch.load(path, map_location="cpu")
        embeddings = payload["embeddings"]
        metadata = payload["metadata"]

        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(f"{split}: expected [N,{expected_dim}] got {tuple(embeddings.shape)}")
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise ValueError(f"{split}: embeddings contain NaN or Inf")
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"{split}: embedding count and metadata count mismatch")

        report[split] = {
            "num_samples": embeddings.shape[0],
            "shape": tuple(embeddings.shape),
        }
    return report
