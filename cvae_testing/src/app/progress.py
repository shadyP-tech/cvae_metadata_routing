from __future__ import annotations

import importlib

try:
    tqdm = getattr(importlib.import_module("tqdm"), "tqdm")
except Exception:  # pragma: no cover - fallback for environments without tqdm
    tqdm = None


class ProgressTracker:
    def __init__(self, total: int, desc: str) -> None:
        self._count = 0
        self._total = max(int(total), 1)
        self._bar = tqdm(total=self._total, desc=desc, unit="step") if tqdm is not None else None

    def advance(self, message: str) -> None:
        self._count += 1
        if self._bar is not None:
            self._bar.set_postfix_str(message)
            self._bar.update(1)
        else:
            print(f"[{self._count}/{self._total}] {message}")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
