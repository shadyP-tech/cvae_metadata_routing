from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext


class BaseExperiment(ABC):
    @abstractmethod
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        raise NotImplementedError
