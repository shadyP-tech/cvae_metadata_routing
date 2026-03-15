from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, Tuple


class DatasetAdapter(Protocol):
    def prepare_records(self, project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
        ...
