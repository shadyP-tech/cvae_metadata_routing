from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext


class TrackingClient:
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        raise NotImplementedError

    def log_artifact(self, path: Path, artifact_name: str, artifact_type: str) -> None:
        raise NotImplementedError

    def finish(self, status: str) -> None:
        raise NotImplementedError


class NullTrackingClient(TrackingClient):
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        return

    def log_artifact(self, path: Path, artifact_name: str, artifact_type: str) -> None:
        return

    def finish(self, status: str) -> None:
        return


class WandbTrackingClient(TrackingClient):
    def __init__(self, wandb_module: Any, run: Any) -> None:
        self._wandb = wandb_module
        self._run = run

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        scalar_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                scalar_metrics[key] = int(value)
            elif isinstance(value, (int, float)):
                scalar_metrics[key] = value
            else:
                self._run.summary[f"meta/{key}"] = str(value)

        if not scalar_metrics:
            return

        if step is None:
            self._run.log(scalar_metrics)
        else:
            self._run.log(scalar_metrics, step=step)

    def log_artifact(self, path: Path, artifact_name: str, artifact_type: str) -> None:
        if not path.exists():
            return
        artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self, status: str) -> None:
        self._run.summary["run_status"] = status
        self._run.finish()


def create_tracking_client(cfg: Dict[str, Any], run_ctx: RunContext, *, mode: str, resume: bool) -> TrackingClient:
    tracking_cfg = cfg.get("tracking", {})
    if not bool(tracking_cfg.get("enabled", False)):
        return NullTrackingClient()

    backend = str(tracking_cfg.get("backend", "wandb")).strip().lower()
    if backend != "wandb":
        raise ValueError(f"Unsupported tracking backend: {backend}")

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "tracking.enabled is true, but the wandb package is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    exp_cfg = cfg.get("experiment", {})
    run = wandb.init(
        project=str(tracking_cfg.get("project", "cvae-testing")),
        entity=tracking_cfg.get("entity"),
        name=str(tracking_cfg.get("run_name", run_ctx.run_root.name)),
        group=tracking_cfg.get("group", exp_cfg.get("name")),
        tags=list(tracking_cfg.get("tags", [])) + [f"mode:{mode}", f"resume:{int(bool(resume))}"],
        config=cfg,
        dir=str(run_ctx.run_root),
    )
    run.summary["run_root"] = str(run_ctx.run_root)
    return WandbTrackingClient(wandb_module=wandb, run=run)
