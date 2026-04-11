from __future__ import annotations

import base64
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import pstdev
from typing import Any

try:  # pragma: no cover - exercised indirectly in training environments
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - lightweight fallback for tests without transformers
    class TrainerCallback:  # type: ignore[no-redef]
        pass


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pQ7cAAAAASUVORK5CYII="
)
_RESERVED_COLUMNS = ("step", "epoch", "source", "recorded_at")


def write_training_summary(path: Path, metrics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def _coerce_scalar(value: Any) -> bool | int | float | str | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # pragma: no cover - defensive scalar conversion
            value = str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    dynamic = {
        key
        for row in rows
        for key in row
        if key not in _RESERVED_COLUMNS
    }
    return [*(_RESERVED_COLUMNS), *sorted(dynamic)]


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PLACEHOLDER_PNG)


def _build_numeric_series(
    rows: list[dict[str, Any]], metric_keys: list[str]
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for metric in metric_keys:
        points: list[tuple[float, float]] = []
        fallback_step = 0.0
        for row in rows:
            value = row.get(metric)
            if not _is_numeric(value):
                continue
            step_value = row.get("step")
            if _is_numeric(step_value):
                x_value = float(step_value)
            else:
                fallback_step += 1.0
                x_value = fallback_step
            points.append((x_value, float(value)))
        if points:
            series[metric] = points
    return series


def _plot_metrics(
    *,
    rows: list[dict[str, Any]],
    metric_keys: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _build_numeric_series(rows, metric_keys)
    try:  # pragma: no cover - matplotlib is optional in non-training test environments
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:
        _write_placeholder_png(output_path)
        return {
            "path": str(output_path),
            "metrics": metric_keys,
            "rendered": False,
            "reason": "matplotlib unavailable",
        }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if series:
        for metric, points in series.items():
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=metric)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No metrics logged", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "path": str(output_path),
        "metrics": list(series),
        "rendered": bool(series),
        "reason": None if series else "no matching metrics",
    }


def _metric_candidates(
    rows: list[dict[str, Any]], *, preferred: list[str], contains: str
) -> list[str]:
    numeric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if key not in _RESERVED_COLUMNS and _is_numeric(value)
    }
    ordered = [key for key in preferred if key in numeric_keys]
    extras = sorted(
        key
        for key in numeric_keys
        if key not in ordered and contains in key.lower()
    )
    return ordered + extras


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


class TrainingTelemetry:
    def __init__(self, stage: str, output_dir: Path) -> None:
        self.stage = stage
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.plots_dir = output_dir / "plots"
        self.tensorboard_dir = output_dir / "tensorboard"
        self.metrics_jsonl_path = self.logs_dir / f"{stage}_metrics.jsonl"
        self.metrics_csv_path = self.logs_dir / f"{stage}_metrics.csv"
        self.summary_path = self.logs_dir / f"{stage}_summary.json"
        self.rows: list[dict[str, Any]] = []

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def record_metrics(
        self,
        metrics: dict[str, Any],
        *,
        step: int | float | None = None,
        epoch: int | float | None = None,
        source: str,
    ) -> None:
        normalized: dict[str, Any] = {
            "step": _coerce_scalar(step),
            "epoch": _coerce_scalar(epoch),
            "source": source,
            "recorded_at": _now_utc(),
        }
        for key, value in metrics.items():
            if key == "total_flos":
                continue
            normalized[key] = _coerce_scalar(value)
        self.rows.append(normalized)
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, sort_keys=True) + "\n")

    def absorb_trainer_history(self, trainer: Any) -> None:
        if self.rows:
            return
        state = getattr(trainer, "state", None)
        log_history = getattr(state, "log_history", None)
        if not isinstance(log_history, list):
            return
        for row in log_history:
            if not isinstance(row, dict):
                continue
            self.record_metrics(
                row,
                step=row.get("step", getattr(state, "global_step", None)),
                epoch=row.get("epoch", getattr(state, "epoch", None)),
                source="trainer_state",
            )

    def finalize(
        self,
        *,
        trainer: Any | None = None,
        final_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if trainer is not None:
            self.absorb_trainer_history(trainer)
        if final_metrics:
            state = getattr(trainer, "state", None)
            self.record_metrics(
                final_metrics,
                step=getattr(state, "global_step", None),
                epoch=getattr(state, "epoch", None),
                source="train_result",
            )

        unique_rows = _dedupe_rows(self.rows)
        if unique_rows != self.rows:
            self.rows = unique_rows
            self.metrics_jsonl_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in self.rows)
                + ("\n" if self.rows else ""),
                encoding="utf-8",
            )

        if self.rows:
            with self.metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_ordered_columns(self.rows))
                writer.writeheader()
                writer.writerows(self.rows)
        else:
            self.metrics_csv_path.write_text("", encoding="utf-8")

        if self.stage == "sft":
            plots = {
                "loss": _plot_metrics(
                    rows=self.rows,
                    metric_keys=_metric_candidates(
                        self.rows,
                        preferred=["loss", "train_loss", "eval_loss"],
                        contains="loss",
                    ),
                    output_path=self.plots_dir / "sft_loss.png",
                    title="SFT Loss Over Time",
                    ylabel="loss",
                )
            }
        else:
            plots = {
                "reward": _plot_metrics(
                    rows=self.rows,
                    metric_keys=_metric_candidates(
                        self.rows,
                        preferred=["reward_mean", "reward", "mean_reward", "reward_max", "reward_min"],
                        contains="reward",
                    ),
                    output_path=self.plots_dir / "grpo_reward.png",
                    title="GRPO Reward Over Time",
                    ylabel="reward",
                )
            }

        latest_metrics = self.rows[-1] if self.rows else {}
        summary = {
            "stage": self.stage,
            "row_count": len(self.rows),
            "tensorboard_dir": str(self.tensorboard_dir),
            "metrics_jsonl": str(self.metrics_jsonl_path),
            "metrics_csv": str(self.metrics_csv_path),
            "plots": plots,
            "latest_metrics": latest_metrics,
        }
        write_training_summary(self.summary_path, summary)
        return summary


class TrainingTelemetryCallback(TrainerCallback):
    def __init__(self, telemetry: TrainingTelemetry) -> None:
        super().__init__()
        self.telemetry = telemetry

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        if logs:
            self.telemetry.record_metrics(
                logs,
                step=getattr(state, "global_step", None),
                epoch=getattr(state, "epoch", None),
                source="trainer",
            )
        return control


def reward_log_row(rewards: list[float], *, step: int) -> dict[str, float | int]:
    if not rewards:
        return {"step": step, "reward_count": 0, "reward_mean": 0.0}
    mean_reward = sum(rewards) / len(rewards)
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    return {
        "step": step,
        "reward_count": len(rewards),
        "reward_mean": mean_reward,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_std": reward_std,
    }
