"""Shared metrics logging utilities."""

import json
from pathlib import Path

import mlflow
import numpy as np


class MetricsLogger:
    """Logs to both JSONL file and MLflow."""

    def __init__(self, filepath, end_run_on_close: bool = True):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "w")
        self.end_run_on_close = end_run_on_close

    def log(self, metrics: dict, step: int = None):
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()

        if step is None and "final" not in metrics:
            step = metrics.get("steps", metrics.get("update", 0))

        if step is not None:
            mlflow_metrics = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating)) and k not in ("steps", "update")
            }
            mlflow.log_metrics(mlflow_metrics, step=step)

    def close(self):
        self.file.close()
        mlflow.log_artifact(str(self.filepath))
        if self.end_run_on_close:
            mlflow.end_run()
