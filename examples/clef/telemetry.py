"""Decision-training metrics, traces, and native Miles dashboard streams."""

import json
import re
import time
from pathlib import Path
from typing import Any

import wandb
from prometheus_client import CollectorRegistry, Gauge, start_http_server

from miles.dashboard.store import Meta, MetricsRecord, MetricStore


class Telemetry:
    def __init__(self, output_dir: Path, run_name: str, config: dict[str, Any], project: str, entity: str, port: int) -> None:
        self.output_dir = output_dir
        started = time.time()
        self.store = MetricStore(output_dir / "dashboard")
        self.store.write_meta(Meta(run_name=run_name, start_ts=started, args=config))
        self.registry = CollectorRegistry()
        self.gauges: dict[str, Gauge] = {}
        self.run_name = run_name
        self.server, self.thread = start_http_server(port, registry=self.registry)
        self.run = wandb.init(project=project, entity=entity or None, name=run_name, config=config) if project else None
        if self.run is not None:
            (output_dir / "wandb-url.txt").write_text(self.run.url)
            dashboard_args = {**config, "wandb_team": self.run.entity, "wandb_project": self.run.project, "wandb_run_id": self.run.id}
            self.store.write_meta(Meta(run_name=run_name, start_ts=started, args=dashboard_args))

    def log(self, metrics: dict[str, float], step: int) -> None:
        values = {"train/step": step, **metrics}
        self.store.append(MetricsRecord(ts=time.time(), step_key="train/step", step=step, metrics=values))
        self.store.flush()
        with (self.output_dir / "metrics.jsonl").open("a") as writer:
            writer.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.run is not None:
            self.run.log(values, step=step)
        for key, value in values.items():
            if key not in self.gauges:
                safe = "miles_metric_" + re.sub(r"[^a-zA-Z0-9_]", "_", key)
                self.gauges[key] = Gauge(safe, key, ["run_name"], registry=self.registry)
            self.gauges[key].labels(self.run_name).set(value)
        print("METRICS", json.dumps({"step": step, **metrics}), flush=True)

    def close(self) -> None:
        self.store.flush()
        self.server.shutdown()
        if self.run is not None:
            self.run.finish()
