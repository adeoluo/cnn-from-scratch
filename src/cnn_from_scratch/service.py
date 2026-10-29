"""A dependency-free HTTP service exposing the trained convolutional classifier.

The service trains the network once at startup, then classifies images and
reports held-out accuracy. The routing logic is a pure function tested without a
socket.

Endpoints
---------
``GET  /health``    liveness and class names.
``GET  /ready``     readiness once the network is trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``POST /classify``  body ``{"image": [[...]]}`` (one channel); returns class and probabilities.
``GET  /evaluate``  accuracy and confusion matrix on held-out data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cnn_from_scratch.data import CLASS_NAMES, ImageTask, test_data
from cnn_from_scratch.evaluation import EvalReport, evaluate
from cnn_from_scratch.model import CNNConfig, Sequential, softmax
from cnn_from_scratch.observability import ServiceMetrics
from cnn_from_scratch.training import TrainConfig, train

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ClassifierService:
    """Holds the trained network and answers classify and evaluate requests."""

    model: Sequential
    task: ImageTask
    test_images: list[list[list[list[float]]]]
    test_labels: list[int]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(
        cls, *, task: ImageTask | None = None, train_cfg: TrainConfig | None = None
    ) -> ClassifierService:
        task = task or ImageTask()
        cfg = CNNConfig(image_size=task.size, n_classes=task.n_classes)
        result = train(task, cfg, train_cfg or TrainConfig())
        images, labels = test_data(task, seed=0)
        return cls(model=result.model, task=task, test_images=images, test_labels=labels)

    def classify(self, grid: list[list[float]]) -> JsonObject:
        logits = self.model.forward([grid])
        probs = softmax(logits)
        pred = max(range(len(probs)), key=lambda c: probs[c])
        return {
            "class": CLASS_NAMES[pred],
            "class_index": pred,
            "probabilities": {
                name: round(p, 6) for name, p in zip(CLASS_NAMES, probs, strict=True)
            },
        }

    def evaluate(self) -> EvalReport:
        return evaluate(self.model, self.test_images, self.test_labels, self.task.n_classes)


def _valid_image(value: Any, size: int) -> list[list[float]] | str:
    if not isinstance(value, list) or len(value) != size:
        return f"must be a {size}x{size} list of rows"
    grid: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            return f"each row must have {size} numbers"
        parsed: list[float] = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return "pixels must be numbers"
            parsed.append(float(v))
        grid.append(parsed)
    return grid


def _route(
    method: str, path: str, body: Mapping[str, Any], service: ClassifierService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "classes": list(CLASS_NAMES), "image_size": service.task.size}
    if method == "GET" and path == "/evaluate":
        report = service.evaluate()
        return 200, {
            "accuracy": round(report.accuracy, 6),
            "correct": report.correct,
            "total": report.total,
            "confusion": report.confusion,
        }
    if method == "POST" and path == "/classify":
        grid = _valid_image(body.get("image"), service.task.size)
        if isinstance(grid, str):
            return 400, {"error": f"'image' {grid}"}
        return 200, service.classify(grid)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: ClassifierService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: ClassifierService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": ClassifierService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"cnn-from-scratch service listening on http://{host}:{port}")
    server.serve_forever()
