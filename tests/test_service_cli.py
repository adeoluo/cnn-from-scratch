"""Tests for the HTTP service routing and the CLI."""

from __future__ import annotations

import random

from cnn_from_scratch.cli import main
from cnn_from_scratch.data import make_image
from cnn_from_scratch.service import handle


def test_health_and_ready(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle("GET", "/health", {}, service)
    assert status == 200 and len(payload["classes"]) == service.task.n_classes
    status, payload = handle("GET", "/ready", {}, service)
    assert status == 200 and payload["status"] == "ready"


def test_classify_route(service) -> None:  # type: ignore[no-untyped-def]
    image = make_image(service.task, 1, random.Random(3))
    status, payload = handle("POST", "/classify", {"image": image[0]}, service)
    assert status == 200
    assert payload["class_index"] in range(service.task.n_classes)
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6


def test_classify_validation(service) -> None:  # type: ignore[no-untyped-def]
    size = service.task.size
    for body in (
        {"image": [[0.0] * size]},
        {"image": [[0.0] * (size - 1) for _ in range(size)]},
        {"image": [["x"] * size for _ in range(size)]},
        {},
    ):
        status, payload = handle("POST", "/classify", body, service)
        assert status == 400 and "error" in payload


def test_evaluate_route(service) -> None:  # type: ignore[no-untyped-def]
    status, payload = handle("GET", "/evaluate", {}, service)
    assert status == 200
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert len(payload["confusion"]) == service.task.n_classes


def test_unknown_route_and_metrics(service) -> None:  # type: ignore[no-untyped-def]
    status, _ = handle("GET", "/nope", {}, service)
    assert status == 404
    handle("GET", "/evaluate", {}, service)
    assert "service_requests_total" in service.metrics.render_prometheus()


def test_cli_train_and_evaluate(capsys) -> None:  # type: ignore[no-untyped-def]
    common = ["--size", "8", "--epochs", "4"]
    main(["train", *common])
    main(["evaluate", *common])
    out = capsys.readouterr().out
    assert "convolutional network:" in out
    assert "Confusion matrix" in out
