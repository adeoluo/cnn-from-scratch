"""A deterministic micro-benchmark for the convolutional classifier.

Trains the default CNN on the oriented-stroke task and reports training time,
held-out accuracy against a linear baseline, and mean confidence in the true
class. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from cnn_from_scratch.data import ImageTask, test_data, train_data
from cnn_from_scratch.evaluation import LogisticRegression, evaluate, mean_confidence
from cnn_from_scratch.model import CNNConfig
from cnn_from_scratch.training import TrainConfig, train


def main() -> None:
    task = ImageTask()
    cfg = CNNConfig(image_size=task.size, n_classes=task.n_classes)

    start = time.perf_counter()
    result = train(task, cfg, TrainConfig())
    elapsed = time.perf_counter() - start

    test_i, test_l = test_data(task, seed=0)
    train_i, train_l = train_data(task, seed=0)
    report = evaluate(result.model, test_i, test_l, task.n_classes)
    baseline = LogisticRegression.fit(train_i, train_l, task.n_classes).evaluate(test_i, test_l)

    print(f"Task: {task.n_classes} oriented strokes at random positions, {task.size}x{task.size}")
    print(f"Training (15 epochs): {elapsed:.0f} s")
    print(f"Held-out accuracy: CNN {report.accuracy:.3f} vs linear baseline {baseline:.3f}")
    print(f"Mean confidence in true class: {mean_confidence(result.model, test_i, test_l):.3f}")


if __name__ == "__main__":
    main()
