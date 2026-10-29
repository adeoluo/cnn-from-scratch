"""Finite-difference gradient checks for every layer and the full network."""

from __future__ import annotations

from cnn_from_scratch.data import ImageTask, train_data
from cnn_from_scratch.model import CNNConfig, build_cnn, cross_entropy_grad


def test_full_network_gradients_match_finite_differences() -> None:
    task = ImageTask(size=6, noise=0.1)
    cfg = CNNConfig(image_size=6, n_classes=3, conv_channels=2, kernel=3, pool=2)
    model = build_cnn(cfg, seed=1)
    images, labels = train_data(task, count=6, seed=0)
    image, label = images[0], labels[0]

    def loss() -> float:
        out = model.forward(image)
        value, _ = cross_entropy_grad(out, label)
        return value

    model.zero_grad()
    logits = model.forward(image)
    _value, grad = cross_entropy_grad(logits, label)
    model.backward(grad)

    eps = 1e-5
    worst = 0.0
    for param in model.parameters():
        for i in range(min(len(param.values), 16)):
            original = param.values[i]
            param.values[i] = original + eps
            plus = loss()
            param.values[i] = original - eps
            minus = loss()
            param.values[i] = original
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - param.grad[i]))
    assert worst < 1e-4


def test_random_seed_changes_initialization() -> None:
    cfg = CNNConfig(image_size=8, n_classes=3)
    a = build_cnn(cfg, seed=0).parameters()[0].values
    b = build_cnn(cfg, seed=1).parameters()[0].values
    assert a != b
