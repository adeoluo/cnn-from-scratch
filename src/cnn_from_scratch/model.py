"""A sequential container, a small convolutional network, and the loss."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from cnn_from_scratch.layers import Conv2D, Dense, Flatten, Layer, MaxPool2D, Param, ReLU

Vector = list[float]
Image = list[list[list[float]]]


class Sequential:
    """A stack of layers applied in order, with gradients propagated in reverse."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def parameters(self) -> list[Param]:
        params: list[Param] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x: object) -> Vector:
        out: object = x
        for layer in self.layers:
            out = layer.forward(out)
        return out  # type: ignore[return-value]

    def backward(self, grad: Vector) -> None:
        g: object = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


@dataclass(frozen=True)
class CNNConfig:
    """Shape of the convolutional classifier."""

    image_size: int
    n_classes: int
    in_channels: int = 1
    conv_channels: int = 6
    kernel: int = 3
    pool: int = 2


def build_cnn(cfg: CNNConfig, *, seed: int = 0) -> Sequential:
    """Construct a small conv to pool to dense classifier."""

    rng = random.Random(seed)
    conv_out = cfg.image_size - cfg.kernel + 1
    pooled = conv_out // cfg.pool
    flat = cfg.conv_channels * pooled * pooled
    return Sequential(
        [
            Conv2D(cfg.in_channels, cfg.conv_channels, cfg.kernel, rng),
            ReLU(),
            MaxPool2D(cfg.pool),
            Flatten(),
            Dense(flat, cfg.n_classes, rng),
        ]
    )


def softmax(logits: Vector) -> Vector:
    """A numerically stable softmax over a vector of logits."""

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_grad(logits: Vector, target: int) -> tuple[float, Vector]:
    """Return the softmax cross-entropy loss and its gradient with respect to logits."""

    probs = softmax(logits)
    loss = -math.log(max(probs[target], 1e-12))
    grad = [p - (1.0 if i == target else 0.0) for i, p in enumerate(probs)]
    return loss, grad
