"""Evaluation: accuracy, confusion matrix, and a linear baseline.

The baseline is multinomial logistic regression on raw pixels, trained by the
same cross-entropy gradient. It has no notion of spatial adjacency, so the gap to
the convolutional network measures what convolution buys on this task.
"""

from __future__ import annotations

from dataclasses import dataclass

from cnn_from_scratch.model import Sequential, cross_entropy_grad, softmax

Image = list[list[list[float]]]


@dataclass(frozen=True)
class EvalReport:
    """Accuracy and a confusion matrix over a labeled set."""

    accuracy: float
    correct: int
    total: int
    confusion: list[list[int]]

    def format(self) -> str:
        return f"accuracy={self.accuracy:.4f} ({self.correct}/{self.total})"


def evaluate(
    model: Sequential, images: list[Image], labels: list[int], n_classes: int
) -> EvalReport:
    """Return accuracy and a confusion matrix for ``model`` over the data."""

    confusion = [[0] * n_classes for _ in range(n_classes)]
    correct = 0
    for image, label in zip(images, labels, strict=True):
        logits = model.forward(image)
        pred = max(range(len(logits)), key=lambda c: logits[c])
        confusion[label][pred] += 1
        if pred == label:
            correct += 1
    total = len(images)
    return EvalReport(
        accuracy=correct / total if total else 0.0,
        correct=correct,
        total=total,
        confusion=confusion,
    )


class LogisticRegression:
    """Multinomial logistic regression on flattened pixels (the linear baseline)."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.weight = [0.0] * (n_classes * n_features)
        self.bias = [0.0] * n_classes

    def logits(self, pixels: list[float]) -> list[float]:
        out = []
        for c in range(self.n_classes):
            base = c * self.n_features
            acc = self.bias[c]
            for i in range(self.n_features):
                acc += self.weight[base + i] * pixels[i]
            out.append(acc)
        return out

    @classmethod
    def fit(
        cls,
        images: list[Image],
        labels: list[int],
        n_classes: int,
        *,
        epochs: int = 40,
        lr: float = 0.1,
    ) -> LogisticRegression:
        flat = [_flatten(img) for img in images]
        model = cls(len(flat[0]), n_classes)
        for _ in range(epochs):
            for pixels, label in zip(flat, labels, strict=True):
                _loss, grad = cross_entropy_grad(model.logits(pixels), label)
                for c in range(n_classes):
                    base = c * model.n_features
                    g = grad[c]
                    model.bias[c] -= lr * g
                    for i in range(model.n_features):
                        model.weight[base + i] -= lr * g * pixels[i]
        return model

    def evaluate(self, images: list[Image], labels: list[int]) -> float:
        correct = 0
        for image, label in zip(images, labels, strict=True):
            logits = self.logits(_flatten(image))
            if max(range(len(logits)), key=lambda c: logits[c]) == label:
                correct += 1
        return correct / len(images) if images else 0.0


def _flatten(image: Image) -> list[float]:
    return [v for channel in image for row in channel for v in row]


def mean_confidence(model: Sequential, images: list[Image], labels: list[int]) -> float:
    """Average softmax probability assigned to the true class (a calibration readout)."""

    total = 0.0
    for image, label in zip(images, labels, strict=True):
        total += softmax(model.forward(image))[label]
    return total / len(images) if images else 0.0
