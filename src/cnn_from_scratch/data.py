"""A deterministic synthetic image-classification task with spatial structure.

Each image contains one oriented stroke on a noisy background: a horizontal bar,
a vertical bar, or a diagonal. The class is the orientation. The orientation is a
spatial pattern, so a convolutional network that sees local structure should beat
a linear model on raw pixels, which has no notion of adjacency. Images and splits
are generated with a seeded random generator and are disjoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

Image = list[list[list[float]]]  # (channels, height, width)

CLASS_NAMES = ("horizontal", "vertical", "diagonal")


@dataclass(frozen=True)
class ImageTask:
    """Shape of the generated images."""

    size: int = 8
    noise: float = 0.15

    @property
    def n_classes(self) -> int:
        return len(CLASS_NAMES)


def _blank(size: int, rng: random.Random, noise: float) -> list[list[float]]:
    return [[rng.uniform(0.0, noise) for _ in range(size)] for _ in range(size)]


def make_image(task: ImageTask, label: int, rng: random.Random) -> Image:
    """Render one noisy image with a short oriented stroke at a random position.

    The discriminative motif (a length-three horizontal, vertical, or diagonal
    segment) is placed at a random location, so the class is a local pattern that
    can appear anywhere. A convolutional filter detects it regardless of position;
    a linear model on raw pixels cannot, because no fixed pixel is reliably lit.
    """

    size = task.size
    grid = _blank(size, rng, task.noise)
    seg = 3
    if label == 0:  # horizontal segment
        row = rng.randint(0, size - 1)
        col = rng.randint(0, size - seg)
        for d in range(seg):
            grid[row][col + d] = min(1.0, grid[row][col + d] + 1.0)
    elif label == 1:  # vertical segment
        row = rng.randint(0, size - seg)
        col = rng.randint(0, size - 1)
        for d in range(seg):
            grid[row + d][col] = min(1.0, grid[row + d][col] + 1.0)
    else:  # diagonal segment
        row = rng.randint(0, size - seg)
        col = rng.randint(0, size - seg)
        for d in range(seg):
            grid[row + d][col + d] = min(1.0, grid[row + d][col + d] + 1.0)
    return [grid]


def make_dataset(task: ImageTask, count: int, *, seed: int) -> tuple[list[Image], list[int]]:
    """Return ``count`` balanced, labeled images drawn with a seeded generator."""

    rng = random.Random(seed)
    images: list[Image] = []
    labels: list[int] = []
    for i in range(count):
        label = i % task.n_classes
        images.append(make_image(task, label, rng))
        labels.append(label)
    return images, labels


def train_data(
    task: ImageTask, count: int = 120, *, seed: int = 0
) -> tuple[list[Image], list[int]]:
    """Return the training split."""

    return make_dataset(task, count, seed=seed)


def test_data(task: ImageTask, count: int = 90, *, seed: int = 0) -> tuple[list[Image], list[int]]:
    """Return a held-out split generated from a different seed stream."""

    return make_dataset(task, count, seed=seed + 101)
