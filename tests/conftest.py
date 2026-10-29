"""Shared session-scoped fixtures so the network is trained once for all tests."""

from __future__ import annotations

import pytest

from cnn_from_scratch.data import ImageTask, test_data
from cnn_from_scratch.model import CNNConfig
from cnn_from_scratch.service import ClassifierService
from cnn_from_scratch.training import TrainConfig, TrainResult, train


@pytest.fixture(scope="session")
def small_task() -> ImageTask:
    return ImageTask(size=8, noise=0.15)


@pytest.fixture(scope="session")
def small_cfg(small_task: ImageTask) -> CNNConfig:
    return CNNConfig(image_size=small_task.size, n_classes=small_task.n_classes, conv_channels=6)


@pytest.fixture(scope="session")
def trained(small_task: ImageTask, small_cfg: CNNConfig) -> TrainResult:
    return train(small_task, small_cfg, TrainConfig(epochs=15, seed=0))


@pytest.fixture(scope="session")
def service(small_task: ImageTask, trained: TrainResult) -> ClassifierService:
    images, labels = test_data(small_task, seed=0)
    return ClassifierService(
        model=trained.model, task=small_task, test_images=images, test_labels=labels
    )
