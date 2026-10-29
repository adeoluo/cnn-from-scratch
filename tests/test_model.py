"""Tests for layer shapes, data, training, and the baseline comparison."""

from __future__ import annotations

import random

from cnn_from_scratch.data import CLASS_NAMES, ImageTask, make_image, train_data
from cnn_from_scratch.data import test_data as held_out_data
from cnn_from_scratch.evaluation import LogisticRegression, evaluate, mean_confidence
from cnn_from_scratch.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from cnn_from_scratch.model import CNNConfig, build_cnn


def test_conv2d_output_shape() -> None:
    conv = Conv2D(1, 3, 3, random.Random(0))
    out = conv.forward([[[0.0] * 6 for _ in range(6)]])
    assert len(out) == 3
    assert len(out[0]) == 4 and len(out[0][0]) == 4


def test_maxpool_halves_spatial_dims() -> None:
    pool = MaxPool2D(2)
    out = pool.forward([[[float(i) for i in range(4)] for _ in range(4)]])
    assert len(out[0]) == 2 and len(out[0][0]) == 2


def test_relu_zeroes_negatives() -> None:
    relu = ReLU()
    out = relu.forward([[[-1.0, 2.0], [3.0, -4.0]]])
    assert out == [[[0.0, 2.0], [3.0, 0.0]]]


def test_flatten_roundtrip_shape() -> None:
    flat = Flatten()
    vec = flat.forward([[[1.0, 2.0], [3.0, 4.0]]])
    assert vec == [1.0, 2.0, 3.0, 4.0]
    back = flat.backward([10.0, 20.0, 30.0, 40.0])
    assert back == [[[10.0, 20.0], [30.0, 40.0]]]


def test_dense_shapes() -> None:
    dense = Dense(4, 3, random.Random(0))
    out = dense.forward([1.0, 2.0, 3.0, 4.0])
    assert len(out) == 3


def test_build_cnn_classifies_to_n_classes() -> None:
    task = ImageTask()
    model = build_cnn(CNNConfig(image_size=task.size, n_classes=task.n_classes), seed=0)
    image = make_image(task, 0, random.Random(0))
    assert len(model.forward(image)) == task.n_classes


def test_dataset_is_balanced_and_shaped() -> None:
    task = ImageTask()
    images, labels = train_data(task, count=30, seed=0)
    assert all(len(img) == 1 and len(img[0]) == task.size for img in images)
    counts = [labels.count(c) for c in range(task.n_classes)]
    assert counts == [10, 10, 10]


def test_cnn_beats_linear_baseline(trained, small_task) -> None:  # type: ignore[no-untyped-def]
    test_i, test_l = held_out_data(small_task, count=60, seed=0)
    report = evaluate(trained.model, test_i, test_l, small_task.n_classes)
    train_i, train_l = train_data(small_task, seed=0)
    baseline = LogisticRegression.fit(train_i, train_l, small_task.n_classes).evaluate(
        test_i, test_l
    )
    assert report.accuracy > 0.8
    assert report.accuracy > baseline + 0.2
    assert trained.history[0][2] < trained.history[-1][2]


def test_training_is_deterministic(small_task, small_cfg) -> None:  # type: ignore[no-untyped-def]
    from cnn_from_scratch.training import TrainConfig, train

    a = train(small_task, small_cfg, TrainConfig(epochs=3, seed=0))
    b = train(small_task, small_cfg, TrainConfig(epochs=3, seed=0))
    assert a.model.parameters()[0].values == b.model.parameters()[0].values


def test_mean_confidence_in_range(trained, small_task) -> None:  # type: ignore[no-untyped-def]
    test_i, test_l = held_out_data(small_task, count=30, seed=0)
    assert 0.0 <= mean_confidence(trained.model, test_i, test_l) <= 1.0


def test_class_names_match_count() -> None:
    assert len(CLASS_NAMES) == ImageTask().n_classes
