"""CNN From Scratch - a convolutional neural network and its autodiff, from scratch."""

from __future__ import annotations

from cnn_from_scratch.data import CLASS_NAMES, ImageTask, make_image, test_data, train_data
from cnn_from_scratch.evaluation import EvalReport, LogisticRegression, evaluate, mean_confidence
from cnn_from_scratch.layers import Conv2D, Dense, Flatten, MaxPool2D, Param, ReLU
from cnn_from_scratch.model import CNNConfig, Sequential, build_cnn, cross_entropy_grad, softmax
from cnn_from_scratch.training import Adam, TrainConfig, TrainResult, train

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "Adam",
    "CNNConfig",
    "Conv2D",
    "Dense",
    "EvalReport",
    "Flatten",
    "ImageTask",
    "LogisticRegression",
    "MaxPool2D",
    "Param",
    "ReLU",
    "Sequential",
    "TrainConfig",
    "TrainResult",
    "__version__",
    "build_cnn",
    "cross_entropy_grad",
    "evaluate",
    "make_image",
    "mean_confidence",
    "softmax",
    "test_data",
    "train",
    "train_data",
]
