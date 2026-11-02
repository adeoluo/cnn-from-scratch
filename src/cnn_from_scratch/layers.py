"""Neural network layers with explicit forward and backward passes.

Rather than a general autodiff graph, each layer implements ``forward`` and
``backward`` directly, which is the clearest way to show how gradients flow
through a convolutional network. A layer caches whatever its backward pass needs
during the forward pass. The arithmetic runs on numpy arrays: convolution is
written as im2col (every receptive field laid out as a column) followed by one
matrix product, and its backward pass scatters the column gradients back onto the
image. The correctness of every backward pass is checked against finite
differences in the test suite.

Shapes
------
Images are ``(channels, height, width)``. Convolution and pooling preserve that
layout; :class:`Flatten` turns it into a vector for the dense head. Layers accept
nested lists or arrays and return arrays.
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

Array = npt.NDArray[np.float64]
Vector = list[float]
Image = list[list[list[float]]]  # (channels, height, width)


def as_array(x: object) -> Array:
    return np.asarray(x, dtype=np.float64)


class Layer(Protocol):
    """A network layer with a forward pass and a gradient-propagating backward pass."""

    def forward(self, x: Any) -> Any: ...
    def backward(self, grad: Any) -> Any: ...
    def parameters(self) -> list[Param]: ...


class Param:
    """A trainable tensor and its gradient, stored as flat arrays with a shape."""

    __slots__ = ("grad", "shape", "values")

    def __init__(self, values: list[float], shape: tuple[int, ...]) -> None:
        self.values: Array = np.asarray(values, dtype=np.float64)
        self.shape = shape
        self.grad: Array = np.zeros_like(self.values)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def view(self) -> Array:
        """The values reshaped to ``shape`` (a view: in-place updates show through)."""
        return self.values.reshape(self.shape)

    def grad_view(self) -> Array:
        return self.grad.reshape(self.shape)


def _he_scale(fan_in: int) -> float:
    return math.sqrt(2.0 / fan_in)


class Conv2D:
    """A valid (no padding) 2-D convolution over multi-channel images.

    Weights have shape ``(out_channels, in_channels, k, k)``. The forward pass
    slides each filter over the image; the backward pass accumulates gradients
    for the weights, biases, and input by reversing that sum.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, rng: random.Random
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        fan_in = in_channels * kernel * kernel
        scale = _he_scale(fan_in)
        w = [rng.gauss(0.0, scale) for _ in range(out_channels * fan_in)]
        self.weight = Param(w, (out_channels, in_channels, kernel, kernel))
        self.bias = Param([0.0] * out_channels, (out_channels,))
        self._columns: Array | None = None
        self._in_shape: tuple[int, int, int] | None = None

    def parameters(self) -> list[Param]:
        return [self.weight, self.bias]

    def forward(self, x: object) -> Array:
        image = as_array(x)
        c_in, h, w = image.shape
        k = self.kernel
        out_h, out_w = h - k + 1, w - k + 1
        # (c_in, out_h, out_w, k, k) windows -> columns of shape (c_in*k*k, out_h*out_w).
        windows = sliding_window_view(image, (k, k), axis=(1, 2))
        columns = windows.transpose(0, 3, 4, 1, 2).reshape(c_in * k * k, out_h * out_w)
        self._columns = columns
        self._in_shape = (c_in, h, w)
        weights = self.weight.values.reshape(self.out_channels, c_in * k * k)
        out = weights @ columns + self.bias.values[:, None]
        return out.reshape(self.out_channels, out_h, out_w)

    def backward(self, grad: object) -> Array:
        assert self._columns is not None and self._in_shape is not None
        c_in, h, w = self._in_shape
        k = self.kernel
        out_h, out_w = h - k + 1, w - k + 1
        g = as_array(grad).reshape(self.out_channels, out_h * out_w)
        self.bias.grad += g.sum(axis=1)
        self.weight.grad += (g @ self._columns.T).reshape(-1)
        weights = self.weight.values.reshape(self.out_channels, c_in * k * k)
        d_columns = (weights.T @ g).reshape(c_in, k, k, out_h, out_w)
        dx = np.zeros((c_in, h, w))
        for di in range(k):
            for dj in range(k):
                dx[:, di : di + out_h, dj : dj + out_w] += d_columns[:, di, dj]
        return dx


class ReLU:
    """Rectified linear activation over an image or a vector."""

    def __init__(self) -> None:
        self._mask: Array | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: object) -> Array:
        values = as_array(x)
        self._mask = (values > 0).astype(np.float64)
        return values * self._mask

    def backward(self, grad: object) -> Array:
        assert self._mask is not None
        return as_array(grad) * self._mask


class MaxPool2D:
    """Non-overlapping max pooling with a square window."""

    def __init__(self, size: int = 2) -> None:
        self.size = size
        self._argmax: npt.NDArray[np.intp] | None = None
        self._in_shape: tuple[int, int, int] | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: object) -> Array:
        image = as_array(x)
        c, h, w = image.shape
        s = self.size
        out_h, out_w = h // s, w // s
        self._in_shape = (c, h, w)
        # (c, out_h, out_w, s*s) windows in row-major order, so argmax picks the first
        # maximum exactly as a row-by-row scan with a strict ">" would.
        windows = (
            image[:, : out_h * s, : out_w * s]
            .reshape(c, out_h, s, out_w, s)
            .transpose(0, 1, 3, 2, 4)
            .reshape(c, out_h, out_w, s * s)
        )
        self._argmax = windows.argmax(axis=-1)
        out: Array = np.take_along_axis(windows, self._argmax[..., None], axis=-1)[..., 0]
        return out

    def backward(self, grad: object) -> Array:
        assert self._argmax is not None and self._in_shape is not None
        c, h, w = self._in_shape
        s = self.size
        out_h, out_w = h // s, w // s
        g = as_array(grad)
        di, dj = np.divmod(self._argmax, s)
        rows = np.arange(out_h)[None, :, None] * s + di
        cols = np.arange(out_w)[None, None, :] * s + dj
        channels = np.broadcast_to(np.arange(c)[:, None, None], rows.shape)
        dx = np.zeros((c, h, w))
        np.add.at(dx, (channels, rows, cols), g)
        return dx


class Flatten:
    """Reshape an image into a vector and back on the backward pass."""

    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: object) -> Array:
        image = as_array(x)
        self._shape = image.shape
        return image.reshape(-1)

    def backward(self, grad: object) -> Array:
        assert self._shape is not None
        return as_array(grad).reshape(self._shape)


class Dense:
    """A fully connected layer ``y = W x + b``."""

    def __init__(self, in_features: int, out_features: int, rng: random.Random) -> None:
        self.in_features = in_features
        self.out_features = out_features
        scale = _he_scale(in_features)
        self.weight = Param(
            [rng.gauss(0.0, scale) for _ in range(out_features * in_features)],
            (out_features, in_features),
        )
        self.bias = Param([0.0] * out_features, (out_features,))
        self._input: Array | None = None

    def parameters(self) -> list[Param]:
        return [self.weight, self.bias]

    def forward(self, x: object) -> Array:
        vector = as_array(x)
        self._input = vector
        result: Array = self.weight.view() @ vector + self.bias.values
        return result

    def backward(self, grad: object) -> Array:
        assert self._input is not None
        g = as_array(grad)
        self.bias.grad += g
        self.weight.grad += np.outer(g, self._input).reshape(-1)
        result: Array = self.weight.view().T @ g
        return result
