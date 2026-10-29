"""Neural network layers with explicit forward and backward passes.

Rather than a general autodiff graph, each layer implements ``forward`` and
``backward`` directly, which is the clearest way to show how gradients flow
through a convolutional network. A layer caches whatever its backward pass needs
during the forward pass. Everything operates on nested Python lists; there is no
NumPy and no other numerical library. The correctness of every backward pass is
checked against finite differences in the test suite.

Shapes
------
Images are ``(channels, height, width)``. Convolution and pooling preserve that
layout; :class:`Flatten` turns it into a vector for the dense head.
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

Vector = list[float]
Image = list[list[list[float]]]  # (channels, height, width)


class Layer(Protocol):
    """A network layer with a forward pass and a gradient-propagating backward pass."""

    def forward(self, x: Any) -> Any: ...
    def backward(self, grad: Any) -> Any: ...
    def parameters(self) -> list[Param]: ...


class Param:
    """A trainable tensor and its gradient, stored as flat lists with a shape."""

    __slots__ = ("grad", "shape", "values")

    def __init__(self, values: list[float], shape: tuple[int, ...]) -> None:
        self.values = values
        self.shape = shape
        self.grad = [0.0] * len(values)

    def zero_grad(self) -> None:
        for i in range(len(self.grad)):
            self.grad[i] = 0.0


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
        self._input: Image | None = None

    def parameters(self) -> list[Param]:
        return [self.weight, self.bias]

    def _w(self, oc: int, ic: int, i: int, j: int) -> float:
        k = self.kernel
        return self.weight.values[((oc * self.in_channels + ic) * k + i) * k + j]

    def forward(self, x: Image) -> Image:
        self._input = x
        c_in = len(x)
        h = len(x[0])
        w = len(x[0][0])
        k = self.kernel
        out_h = h - k + 1
        out_w = w - k + 1
        out: Image = [[[0.0] * out_w for _ in range(out_h)] for _ in range(self.out_channels)]
        for oc in range(self.out_channels):
            bias = self.bias.values[oc]
            out_c = out[oc]
            for oi in range(out_h):
                for oj in range(out_w):
                    acc = bias
                    for ic in range(c_in):
                        xic = x[ic]
                        for di in range(k):
                            xrow = xic[oi + di]
                            for dj in range(k):
                                acc += self._w(oc, ic, di, dj) * xrow[oj + dj]
                    out_c[oi][oj] = acc
        return out

    def backward(self, grad: Image) -> Image:
        assert self._input is not None
        x = self._input
        c_in = len(x)
        h = len(x[0])
        w = len(x[0][0])
        k = self.kernel
        out_h = h - k + 1
        out_w = w - k + 1
        dx: Image = [[[0.0] * w for _ in range(h)] for _ in range(c_in)]
        for oc in range(self.out_channels):
            gc = grad[oc]
            base = oc * self.in_channels
            for oi in range(out_h):
                for oj in range(out_w):
                    g = gc[oi][oj]
                    if g == 0.0:
                        continue
                    self.bias.grad[oc] += g
                    for ic in range(c_in):
                        xic = x[ic]
                        dxic = dx[ic]
                        widx0 = (base + ic) * k
                        for di in range(k):
                            xrow = xic[oi + di]
                            dxrow = dxic[oi + di]
                            wrow = (widx0 + di) * k
                            for dj in range(k):
                                self.weight.grad[wrow + dj] += g * xrow[oj + dj]
                                dxrow[oj + dj] += g * self.weight.values[wrow + dj]
        return dx


class ReLU:
    """Rectified linear activation over an image or a vector."""

    def __init__(self) -> None:
        self._mask: list[float] | None = None
        self._shape: tuple[int, ...] | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: object) -> object:
        flat, shape = _flatten_any(x)
        self._mask = [1.0 if v > 0 else 0.0 for v in flat]
        self._shape = shape
        return _unflatten([v if v > 0 else 0.0 for v in flat], shape)

    def backward(self, grad: object) -> object:
        assert self._mask is not None and self._shape is not None
        gflat, _ = _flatten_any(grad)
        return _unflatten([g * m for g, m in zip(gflat, self._mask, strict=True)], self._shape)


class MaxPool2D:
    """Non-overlapping max pooling with a square window."""

    def __init__(self, size: int = 2) -> None:
        self.size = size
        self._argmax: list[tuple[int, int, int, int]] | None = None
        self._in_shape: tuple[int, int, int] | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: Image) -> Image:
        c = len(x)
        h = len(x[0])
        w = len(x[0][0])
        s = self.size
        out_h = h // s
        out_w = w // s
        self._in_shape = (c, h, w)
        argmax: list[tuple[int, int, int, int]] = []
        out: Image = [[[0.0] * out_w for _ in range(out_h)] for _ in range(c)]
        for ch in range(c):
            for oi in range(out_h):
                for oj in range(out_w):
                    best = -math.inf
                    bi = bj = 0
                    for di in range(s):
                        for dj in range(s):
                            v = x[ch][oi * s + di][oj * s + dj]
                            if v > best:
                                best = v
                                bi, bj = oi * s + di, oj * s + dj
                    out[ch][oi][oj] = best
                    argmax.append((ch, oi * out_w + oj, bi, bj))
        self._argmax = argmax
        return out

    def backward(self, grad: Image) -> Image:
        assert self._argmax is not None and self._in_shape is not None
        c, h, w = self._in_shape
        out_w = w // self.size
        dx: Image = [[[0.0] * w for _ in range(h)] for _ in range(c)]
        for ch, flat_out, bi, bj in self._argmax:
            oi, oj = divmod(flat_out, out_w)
            dx[ch][bi][bj] += grad[ch][oi][oj]
        return dx


class Flatten:
    """Reshape an image into a vector and back on the backward pass."""

    def __init__(self) -> None:
        self._shape: tuple[int, int, int] | None = None

    def parameters(self) -> list[Param]:
        return []

    def forward(self, x: Image) -> Vector:
        c = len(x)
        h = len(x[0])
        w = len(x[0][0])
        self._shape = (c, h, w)
        return [x[ch][i][j] for ch in range(c) for i in range(h) for j in range(w)]

    def backward(self, grad: Vector) -> Image:
        assert self._shape is not None
        c, h, w = self._shape
        out: Image = [[[0.0] * w for _ in range(h)] for _ in range(c)]
        idx = 0
        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    out[ch][i][j] = grad[idx]
                    idx += 1
        return out


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
        self._input: Vector | None = None

    def parameters(self) -> list[Param]:
        return [self.weight, self.bias]

    def forward(self, x: Vector) -> Vector:
        self._input = x
        out = []
        for o in range(self.out_features):
            base = o * self.in_features
            acc = self.bias.values[o]
            for i in range(self.in_features):
                acc += self.weight.values[base + i] * x[i]
            out.append(acc)
        return out

    def backward(self, grad: Vector) -> Vector:
        assert self._input is not None
        x = self._input
        dx = [0.0] * self.in_features
        for o in range(self.out_features):
            g = grad[o]
            base = o * self.in_features
            self.bias.grad[o] += g
            for i in range(self.in_features):
                self.weight.grad[base + i] += g * x[i]
                dx[i] += g * self.weight.values[base + i]
        return dx


def _flatten_any(x: object) -> tuple[list[float], tuple[int, ...]]:
    if isinstance(x, list) and x and isinstance(x[0], list):
        img: Image = x
        c = len(img)
        h = len(img[0])
        w = len(img[0][0])
        return ([img[ch][i][j] for ch in range(c) for i in range(h) for j in range(w)], (c, h, w))
    vec: Vector = x  # type: ignore[assignment]
    return (list(vec), (len(vec),))


def _unflatten(flat: list[float], shape: tuple[int, ...]) -> object:
    if len(shape) == 1:
        return flat
    c, h, w = shape
    out: Image = [[[0.0] * w for _ in range(h)] for _ in range(c)]
    idx = 0
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                out[ch][i][j] = flat[idx]
                idx += 1
    return out
