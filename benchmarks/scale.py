"""A deterministic scale benchmark for CNN inference throughput.

Trains one convolutional classifier on the oriented-stroke task, then runs the
from-scratch forward pass over progressively larger synthetic image sets. For
each workload it reports wall-clock seconds, inference throughput in images per
second, peak memory in MiB measured with the standard-library ``tracemalloc``
module, and held-out classification accuracy as the quality metric. Timing is
measured without tracemalloc (which otherwise slows every allocation); peak
memory is traced separately around the dataset plus a single forward pass, which
is faithful because ``evaluate`` processes one image at a time and discards each.

The data is synthetic and the network is intentionally small. This pure-Python
from-scratch implementation is not competitive on wall-clock time with compiled
tensor libraries; the goal is to show how the hand-written convolution forward
pass scales with the number of images. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from cnn_from_scratch.data import ImageTask, make_dataset
from cnn_from_scratch.evaluation import evaluate
from cnn_from_scratch.model import CNNConfig, Sequential
from cnn_from_scratch.training import TrainConfig, train

WORKLOAD_SIZES: tuple[int, ...] = (750, 1500, 3000, 6000)
_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class ScaleRow:
    """One measured workload: size, timing, throughput, memory, and quality."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    accuracy: float


def _train_model(epochs: int) -> tuple[Sequential, ImageTask]:
    task = ImageTask()
    cfg = CNNConfig(image_size=task.size, n_classes=task.n_classes)
    result = train(task, cfg, TrainConfig(epochs=epochs))
    return result.model, task


def _peak_mib(model: Sequential, task: ImageTask, size: int) -> float:
    """Peak memory to hold a workload of ``size`` images and run one forward pass.

    ``evaluate`` processes one image at a time and discards each result, so the
    peak is the dataset plus a single forward pass; tracing that is faithful and
    avoids tracemalloc's heavy per-allocation slowdown over the whole workload.
    """

    tracemalloc.start()
    tracemalloc.reset_peak()
    images, _ = make_dataset(task, size, seed=42)
    model.forward(images[0])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / _BYTES_PER_MIB


def run_scale(sizes: tuple[int, ...] = WORKLOAD_SIZES, *, epochs: int = 15) -> list[ScaleRow]:
    """Train one small model, then measure forward-pass scaling across sizes."""

    model, task = _train_model(epochs)
    rows: list[ScaleRow] = []
    for size in sizes:
        images, labels = make_dataset(task, size, seed=42)
        start = time.perf_counter()
        report = evaluate(model, images, labels, task.n_classes)
        elapsed = time.perf_counter() - start
        rows.append(
            ScaleRow(
                size=size,
                seconds=elapsed,
                throughput=size / elapsed if elapsed else 0.0,
                peak_mib=_peak_mib(model, task, size),
                accuracy=report.accuracy,
            )
        )
    return rows


def main() -> None:
    print("Synthetic oriented-stroke images; small from-scratch CNN.")
    print("Pure-Python inference; not benchmarked against compiled tensor libraries.")
    print(f"{'images':>8}  {'seconds':>9}  {'img/sec':>9}  {'peak MiB':>9}  {'accuracy':>9}")
    for row in run_scale():
        print(
            f"{row.size:>8d}  {row.seconds:>9.3f}  {row.throughput:>9.1f}  "
            f"{row.peak_mib:>9.2f}  {row.accuracy:>9.3f}"
        )


if __name__ == "__main__":
    main()
