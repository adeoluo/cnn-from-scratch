# Evaluation Report - CNN From Scratch

> Reproduce with `python -m cnn_from_scratch train` and `... evaluate`. All runs are deterministic
> given a seed; numbers below use seed 0 and the default configuration.

## Setup

- **Task:** classify the orientation (horizontal, vertical, diagonal) of a short stroke placed at
  a random position on a noisy 8x8 image.
- **Splits:** 120 training and 90 held-out images, balanced across classes, generated from
  separate seed streams.
- **Network:** one Conv2D layer of 6 filters (3x3), ReLU, 2x2 max pooling, flatten, dense softmax.
- **Optimizer:** Adam from scratch, 15 epochs, learning rate 0.01.
- **Baseline:** multinomial logistic regression on raw pixels, trained by the same cross-entropy
  gradient.
- **Hardware / cost:** CPU only; training completes in a few seconds. No accelerators, no external
  services, no API cost.

## Correctness: gradients verified against finite differences

For every layer (convolution weights and biases, dense weights and biases) the analytic gradient
is compared to a central finite-difference estimate on a small network. All agree to better than
`1e-4`; the worst observed difference is around `1e-11`. This is enforced as a test.

## Headline results

| Metric | Convolutional network | Logistic-regression baseline | Delta |
|---|---:|---:|---:|
| held-out accuracy | **0.967** | 0.389 | +0.578 |
| mean confidence in true class | 0.944 | - | - |

The convolutional network beats a linear model on the same pixels by nearly 58 points. The linear
model sits just above the 0.333 chance level because the motif moves and no fixed pixel is
reliably informative; the convolutional filter detects the motif wherever it appears.

## Training dynamics

Training accuracy rises from about 0.36 after the first epoch to 1.0 within a dozen epochs, while
held-out accuracy settles near 0.97. Loss decreases monotonically under Adam.

## Service performance

The `/classify` handler is a single forward pass with no training on the request path. Under a
concurrent stress benchmark (`uv run python benchmarks/stress.py`, 5000 requests, 8 workers) it
sustains roughly 1,400 requests per second with p50 near 0.32 ms and p95 near 2.7 ms.

## Reproducibility

- Deterministic given a seed: two runs of the default configuration produce identical weights.
- No network access, no external data, no accelerators.
- Regenerate every number with the CLI and `benchmarks/benchmark.py`.
