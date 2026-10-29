# Model / System Card - CNN From Scratch

## Overview
- **What it is:** a from-scratch convolutional neural network with explicit backpropagation,
  trained on a synthetic oriented-stroke classification task. No numerical libraries.
- **Base model:** none. Parameters are initialized and trained from scratch.
- **Version and date:** 0.1.0, 2026.

## Intended use
- **Primary use case:** a readable, reproducible demonstration of convolution, pooling,
  backpropagation, and the translation invariance that distinguishes a CNN from a linear model.
- **Primary users:** engineers and researchers learning convolutional network internals.

## Out-of-scope / prohibited use
- Not an image recognizer for real photographs; the task is synthetic shapes.
- Not a performance reference; pure-Python convolution is slow by design.
- The accuracy numbers describe a shape task and must not be cited as an image benchmark.

## Performance
- Held-out accuracy 0.967 versus 0.389 for a logistic-regression baseline.
- Gradients verified against finite differences to better than 1e-4.
- Full method and tables: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- Tiny, single-block network on small synthetic images; no claims about scale or natural images.
- Pure-Python convolution is slow; not a substitute for an optimized kernel.

## Ethical considerations
- No personal data, no network access, no external model. Inputs are synthetic grayscale images.
