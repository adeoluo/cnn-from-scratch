# ADR-0003: A translation task so that convolution actually matters

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
To show what a convolutional network buys, the task must reward translation invariance. An early
version drew each shape at a fixed location; a linear model solved it perfectly, because fixed
pixels were reliably informative, and the convolutional network had nothing to prove.

## Decision
Place the discriminative motif (a short oriented stroke) at a random position on each image. The
class is the orientation, which can now appear anywhere. A convolutional filter detects the motif
regardless of position; a linear model on raw pixels cannot.

## Consequences
- The comparison is honest and sharp: the linear baseline sits near chance (0.389 on three
  classes) while the convolutional network reaches 0.967.
- The result is attributable to translation invariance specifically, not to capacity or depth.
- The task is synthetic; the claim is about the mechanism, not about natural image recognition.
