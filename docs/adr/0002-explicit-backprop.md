# ADR-0002: Explicit per-layer backpropagation, not a general autodiff graph

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The project needs gradients through convolution and pooling with no external library. A general
reverse-mode autodiff engine is one option. The other is to give each layer an explicit
``forward`` and ``backward`` method, which is the classic way convolutional networks are taught.

## Decision
Implement each layer (Conv2D, ReLU, MaxPool2D, Flatten, Dense) with its own forward and backward
pass. A layer caches whatever its backward needs during the forward pass, and a ``Sequential``
container chains the layers and propagates the gradient in reverse order.

## Consequences
- The convolution and pooling gradients are written out in full, which is exactly the mechanism
  worth understanding; nothing is hidden behind a graph.
- The code is specific to a feed-forward stack; branching architectures would need more
  machinery. That is an acceptable limit for a teaching-grade convolutional classifier.
- Every backward pass is validated against finite differences in the test suite, so the
  hand-written gradients are held to a numerical standard.
