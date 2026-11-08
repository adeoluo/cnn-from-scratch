# AGENTS.md

Working notes for coding agents and contributors in **CNN From Scratch**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A convolutional neural network and its reverse-mode backpropagation, built from scratch in Python with every forward and backward pass written by hand and every gradient verified against finite differences.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/cnn_from_scratch/__init__.py`: CNN From Scratch - a convolutional neural network and its autodiff, from scratch.
- `src/cnn_from_scratch/cli.py`: Command line interface for the convolutional network.
- `src/cnn_from_scratch/data.py`: A deterministic synthetic image-classification task with spatial structure.
- `src/cnn_from_scratch/evaluation.py`: Evaluation: accuracy, confusion matrix, and a linear baseline.
- `src/cnn_from_scratch/layers.py`: Neural network layers with explicit forward and backward passes.
- `src/cnn_from_scratch/model.py`: A sequential container, a small convolutional network, and the loss.
- `src/cnn_from_scratch/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/cnn_from_scratch/service.py`: A standard-library HTTP service exposing the trained convolutional classifier.
- `src/cnn_from_scratch/training.py`: Training the convolutional network with an Adam optimizer from scratch.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the HTTP service |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
