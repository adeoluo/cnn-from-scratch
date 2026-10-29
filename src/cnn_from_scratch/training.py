"""Training the convolutional network with an Adam optimizer from scratch."""

from __future__ import annotations

from dataclasses import dataclass, field

from cnn_from_scratch.data import ImageTask, train_data
from cnn_from_scratch.layers import Param
from cnn_from_scratch.model import CNNConfig, Sequential, build_cnn, cross_entropy_grad

Image = list[list[list[float]]]


@dataclass(frozen=True)
class TrainConfig:
    """Optimization hyperparameters."""

    epochs: int = 15
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 0


class Adam:
    """Adam over a fixed list of parameter tensors."""

    def __init__(self, params: list[Param], cfg: TrainConfig) -> None:
        self.params = params
        self.cfg = cfg
        self.t = 0
        self.m = [[0.0] * len(p.values) for p in params]
        self.v = [[0.0] * len(p.values) for p in params]

    def step(self) -> None:
        cfg = self.cfg
        self.t += 1
        b1t = 1.0 - cfg.beta1**self.t
        b2t = 1.0 - cfg.beta2**self.t
        for idx, p in enumerate(self.params):
            m = self.m[idx]
            v = self.v[idx]
            for i in range(len(p.values)):
                g = p.grad[i]
                m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g
                v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g
                m_hat = m[i] / b1t
                v_hat = v[i] / b2t
                p.values[i] -= cfg.learning_rate * m_hat / ((v_hat**0.5) + cfg.eps)


@dataclass
class TrainResult:
    """The trained network and its per-epoch loss and accuracy history."""

    model: Sequential
    history: list[tuple[int, float, float]] = field(default_factory=list)


def train(
    task: ImageTask,
    model_cfg: CNNConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> TrainResult:
    """Train the CNN on the image task, returning it with a loss and accuracy history."""

    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or CNNConfig(image_size=task.size, n_classes=task.n_classes)
    model = build_cnn(model_cfg, seed=train_cfg.seed)
    optimizer = Adam(model.parameters(), train_cfg)
    images, labels = train_data(task, seed=train_cfg.seed)
    history: list[tuple[int, float, float]] = []

    order = list(range(len(images)))
    rng_state = train_cfg.seed
    for epoch in range(1, train_cfg.epochs + 1):
        rng_state = _shuffle(order, rng_state)
        total_loss = 0.0
        correct = 0
        for idx in order:
            logits = model.forward(images[idx])
            loss, grad = cross_entropy_grad(logits, labels[idx])
            total_loss += loss
            if max(range(len(logits)), key=lambda c: logits[c]) == labels[idx]:
                correct += 1
            model.zero_grad()
            model.backward(grad)
            optimizer.step()
        history.append((epoch, total_loss / len(images), correct / len(images)))
    return TrainResult(model=model, history=history)


def _shuffle(order: list[int], state: int) -> int:
    """A deterministic in-place Fisher-Yates shuffle with an explicit LCG state."""

    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return state
