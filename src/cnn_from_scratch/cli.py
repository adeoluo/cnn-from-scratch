"""Command line interface for the convolutional network."""

from __future__ import annotations

import argparse

from cnn_from_scratch.data import CLASS_NAMES, ImageTask, test_data, train_data
from cnn_from_scratch.evaluation import LogisticRegression, evaluate, mean_confidence
from cnn_from_scratch.model import CNNConfig
from cnn_from_scratch.training import TrainConfig, train


def _train(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    task = ImageTask(size=args.size, noise=args.noise)
    cfg = CNNConfig(image_size=task.size, n_classes=task.n_classes)
    result = train(task, cfg, TrainConfig(epochs=args.epochs, seed=args.seed))
    return task, result


def _cmd_train(args: argparse.Namespace) -> None:
    task, result = _train(args)
    test_i, test_l = test_data(task, seed=args.seed)
    train_i, train_l = train_data(task, seed=args.seed)
    report = evaluate(result.model, test_i, test_l, task.n_classes)
    baseline = LogisticRegression.fit(train_i, train_l, task.n_classes).evaluate(test_i, test_l)
    print(f"Trained {args.epochs} epochs on the oriented-stroke task ({task.n_classes} classes).")
    print(f"  convolutional network: {report.format()}")
    print(f"  logistic-regression baseline: accuracy={baseline:.4f}")
    print(f"  mean confidence in true class: {mean_confidence(result.model, test_i, test_l):.4f}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    task, result = _train(args)
    test_i, test_l = test_data(task, seed=args.seed)
    report = evaluate(result.model, test_i, test_l, task.n_classes)
    print(report.format())
    print("Confusion matrix (rows = true, columns = predicted):")
    print("        " + "  ".join(f"{n[:4]:>4}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, report.confusion, strict=True):
        print(f"  {name[:6]:>6}  " + "  ".join(f"{c:>4}" for c in row))


def _cmd_serve(args: argparse.Namespace) -> None:
    from cnn_from_scratch.service import serve

    serve(host=args.host, port=args.port)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cnn-from-scratch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train and compare to a linear baseline")
    _add_common(train_p)

    eval_p = sub.add_parser("evaluate", help="train, then report accuracy and a confusion matrix")
    _add_common(eval_p)

    serve_p = sub.add_parser("serve", help="start the HTTP service")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    handlers = {"train": _cmd_train, "evaluate": _cmd_evaluate, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
