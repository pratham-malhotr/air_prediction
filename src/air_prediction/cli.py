"""Command line entry points shared by local and Colab workflows."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import FILENAME, download_dataset, load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Project 19 — Phase 1 hourly PM2.5 forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download UCI 381 with a checksum and provenance")
    download.add_argument("--data", default=f"data/raw/{FILENAME}")
    for name in ("eda", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default=f"data/raw/{FILENAME}")
        sub.add_argument("--config", default="configs/phase1.toml")
        sub.add_argument("--output", default="artifacts/phase1/eda" if name == "eda" else "artifacts/phase1")
        if name == "train":
            sub.add_argument("--models", nargs="+", choices=["xgboost", "ridge"], default=["xgboost", "ridge"])
            sub.add_argument("--ablations", action="store_true", help="Compare feature groups on validation dates only")
    demo = commands.add_parser("demo", help="Run a separately labeled synthetic integration demo with Ridge")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--config", default="configs/demo.toml")
    predict = commands.add_parser("predict", help="Forecast the next target time from a saved local model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(download_dataset(args.data))
        elif args.command == "eda":
            from .eda import run_eda
            print(run_eda(load_dataset(args.data), args.output, load_config(args.config).validation_start))
        elif args.command == "train":
            from .experiment import run_experiment
            result = run_experiment(args.data, load_config(args.config), args.output, tuple(args.models), args.ablations)
            print(result["metrics"].to_string(index=False))
            print(f"Results saved to {result['output']}")
        elif args.command == "demo":
            from .demo import make_demo
            from .experiment import run_experiment
            path = make_demo(Path(args.output) / "synthetic.csv")
            result = run_experiment(path, load_config(args.config), args.output, ("ridge",))
            print("SYNTHETIC INTEGRATION DEMO — NOT PROJECT RESULTS")
            print(result["metrics"].to_string(index=False))
        else:
            from .predict import predict_latest
            print(json.dumps(predict_latest(args.model, args.data), indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
