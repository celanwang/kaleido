"""One entry point for repository experiments and standalone deployed inference."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kaleido", description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="explicit source checkout for repository commands",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list frozen formal and smoke recipes")
    commands.add_parser("doctor", help="check full-reproduction dependency pins without downloads")
    verify = commands.add_parser("verify", help="validate the reviewed artifact inventory")
    verify.add_argument("--deep", action="store_true", help="also validate NPZ array hashes")
    run = commands.add_parser("run", help="execute one recipe")
    run.add_argument("stage", choices=[f"p{i}" for i in range(6)])
    run.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs/cli"))
    replay = commands.add_parser(
        "reproduce", help="run recipes in isolated processes; failures retain evidence"
    )
    replay.add_argument(
        "--stages",
        nargs="+",
        choices=[f"p{i}" for i in range(6)],
        default=[f"p{i}" for i in range(6)],
    )
    replay.add_argument("--profile", choices=("formal", "smoke"), default="smoke")
    replay.add_argument("--output", type=Path, default=Path("runs/reproduction"))
    export = commands.add_parser(
        "export", help="export a trained P3 decoder as a verified CPU bundle"
    )
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--tokenizer", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser(
        "predict", help="serve local JSON-lines without a repository checkout"
    )
    predict.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    def emit(value):
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        if args.command == "predict":
            import sys

            import torch

            from kaleido.deployment.runtime import Predictor, serve

            torch.set_num_threads(1)
            return serve(Predictor(args.bundle), sys.stdin.buffer, sys.stdout)
        if args.command == "export":
            from kaleido.deployment.export import export_decoder

            path = export_decoder(args.weights, args.tokenizer, args.output, root=args.root)
            emit({"status": "completed", "bundle": str(path)})
            return 0
        from dataclasses import asdict

        from kaleido.repro.integrity import verify_repository
        from kaleido.repro.registry import RECIPES, doctor, repository, reproduce, run_recipe

        if args.command == "list":
            emit({k: asdict(v) for k, v in RECIPES.items()})
        elif args.command == "doctor":
            result = doctor(args.root)
            emit(result)
            return 0 if result["ok"] else 1
        elif args.command == "verify":
            root = repository(args.root)
            result = verify_repository(root, deep=args.deep)
            emit(result)
            return 0 if result["ok"] else 1
        elif args.command == "run":
            emit(
                run_recipe(
                    args.stage,
                    args.root,
                    args.output,
                    profile=args.profile,
                    config_path=args.config,
                    emit=emit,
                )
            )
        elif args.command == "reproduce":
            path = reproduce(
                args.root, args.output, stages=args.stages, profile=args.profile, emit=emit
            )
            emit({"status": "completed", "run": str(path.resolve()), "profile": args.profile})
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        emit({"status": "failed", "error": type(error).__name__, "message": str(error)})
        return 1
    return 0
