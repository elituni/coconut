"""Command-line entry points for training, evaluation, and a single question."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import transformers

from .data import load_examples
from .inference import evaluate, generate
from .training import TrainConfig, load_model_tokenizer, select_device, train, write_json


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Run a JSON-configured curriculum")
    training.add_argument("--config", default="configs/prosqa.json")
    training.add_argument("--model", help="Override initial model with a local checkpoint")
    for command in ("evaluate", "generate"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--device", default="auto")
        sub.add_argument(
            "--max-thoughts",
            type=int,
            default=None,
            help="Default: 0 for a recorded stage-0 checkpoint, otherwise 12",
        )
        sub.add_argument("--max-new-tokens", type=int, default=50)
        sub.add_argument("--latent-policy", choices=("adaptive", "fixed"), default="adaptive")
        sub.add_argument(
            "--answer-format",
            choices=("marker", "plain"),
            default="marker",
            help="Use plain explicitly for older checkpoints without <ans>",
        )
        sub.add_argument(
            "--no-eot-newline",
            action="store_true",
            help="Legacy experiment: omit the newline after <eot>",
        )
        if command == "evaluate":
            sub.add_argument("--data", required=True)
            sub.add_argument("--output", required=True)
            sub.add_argument(
                "--limit", type=int, help="Evaluate the first N examples; labeled a subset"
            )
        else:
            sub.add_argument("--question", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        if args.model:
            config.model = args.model
        train(config)
        return
    if args.command == "evaluate":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive.")
        if Path(args.output).exists():
            raise ValueError("Evaluation output already exists; choose a new path.")
    base, tokenizer = load_model_tokenizer(
        args.checkpoint, require_answer_marker=args.answer_format == "marker"
    )
    base.to(select_device(args.device))
    max_thoughts = args.max_thoughts
    if max_thoughts is None:
        metadata_path = Path(args.checkpoint) / "coconut_config.json"
        stage = (
            json.loads(metadata_path.read_text()).get("stage") if metadata_path.exists() else None
        )
        max_thoughts = 0 if stage == 0 else 12
    options = {
        "max_thoughts": max_thoughts,
        "max_new_tokens": args.max_new_tokens,
        "latent_policy": args.latent_policy,
        "eot_newline": not args.no_eot_newline,
    }
    if args.command == "generate":
        print(json.dumps(asdict(generate(base, tokenizer, args.question, **options)), indent=2))
    else:
        examples = load_examples(args.data)
        result = evaluate(
            base, tokenizer, examples[: args.limit], answer_format=args.answer_format, **options
        )
        result.update(
            {
                "checkpoint": args.checkpoint,
                "data": args.data,
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "subset": args.limit is not None and args.limit < len(examples),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        )
        write_json(args.output, result)
        print(f"Exact match: {result['correct']}/{result['total']} ({result['accuracy']:.2%})")


if __name__ == "__main__":
    main()
