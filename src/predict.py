#!/usr/bin/env python
"""Score an ALICE split with a checkpoint produced by ``scripts/train.py``.

The run configuration is read back from ``<checkpoint>/run_config.json`` so
that inference uses exactly the input fields, format and pooling the model was
trained with. Any of those can still be overridden on the command line.

Examples::

    python scripts/predict.py --checkpoint runs/mmbert_lp_ar --split test_uq
    python scripts/predict.py --checkpoint runs/llama3b_ke_qsar \
        --split test_ua --batch-size 8 --output-dir preds/llama3b_ke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from data_utils.data_loader import AliceLoader
from data_utils.data_prep import get_tokenizer
from metrics import evaluation_report, save_report
from modelling.modelling_utils import ModelConfig, build_collate_fn, build_model, encode_split, predict
from utils import default_device, save_predictions, set_seed

CONFIG_KEYS = (
    "base_model",
    "approach",
    "task",
    "input_fields",
    "input_format",
    "pool_type",
    "max_length",
    "use_lora",
    "use_4bit",
    "lora_rank",
    "lora_alpha",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help="directory or .zip with the release files; auto-detected by default")
    parser.add_argument(
        "--split", default="test_ua", choices=["train", "val", "test_ua", "test_uq"]
    )
    parser.add_argument("--output-dir", default=None, help="defaults to <checkpoint>/predictions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    for key in ("task", "base-model", "approach", "input-fields", "input-format", "pool-type"):
        parser.add_argument(f"--{key}", default=None, help="override the value in run_config.json")
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> ModelConfig:
    config_path = Path(args.checkpoint) / "run_config.json"
    stored = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        stored = {key: saved[key] for key in CONFIG_KEYS if key in saved}
    else:
        print(f"warning: {config_path} not found, falling back to CLI arguments", flush=True)

    overrides = {
        key: getattr(args, key)
        for key in CONFIG_KEYS
        if getattr(args, key, None) not in (None, "")
    }
    return ModelConfig(**{**stored, **overrides})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    config = load_config(args)

    loader = AliceLoader(data_dir=args.data_dir, task=config.task, seed=args.seed)
    instances = loader.splits[args.split]
    print(f"{args.split}: {len(instances)} instances ({config.task})", flush=True)

    tokenizer = get_tokenizer(config.base_model)
    dataset = encode_split(instances, tokenizer, config)
    collate_fn = build_collate_fn(config, tokenizer)

    model = build_model(config, checkpoint=args.checkpoint, pad_token_id=tokenizer.pad_token_id)
    device = default_device()
    model.to(device)

    rows, elapsed = predict(
        model, dataset, collate_fn, batch_size=args.batch_size, device=device, desc=args.split
    )
    report = evaluation_report(
        [row["gold_global"] for row in rows],
        [row["pred_global"] for row in rows],
        [row["subject"] for row in rows],
    )
    report["seconds_per_instance"] = elapsed / max(1, len(rows))

    output_dir = Path(args.output_dir or Path(args.checkpoint) / "predictions")
    save_predictions(rows, output_dir / f"{args.split}.csv")
    save_report(report, output_dir / f"{args.split}_metrics.json")

    overall = report["overall"]
    print(
        f"{args.split}: acc={overall['accuracy']:.4f} "
        f"f1_macro={overall['f1_macro']:.4f} qwk={overall['qwk']:.4f}",
        flush=True,
    )
    print(f"wrote {output_dir}/{args.split}.csv", flush=True)


if __name__ == "__main__":
    main()
