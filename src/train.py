#!/usr/bin/env python
"""Fine-tune a model on one ALICE subtask, then evaluate on both test splits.

Examples::

    # rubric-retrieval with a masked LM, answer + rubric only
    python scripts/train.py --task lp --approach rubric-retrieval \
        --base-model jhu-clsp/mmBERT-base --input-fields ar \
        --lr 2e-5 --epochs 4 --batch-size 16 \
        --output-dir runs/mmbert_lp_ar

    # rubric-retrieval with a causal LM encoder (QLoRA), full context
    python scripts/train.py --task ke --approach rubric-retrieval \
        --base-model meta-llama/Llama-3.2-3B --input-fields qsar \
        --use-lora --use-4bit --lr 1e-4 --epochs 3 \
        --batch-size 4 --grad-accum 8 \
        --output-dir runs/llama3b_ke_qsar

    # sequence-classification baseline: solution + answer
    python scripts/train.py --task lp --approach seq-classification \
        --base-model jhu-clsp/mmBERT-base --input-fields sa \
        --lr 2e-5 --epochs 4 --output-dir runs/mmbert_lp_seqcls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from data_utils.data_loader import AliceLoader
from data_utils.data_prep import get_tokenizer
from metrics import evaluation_report, save_report
from modelling.modelling_utils import (
    APPROACHES,
    ModelConfig,
    build_collate_fn,
    build_model,
    encode_split,
    is_causal_lm,
    predict,
)
from utils import default_device, save_predictions, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default=None, help="directory or .zip with the release files; auto-detected by default")
    data.add_argument("--task", choices=["lp", "ke", "sk"], default="lp")
    data.add_argument(
        "--subjects", nargs="*", default=None, help="restrict to e.g. bio physik"
    )
    data.add_argument("--train-frac", type=float, default=1.0, help="sub-sample train for smoke runs")
    data.add_argument("--val-frac", type=float, default=0.1)

    model = parser.add_argument_group("model")
    model.add_argument("--approach", choices=APPROACHES, default="rubric-retrieval")
    model.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    model.add_argument(
        "--input-fields",
        default="ar",
        help="letters from q(uestion) s(ample solution) a(nswer) r(ubric), e.g. 'qsar'",
    )
    model.add_argument("--input-format", choices=["auto", "structured", "sep"], default="auto")
    model.add_argument("--pool-type", choices=["auto", "avg", "cls", "last"], default="auto")
    model.add_argument("--max-length", type=int, default=512)
    model.add_argument("--use-lora", action="store_true")
    model.add_argument("--use-4bit", action="store_true", help="QLoRA; requires CUDA + bitsandbytes")
    model.add_argument("--lora-rank", type=int, default=64)
    model.add_argument("--lora-alpha", type=int, default=64)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--lr", type=float, default=2e-5)
    optim.add_argument("--epochs", type=int, default=4)
    optim.add_argument("--batch-size", type=int, default=16)
    optim.add_argument("--eval-batch-size", type=int, default=None)
    optim.add_argument("--grad-accum", type=int, default=1)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--warmup-ratio", type=float, default=0.01)
    optim.add_argument("--clip-norm", type=float, default=1.0)
    optim.add_argument("--bf16", action="store_true")

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--test-only", action="store_true", help="evaluate --output-dir without training")
    return parser.parse_args(argv)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        base_model=args.base_model,
        approach=args.approach,
        task=args.task,
        input_fields=args.input_fields,
        input_format=args.input_format,
        pool_type=args.pool_type,
        max_length=args.max_length,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump({**vars(args), **config.to_dict()}, fh, indent=2, default=str)

    loader = AliceLoader(
        data_dir=args.data_dir,
        task=args.task,
        subjects=args.subjects,
        val_frac=args.val_frac,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(loader, flush=True)

    tokenizer = get_tokenizer(config.base_model)
    collate_fn = build_collate_fn(config, tokenizer)
    encoded = {
        name: encode_split(instances, tokenizer, config) for name, instances in loader.splits.items()
    }

    checkpoint = str(output_dir) if args.test_only else None
    model = build_model(config, checkpoint=checkpoint, pad_token_id=tokenizer.pad_token_id)

    if not args.test_only:
        from transformers import Trainer, TrainingArguments

        causal = is_causal_lm(config.base_model)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.clip_norm,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=args.bf16,
            optim="paged_adamw_32bit" if (causal and args.use_4bit) else "adamw_torch",
            gradient_checkpointing=causal,
            gradient_checkpointing_kwargs={"use_reentrant": False} if causal else None,
            # our collators consume the raw columns, so keep them
            remove_unused_columns=False,
            label_names=["labels"],
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            seed=args.seed,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["val"],
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )
        print(f"training on {len(encoded['train'])} instances", flush=True)
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        model = trainer.model

    device = default_device()
    model.to(device)

    summary: dict[str, dict] = {}
    for split in ("test_ua", "test_uq"):
        rows, elapsed = predict(
            model,
            encoded[split],
            collate_fn,
            batch_size=args.eval_batch_size or args.batch_size,
            device=device,
            desc=split,
        )
        report = evaluation_report(
            [row["gold_global"] for row in rows],
            [row["pred_global"] for row in rows],
            [row["subject"] for row in rows],
        )
        report["seconds_per_instance"] = elapsed / max(1, len(rows))
        save_predictions(rows, output_dir / "predictions" / f"{split}.csv")
        save_report(report, output_dir / "predictions" / f"{split}_metrics.json")
        summary[split] = report["overall"]
        overall = report["overall"]
        print(
            f"{split}: acc={overall['accuracy']:.4f} "
            f"f1_macro={overall['f1_macro']:.4f} qwk={overall['qwk']:.4f}",
            flush=True,
        )

    save_report(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
