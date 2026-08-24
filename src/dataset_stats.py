#!/usr/bin/env python
"""Print ALICE dataset statistics, and check that the release files parse.

Run this first after cloning: it exercises the whole data-processing path on
every subtask and reports the numbers quoted in the paper.

    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --show-example --task ke
"""

from __future__ import annotations

import argparse
from collections import Counter


from data_utils.data_loader import (
    DataSource,
    SUBJECTS,
    TASKS,
    AliceLoader,
    load_question_meta,
    subject_of,
)


def raw_split_stats(source: DataSource) -> None:
    print("Raw splits")
    print(f"  {'split':<9}{'answers':>9}{'questions':>11}   subjects")
    for split in ("train", "test_ua", "test_uq"):
        records = source.read_json(f"{split}.json")
        questions = {record["question_id"] for record in records}
        by_subject = Counter(subject_of(record["question_id"]) for record in records)
        breakdown = "  ".join(f"{s}={by_subject.get(s, 0)}" for s in SUBJECTS)
        print(f"  {split:<9}{len(records):>9}{len(questions):>11}   {breakdown}")

    meta = load_question_meta(source)
    print(f"  question_meta entries: {len(meta)}")
    print()


def task_stats(source: DataSource) -> None:
    print("Scoring instances per subtask (after expanding multi-label items)")
    print(f"  {'task':<6}{'train':>9}{'val':>8}{'test_ua':>10}{'test_uq':>10}{'levels/item':>14}")
    for task in TASKS:
        loader = AliceLoader(data_dir=source, task=task)
        sizes = {name: len(items) for name, items in loader.splits.items()}
        all_items = [i for split in loader.splits.values() for i in split]
        rubric_counts = Counter(instance.num_rubrics for instance in all_items)
        spread = ", ".join(f"{k}:{v}" for k, v in sorted(rubric_counts.items()))
        print(
            f"  {task:<6}{sizes['train']:>9}{sizes['val']:>8}"
            f"{sizes['test_ua']:>10}{sizes['test_uq']:>10}   {spread}"
        )
    print()


def label_stats(source: DataSource) -> None:
    print("Global level distribution (all splits pooled)")
    for task in TASKS:
        loader = AliceLoader(data_dir=source, task=task)
        levels = Counter(
            instance.global_level for split in loader.splits.values() for instance in split
        )
        total = sum(levels.values())
        share = "  ".join(
            f"{level}={count} ({count / total:.1%})" for level, count in sorted(levels.items())
        )
        print(f"  {task:<4} {share}")
    print()


def subject_coverage(source: DataSource) -> None:
    print("Subject coverage per subtask")
    print(f"  {'task':<6}" + "".join(f"{s:>10}" for s in SUBJECTS))
    for task in TASKS:
        loader = AliceLoader(data_dir=source, task=task)
        counts = Counter(
            instance.subject for split in loader.splits.values() for instance in split
        )
        print(f"  {task:<6}" + "".join(f"{counts.get(s, 0):>10}" for s in SUBJECTS))
    print("  (mathematics is annotated for learning performance only)")
    print()


def show_example(source: DataSource, task: str) -> None:
    loader = AliceLoader(data_dir=source, task=task)
    instance = loader.test_ua[0]
    print(f"Example instance ({task})")
    print(f"  id            {instance.id}")
    print(f"  question_id   {instance.question_id}  [{instance.subject}]")
    if instance.item:
        print(f"  item          {instance.item}")
    print(f"  question      {instance.question[:160]}")
    print(f"  solution      {instance.sample_solution[:160]}")
    print(f"  answer        {instance.answer.strip()[:160]}")
    print(f"  gold level    {instance.level} (position)  /  {instance.global_level} (global)")
    for position, rubric in enumerate(instance.rubrics):
        marker = ">" if position == instance.level else " "
        print(f"   {marker} [{position}] {rubric[:150]}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="directory or .zip with the release files; auto-detected by default")
    parser.add_argument("--show-example", action="store_true")
    parser.add_argument("--task", choices=TASKS, default="lp")
    args = parser.parse_args(argv)

    source = DataSource(args.data_dir)
    print(f"ALICE benchmark — {source.path}\n")
    raw_split_stats(source)
    task_stats(source)
    label_stats(source)
    subject_coverage(source)
    if args.show_example:
        show_example(source, args.task)


if __name__ == "__main__":
    main()
