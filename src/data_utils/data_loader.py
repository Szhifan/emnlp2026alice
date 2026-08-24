"""Loading and preprocessing for the ALICE benchmark.

This module turns the raw release files in ``alice_data_public/`` into flat
lists of scoring *instances*. It depends only on the standard library so that
the data can be inspected without installing torch/transformers.

Raw layout::

    alice_data_public/
        train.json, test_ua.json, test_uq.json
        question_meta/{bio,chemie,math,physik}.json

A record in a split file looks like::

    {"id": "train_0",
     "question_id": "physik_8",
     "answer": "Keine Ahnung\\n",
     "learning_performance": "0",
     "knowledge_elements": {"Umwandlung": "0"},
     "skills": {"Reasoning": "0"}}

The matching entry in ``question_meta/physik.json`` under key ``physik_8``
carries the ``prompt``, the ``solution`` and, per dimension, a rubric set that
maps a score level to a ``{"rule": ..., "description": ...}`` pair.

Three scoring subtasks are exposed:

``lp``
    Learning performance: one holistic score per answer.
``ke``
    Knowledge elements: one score per knowledge element addressed by the
    question, so a single answer expands into several instances.
``sk``
    Skills: same multi-label structure as ``ke``.

Mathematics questions are annotated for ``lp`` only.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "SUBJECTS",
    "TASKS",
    "TASK_DIMENSION",
    "GLOBAL_NUM_LEVELS",
    "Instance",
    "AliceLoader",
    "subject_of",
    "load_question_meta",
    "DataSource",
]

SUBJECTS: tuple[str, ...] = ("bio", "chemie", "math", "physik")

TASKS: tuple[str, ...] = ("lp", "ke", "sk")

#: Subtask -> field name used in the raw JSON files and in ``question_meta``.
TASK_DIMENSION: dict[str, str] = {
    "lp": "learning_performance",
    "ke": "knowledge_elements",
    "sk": "skills",
}

#: Size of the *global* level scheme per subtask. Individual questions may
#: define only a subset of these levels; see :class:`AliceLoader` for how the
#: two indexings relate.
GLOBAL_NUM_LEVELS: dict[str, int] = {"lp": 3, "ke": 4, "sk": 3}

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The release ships as a zip. Either form works: unzip it and the extracted
#: directory is picked up, or leave it packed and it is read in place.
DEFAULT_DATA_DIR = REPO_ROOT / "alice_data_public"
DEFAULT_DATA_ZIP = REPO_ROOT / "alice_data_public.zip"


def subject_of(question_id: str) -> str:
    """Return the subject prefix of an anonymised question id."""
    subject = question_id.split("_", 1)[0]
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject in question_id {question_id!r}")
    return subject


class DataSource:
    """Reads the release files, whether extracted or still zipped.

    Args:
        path: A directory, a ``.zip``, or ``None`` to auto-detect — an
            extracted ``alice_data_public/`` next to the repo root is
            preferred, otherwise ``alice_data_public.zip``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.is_dir() else DEFAULT_DATA_ZIP
        path = Path(path)

        if path.is_dir():
            self.zipped = False
        elif path.is_file() and zipfile.is_zipfile(path):
            self.zipped = True
        else:
            raise FileNotFoundError(
                f"No ALICE data at {path}. Expected the directory "
                f"{DEFAULT_DATA_DIR.name}/ or the archive {DEFAULT_DATA_ZIP.name} "
                f"in {REPO_ROOT}, or pass --data-dir explicitly."
            )
        self.path = path

    def read_json(self, name: str) -> Any:
        """Load one release file, e.g. ``"train.json"`` or ``"question_meta/bio.json"``."""
        if not self.zipped:
            with open(self.path / name, encoding="utf-8") as fh:
                return json.load(fh)
        with zipfile.ZipFile(self.path) as archive:
            return json.loads(archive.read(self._member(archive, name)).decode("utf-8"))

    @staticmethod
    def _member(archive: zipfile.ZipFile, name: str) -> str:
        """Find ``name`` inside the archive, tolerating a wrapping directory.

        Finder-made zips nest everything under ``alice_data_public/`` and add
        ``__MACOSX`` resource forks, so match on the trailing path instead of
        assuming a layout.
        """
        candidates = [
            entry
            for entry in archive.namelist()
            if not entry.startswith("__MACOSX/")
            and not Path(entry).name.startswith("._")
            and (entry == name or entry.endswith("/" + name))
        ]
        if not candidates:
            raise KeyError(f"{name} not found in {archive.filename}")
        # shortest path wins, so a top-level file beats a nested duplicate
        return min(candidates, key=len)

    def __repr__(self) -> str:
        return f"DataSource({self.path}, zipped={self.zipped})"


def load_question_meta(data_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load every ``question_meta`` file into one ``question_id -> meta`` dict."""
    source = data_dir if isinstance(data_dir, DataSource) else DataSource(data_dir)
    meta: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        meta.update(source.read_json(f"question_meta/{subject}.json"))
    return meta


@dataclass
class Instance:
    """One (answer, rubric set) scoring decision.

    ``level`` indexes into ``rubrics`` and is what models are trained to
    predict. ``global_level`` is the raw annotation on the dataset-wide level
    scheme, kept so that predictions can be mapped back to the published
    label space.
    """

    id: str
    question_id: str
    subject: str
    task: str
    answer: str
    question: str
    sample_solution: str
    rubrics: list[str]
    level: int
    global_level: int
    #: Name of the knowledge element / skill being scored; empty for ``lp``.
    item: str = ""
    #: ``level`` position -> raw level string from ``question_meta``.
    level_names: list[str] = field(default_factory=list)

    @property
    def num_rubrics(self) -> int:
        return len(self.rubrics)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["num_rubrics"] = self.num_rubrics
        return d


def _render_rubric(task: str, item: str, level_name: str, entry: dict[str, str]) -> str:
    """Render one rubric level as the string the model sees.

    Follows the input format documented in the paper appendix: the holistic
    ``lp`` levels are rendered as ``rule: description``, while ``ke``/``sk``
    levels are prefixed with the element/skill name, since the same level
    label means different things for different elements.
    """
    rule = (entry.get("rule") or "").strip()
    description = (entry.get("description") or "").strip()
    if task == "lp":
        prefix = rule or level_name
    else:
        prefix = f"{item} ({rule})" if rule else item
    return f"{prefix}: {description}" if description else prefix


def _sorted_levels(rubric_set: dict[str, Any]) -> list[str]:
    """Return the level keys of a rubric set in ascending numeric order."""
    return sorted(rubric_set, key=float)


class AliceLoader:
    """Load and preprocess the ALICE splits for one subtask.

    Args:
        data_dir: Directory or ``.zip`` holding the release files. ``None``
            auto-detects the release shipped with the repo.
        task: One of ``"lp"``, ``"ke"``, ``"sk"``.
        subjects: Restrict to these subjects; ``None`` keeps all four.
        val_frac: Fraction of ``train`` held out as a validation split. The
            split is instance-wise, matching the paper: it is question-disjoint
            from Test-UQ by construction but shares questions with Test-UA.
        train_frac: Sub-sample the training split before the validation cut,
            for quick smoke runs.
        seed: Seed for the sub-sampling and validation split.

    Attributes:
        train, val, test_ua, test_uq: ``list[Instance]``.

    Level indexing
        Some questions omit intermediate levels, e.g. a knowledge element may
        define only ``{0, 1, 3}``. ``Instance.level`` is the *position* of the
        gold level inside that question's own rubric list (``{0, 1, 3}`` ->
        ``{0, 1, 2}``), which is what rubric-retrieval scores over.
        ``Instance.global_level`` keeps the raw value (``3``), which is what
        the fixed-head sequence-classification baseline predicts.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        task: str = "lp",
        subjects: Sequence[str] | None = None,
        val_frac: float = 0.1,
        train_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if not 0.0 < train_frac <= 1.0:
            raise ValueError("train_frac must be in (0, 1]")
        if not 0.0 <= val_frac < 1.0:
            raise ValueError("val_frac must be in [0, 1)")

        self.source = data_dir if isinstance(data_dir, DataSource) else DataSource(data_dir)
        self.task = task
        self.subjects = tuple(subjects) if subjects else SUBJECTS
        self.seed = seed
        self.meta = load_question_meta(self.source)

        train = self._build("train.json")
        self.test_ua = self._build("test_ua.json")
        self.test_uq = self._build("test_uq.json")

        rng = random.Random(seed)
        rng.shuffle(train)
        if train_frac < 1.0:
            train = train[: max(1, int(len(train) * train_frac))]
        n_val = int(len(train) * val_frac)
        self.val = train[:n_val]
        self.train = train[n_val:]

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self, filename: str) -> list[Instance]:
        records = self.source.read_json(filename)
        instances: list[Instance] = []
        for record in records:
            if subject_of(record["question_id"]) not in self.subjects:
                continue
            instances.extend(self._expand(record))
        return instances

    def _expand(self, record: dict[str, Any]) -> list[Instance]:
        """Turn one raw record into one (``lp``) or several (``ke``/``sk``) instances."""
        question_id = record["question_id"]
        meta = self.meta[question_id]
        common = dict(
            question_id=question_id,
            subject=subject_of(question_id),
            task=self.task,
            answer=record["answer"],
            question=meta.get("prompt", ""),
            sample_solution=meta.get("solution", ""),
        )

        if self.task == "lp":
            rubric_set = meta.get("learning_performance") or {}
            instance = self._make(
                instance_id=record["id"],
                item="",
                rubric_set=rubric_set,
                # Labels are strings for most subjects but integers for math,
                # so normalise before looking anything up.
                gold=str(record["learning_performance"]),
                **common,
            )
            return [instance] if instance else []

        dimension = TASK_DIMENSION[self.task]
        annotations = record.get(dimension) or {}
        meta_sets = meta.get(dimension) or {}
        prefix = "ke" if self.task == "ke" else "sk"

        instances = []
        for position, (item, gold) in enumerate(annotations.items()):
            instance = self._make(
                instance_id=f"{record['id']}_{prefix}{position}",
                item=item,
                rubric_set=meta_sets.get(item) or {},
                gold=str(gold),
                **common,
            )
            if instance:
                instances.append(instance)
        return instances

    def _make(
        self,
        instance_id: str,
        item: str,
        rubric_set: dict[str, Any],
        gold: str,
        **common: Any,
    ) -> Instance | None:
        if not rubric_set:
            return None
        level_names = _sorted_levels(rubric_set)
        if gold not in rubric_set:
            # An annotation outside the question's own rubric set cannot be
            # scored against it; the released splits contain no such case.
            return None
        rubrics = [
            _render_rubric(self.task, item, name, rubric_set[name]) for name in level_names
        ]
        return Instance(
            id=instance_id,
            rubrics=rubrics,
            level=level_names.index(gold),
            global_level=int(float(gold)),
            item=item,
            level_names=level_names,
            **common,
        )

    # ------------------------------------------------------------------
    # access helpers
    # ------------------------------------------------------------------
    @property
    def splits(self) -> dict[str, list[Instance]]:
        return {
            "train": self.train,
            "val": self.val,
            "test_ua": self.test_ua,
            "test_uq": self.test_uq,
        }

    @property
    def num_global_levels(self) -> int:
        """Label-space size for the fixed-head sequence-classification baseline."""
        return GLOBAL_NUM_LEVELS[self.task]

    def as_dicts(self, split: str) -> list[dict[str, Any]]:
        return [instance.to_dict() for instance in self.splits[split]]

    def to_hf_dataset(self, split: str):
        """Convert one split to a ``datasets.Dataset`` (requires ``datasets``)."""
        from datasets import Dataset

        return Dataset.from_list(self.as_dicts(split))

    def __repr__(self) -> str:
        sizes = ", ".join(f"{name}={len(items)}" for name, items in self.splits.items())
        return f"AliceLoader(task={self.task!r}, {sizes})"


def expand_to_pairs(instances: Iterable[Instance]) -> Iterator[dict[str, Any]]:
    """Yield one (answer, single rubric) pair per candidate level.

    Used by encoders that score each candidate independently rather than
    consuming the whole rubric set in one forward pass.
    """
    for instance in instances:
        for position, rubric in enumerate(instance.rubrics):
            pair = instance.to_dict()
            pair["rubric"] = rubric
            pair["rubric_level"] = position
            pair["labels"] = int(position == instance.level)
            yield pair
