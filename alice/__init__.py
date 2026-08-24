"""ALICE: rubric-based short-answer scoring for German STEM education.

Typical use::

    from alice.data import AliceLoader

    loader = AliceLoader(task="lp")
    print(loader)                       # split sizes
    instance = loader.test_ua[0]
    print(instance.answer, instance.rubrics, instance.level)

``alice.data`` is pure standard library. The modelling modules
(``alice.encoding``, ``alice.models``) additionally require torch and
transformers and are therefore imported lazily.
"""

from alice.data import (
    GLOBAL_NUM_LEVELS,
    SUBJECTS,
    TASK_DIMENSION,
    TASKS,
    AliceLoader,
    Instance,
    load_question_meta,
    subject_of,
)

__all__ = [
    "AliceLoader",
    "Instance",
    "SUBJECTS",
    "TASKS",
    "TASK_DIMENSION",
    "GLOBAL_NUM_LEVELS",
    "load_question_meta",
    "subject_of",
]

__version__ = "1.0.0"
