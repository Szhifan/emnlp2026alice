#!/usr/bin/env python
"""Zero-shot rubric scoring with a chat LLM.

Reproduces the prompting setup of the paper (Appendix "Zero-shot LLM
Prompts"): a German system prompt with a task-specific FOCUS line, a user
message carrying the rubric set and the optional context, and a JSON reply
holding a short justification and one integer score.

Three input configurations are compared in the paper:

``full``
    rubric set + question + sample solution + answer
``norub``
    rubric set + answer, no question and no sample solution
``labels``
    level names only (no rubric descriptions) + answer

Backends
    ``--backend openai`` talks to any OpenAI-compatible chat-completions
    endpoint, so it also drives vLLM or Ollama servers via ``--base-url``.
    Set the key in ``OPENAI_API_KEY``.

Example::

    export OPENAI_API_KEY=...
    python scripts/zero_shot_llm.py --task lp --split test_ua \
        --model gpt-4o-mini --input-config full \
        --output-dir runs/zeroshot_gpt4omini_lp_full
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alice.data import DEFAULT_DATA_DIR, AliceLoader, Instance
from alice.metrics import evaluation_report, save_report
from alice.utils import save_predictions

SYSTEM_PROMPT = """Sie werden die Antwort eines K12-Schuelers in MINT-Faechern bewerten.

ZIEL
Weisen Sie basierend auf dem bereitgestellten Rubrikensatz genau EINE ganze Zahl als Punktzahl zu. Sie muessen die Schuelerantwort intern anhand jeder Rubrik bewerten und dann die am besten passende Punktzahl auswaehlen.

{focus}

REGELN
- Beruecksichtigen Sie die Frage und die Beispielantworten nur, wenn sie bereitgestellt werden; sie sind optionale Hilfsmittel zum Verstaendnis, keine strikten Anforderungen.
- Eine Rubrik ist nur erfuellt, wenn ihre erforderlichen Kriterien erfuellt sind. Wenn eine Rubrik "eines von/von mehreren" verwendet, folgen Sie dieser Logik.
- Bewerten Sie ausschliesslich basierend auf der Uebereinstimmung zwischen der Schuelerantwort und den Rubrikkriterien (und der Frage, falls vorhanden). Ignorieren Sie den Stil, es sei denn, er wird von der Rubrik verlangt.
- Geben Sie nur eine ganze Zahl zurueck, die der hoechsten erfuellten Rubrik entspricht.

AUSGABE
Geben Sie NUR ein gueltiges JSON-Objekt mit genau diesen Feldern zurueck:
{{
  "reasoning": "1-3 praegnante Saetze, die erklaeren, warum diese Punktzahl am besten zur Rubrik passt.",
  "score": <Ganzzahl>
}}

WICHTIG
- Fuegen Sie KEINE zusaetzlichen Schluessel, Texte oder Formatierungen ausserhalb des JSON hinzu."""

TASK_FOCUS = {
    "lp": (
        "FOKUS: Bewerten Sie die GESAMTLEISTUNG des Schuelers bei der Bearbeitung der Aufgabe, "
        "einschliesslich Verstaendnis, Anwendung und Kommunikation des Wissens."
    ),
    "ke": (
        "FOKUS: Bewerten Sie die BEHERRSCHUNG SPEZIFISCHER WISSENSELEMENTE (z.B. chemische Reaktionen, "
        "physikalische Gesetze, mathematische Konzepte). Konzentrieren Sie sich auf die Korrektheit und "
        "Tiefe des fachlichen Verstaendnisses der jeweiligen Wissenskomponente."
    ),
    "sk": (
        "FOKUS: Bewerten Sie die ANWENDUNG SPEZIFISCHER FAEHIGKEITEN (z.B. Hypothesen formulieren, "
        "Experimente entwerfen, Daten interpretieren, Schlussfolgerungen ziehen). Konzentrieren Sie sich "
        "auf die Qualitaet der methodischen und prozeduralen Fertigkeiten, nicht nur auf das Fachwissen."
    ),
}

USER_PROMPT = """RUBRIKENSATZ
{rubrics}

FRAGE
{question}

BEISPIELLOESUNGEN
{sample_solutions}

SCHUELERANTWORT
{answer}"""

INPUT_CONFIGS = ("full", "norub", "labels")


def build_user_prompt(instance: Instance, input_config: str) -> str:
    """Render the user message for one instance.

    Rubric candidates are numbered by their position in the question's own
    rubric list, so the model's integer reply indexes ``instance.rubrics``
    directly.
    """
    if input_config == "labels":
        # Level names only: strip the descriptions, keep the ordinal labels.
        lines = [f"{i}: {name}" for i, name in enumerate(instance.level_names)]
    else:
        lines = [f"{i}: {rubric}" for i, rubric in enumerate(instance.rubrics)]

    show_context = input_config == "full"
    return USER_PROMPT.format(
        rubrics="\n".join(lines),
        question=instance.question if show_context else "(nicht bereitgestellt)",
        sample_solutions=instance.sample_solution if show_context else "(nicht bereitgestellt)",
        answer=instance.answer.strip(),
    )


def parse_score(raw: str, num_rubrics: int) -> tuple[int | None, str]:
    """Pull ``score`` out of the model reply, tolerating stray prose or fences."""
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            score = int(payload["score"])
            reasoning = str(payload.get("reasoning", ""))
            if 0 <= score < num_rubrics:
                return score, reasoning
        except (ValueError, KeyError, TypeError):
            pass
    # fall back to the first plausible integer anywhere in the reply
    for token in re.findall(r"-?\d+", text):
        score = int(token)
        if 0 <= score < num_rubrics:
            return score, ""
    return None, ""


class OpenAIBackend:
    """Chat-completions client; works against any OpenAI-compatible server."""

    def __init__(self, model: str, base_url: str | None, temperature: float, max_retries: int):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def __call__(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as error:  # network/rate-limit; back off and retry
                last_error = error
                time.sleep(2**attempt)
        raise RuntimeError(f"request failed after {self.max_retries} attempts") from last_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--task", choices=["lp", "ke", "sk"], default="lp")
    parser.add_argument("--split", default="test_ua", choices=["train", "val", "test_ua", "test_uq"])
    parser.add_argument("--model", required=True, help="e.g. gpt-4o-mini, or a served model name")
    parser.add_argument("--backend", choices=["openai"], default="openai")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (vLLM, Ollama, ...)")
    parser.add_argument("--input-config", choices=INPUT_CONFIGS, default="full")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N instances")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    loader = AliceLoader(data_dir=args.data_dir, task=args.task)
    instances = loader.splits[args.split]
    if args.limit:
        instances = instances[: args.limit]

    system = SYSTEM_PROMPT.format(focus=TASK_FOCUS[args.task])

    if args.dry_run:
        print(system)
        print("\n" + "=" * 60 + "\n")
        print(build_user_prompt(instances[0], args.input_config))
        return

    if args.backend == "openai" and not (os.environ.get("OPENAI_API_KEY") or args.base_url):
        raise SystemExit("set OPENAI_API_KEY, or point --base-url at a local server")
    backend = OpenAIBackend(args.model, args.base_url, args.temperature, args.max_retries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    unparsed = 0
    started = time.perf_counter()
    for position, instance in enumerate(instances, start=1):
        raw = backend(system, build_user_prompt(instance, args.input_config))
        score, reasoning = parse_score(raw, instance.num_rubrics)
        if score is None:
            unparsed += 1
            # An unusable reply still has to be scored: fall back to level 0,
            # the lowest rubric, and record it so the rate is auditable.
            score = 0
        rows.append(
            {
                "id": instance.id,
                "question_id": instance.question_id,
                "subject": instance.subject,
                "gold": instance.level,
                "pred": score,
                "gold_global": instance.global_level,
                "pred_global": int(float(instance.level_names[score])),
                "reasoning": reasoning.replace("\n", " "),
                "raw_output": raw.replace("\n", " "),
            }
        )
        if position % 50 == 0:
            print(f"{position}/{len(instances)}", flush=True)

    report = evaluation_report(
        [row["gold_global"] for row in rows],
        [row["pred_global"] for row in rows],
        [row["subject"] for row in rows],
    )
    report["unparsed_replies"] = unparsed
    report["seconds_per_instance"] = (time.perf_counter() - started) / max(1, len(rows))
    report["config"] = {
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "input_config": args.input_config,
        "temperature": args.temperature,
    }

    save_predictions(rows, output_dir / f"{args.split}.csv")
    save_report(report, output_dir / f"{args.split}_metrics.json")

    overall = report["overall"]
    print(
        f"{args.split}: acc={overall['accuracy']:.4f} "
        f"f1_macro={overall['f1_macro']:.4f} qwk={overall['qwk']:.4f} "
        f"unparsed={unparsed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
