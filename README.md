# ALICE

**A Large-Scale German Benchmark for Rubric-Based Multi-Dimensional Automatic Short Answer Scoring**

ALICE is a short-answer scoring benchmark built from K-12 STEM classroom assessment in German — biology, chemistry, mathematics and physics. Unlike benchmarks that compare a student answer against a single reference solution over a fixed label set, **every question in ALICE carries its own rubric set**, and answers are scored along three dimensions at once. Label semantics live in rubric *text*, not in a classification head.

This repository holds the dataset, the data-processing code, and reference implementations of the modelling approaches reported in the paper.

<p align="center">
  <img src="assets/dataset_example.png" alt="An ALICE instance: question prompt, sample solution and student response at the top, then per-level rubrics for learning performance, knowledge elements and skills, with the gold level highlighted in each dimension." width="560">
</p>

<p align="center"><em>One instance from ALICE. The same student response is scored against three rubric sets; highlighted cells mark the gold level.</em></p>

---

## Contents

- [Scoring dimensions](#scoring-dimensions)
- [Splits](#splits)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Two indexings of a score](#two-indexings-of-a-score)
- [Modelling approaches](#modelling-approaches)
- [Input formats](#input-formats)
- [Training](#training)
- [Inference](#inference)
- [Zero-shot prompting](#zero-shot-prompting)
- [Reported results](#reported-results)
- [Repository layout](#repository-layout)
- [Licence and terms of use](#licence-and-terms-of-use)
- [Citation](#citation)

---

## Scoring dimensions

| Task | Dimension | Structure | Levels |
|---|---|---|---|
| `lp` | **Learning Performance** | one holistic score per answer | 0–2 |
| `ke` | **Knowledge Elements** | one score per knowledge element the question targets | 0–3 |
| `sk` | **Skills** | one score per skill the question targets | 0–2 |

`ke` and `sk` are multi-label: a single answer expands into one scoring instance per element or skill. Mathematics questions are annotated for `lp` only.

## Splits

| Split | Answers | Questions | What it tests |
|---|---|---|---|
| `train` | 10,781 | 90 | — |
| `test_ua` | 2,695 | 89 | **U**nseen **A**nswers to seen questions |
| `test_uq` | 3,096 | 22 | **U**nseen **Q**uestions — the harder generalisation setting |

Test-UQ is question-disjoint from training. A validation set is carved out of `train` instance-wise at load time (10% by default), so it is question-disjoint from Test-UQ by construction but shares questions with Test-UA.

After expanding the multi-label dimensions, the scoring instances per subtask are:

| Task | train | val | test_ua | test_uq |
|---|---|---|---|---|
| `lp` | 9,703 | 1,078 | 2,695 | 3,096 |
| `ke` | 21,516 | 2,390 | 5,851 | 6,304 |
| `sk` | 10,222 | 1,135 | 2,833 | 3,103 |

Reproduce this table with `python src/dataset_stats.py`.

---

## Quick start

```bash
git clone https://github.com/<user>/alice-benchmark.git
cd alice-benchmark
pip install -r requirements.txt
```

The dataset ships as `alice_data_public.zip`. **You do not need to unzip it** — the loader reads the archive in place. Unzip it anyway if you would rather browse the JSON:

```bash
unzip alice_data_public.zip     # optional
```

Inspecting the data needs no dependencies at all — `data_utils.data_loader` is pure standard library:

```bash
python src/dataset_stats.py --show-example --task ke
```

```python
import sys; sys.path.insert(0, "src")   # from the repo root
from data_utils.data_loader import AliceLoader

loader = AliceLoader(task="lp")
print(loader)
# AliceLoader(task='lp', train=9703, val=1078, test_ua=2695, test_uq=3096)

instance = loader.test_ua[0]
instance.answer      # the student response
instance.question    # the question prompt
instance.rubrics     # the candidate levels, as rendered text
instance.level       # index of the gold level in `instance.rubrics`
```

## Data format

```
alice_data_public.zip
  alice_data_public/
    train.json          test_ua.json          test_uq.json
    question_meta/      bio.json  chemie.json  math.json  physik.json
```

`--data-dir` accepts either the archive or an extracted directory; with no flag it finds whichever is present.

A record in a split file:

```json
{
  "id": "train_0",
  "question_id": "physik_8",
  "answer": "Keine Ahnung\n",
  "learning_performance": "0",
  "knowledge_elements": {"Umwandlung": "0"},
  "skills": {"Reasoning": "0"}
}
```

Its rubric metadata, under `physik_8` in `question_meta/physik.json`:

```json
{
  "prompt": "...the question shown to students...",
  "solution": "...a sample solution...",
  "learning_performance": {
    "0": {"rule": "nicht richtig",     "description": "..."},
    "1": {"rule": "teilweise richtig", "description": "..."},
    "2": {"rule": "richtig",           "description": "..."}
  },
  "knowledge_elements": {
    "Umwandlung": {"0": {"rule": "...", "description": "..."}, "...": {}}
  },
  "skills": {
    "Reasoning": {"0": {"rule": "...", "description": "..."}, "...": {}}
  }
}
```

`AliceLoader` joins the two, expands `ke`/`sk` into per-item instances, and renders each rubric level into the string a model sees. Two quirks of the raw files it handles for you: mathematics stores its `learning_performance` label as an integer while every other subject uses a string, and some questions omit intermediate levels.

## Two indexings of a score

A rubric set does not always cover every level. A knowledge element may define only levels `{0, 1, 3}` — three candidates, but the top one is labelled `3`. The two modelling approaches need different indexings, so every `Instance` carries both:

| Field | Meaning | Used by |
|---|---|---|
| `level` | *position* of the gold level within this question's rubric list — `{0, 1, 3}` → `{0, 1, 2}` | rubric-retrieval, which ranks candidates |
| `global_level` | the raw annotated level, on the dataset-wide scheme | sequence classification, whose head is shared across questions |

Both training scripts report metrics on `global_level`, so numbers from the two approaches are directly comparable. Predictions are written with both columns.

---

## Modelling approaches

**Rubric-retrieval.** Each instance is expanded into its candidate rubric levels. Every candidate is encoded *jointly* with the answer (optionally with the question and sample solution), pooled, and projected to a scalar alignment score. Scores compete inside the candidate set through a softmax cross-entropy:

$$P(j \mid i) = \frac{\exp(z_{i,j})}{\sum_k \exp(z_{i,k})}, \qquad \ell_i = -\log P(y_i \mid i)$$

Because ALICE rubrics can overlap semantically rather than being mutually exclusive, judging each level in isolation is not enough — the model has to *contrast* the correct level against the alternatives of the same question. Padded candidate slots are masked before the softmax, so a batch can mix questions with different numbers of levels.

**Sequence classification.** The conventional ASAS baseline: concatenate the sample solution and the answer (optionally the question and rubrics), and predict with a fixed head over the dataset-wide level scheme. For `ke`/`sk` the element or skill name is prepended to the input, since the head cannot otherwise know which item it is scoring.

**Zero-shot prompting.** A chat LLM receives the same input components and picks a level. See [Zero-shot prompting](#zero-shot-prompting).

## Input formats

`--input-fields` selects which segments appear, using one letter each:

| Letter | Segment |
|---|---|
| `a` | student answer (always required) |
| `r` | rubric candidate (required for rubric-retrieval) |
| `q` | question prompt |
| `s` | sample solution |

So `ar` is the base rubric-retrieval input, `qsar` adds full context, and `sa` is the base sequence-classification input. Segments are always emitted in the order question → solution → answer → rubric, whatever order you write the letters in.

`--input-format` controls how segment boundaries are marked, and defaults to the right choice for the encoder:

- `structured` — XML-style German tags `<Frage>`, `<Loesung>`, `<Antwort>`, `<Rubrik>`; the default for causal LLM encoders, following Ruan et al. (2024).
- `sep` — segments joined with the tokeniser's `[SEP]`; the default for masked LMs.

## Training

Masked LM encoder, rubric-retrieval, base input (paper setting: lr 2e-5, 4 epochs, batch 16):

```bash
python src/train.py \
  --task lp --approach rubric-retrieval \
  --base-model jhu-clsp/mmBERT-base \
  --input-fields ar \
  --lr 2e-5 --epochs 4 --batch-size 16 \
  --output-dir runs/mmbert_lp_ar
```

Causal LLM encoder under QLoRA, full context (paper setting: 4-bit, LoRA rank 64, lr 1e-4, 3 epochs, batch 4 × 8 accumulation). Needs CUDA and `bitsandbytes`:

```bash
python src/train.py \
  --task ke --approach rubric-retrieval \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --input-fields qsar \
  --use-lora --use-4bit --lora-rank 64 --lora-alpha 64 \
  --lr 1e-4 --epochs 3 --batch-size 4 --grad-accum 8 --bf16 \
  --output-dir runs/llama3b_ke_qsar
```

Sequence-classification baseline:

```bash
python src/train.py \
  --task lp --approach seq-classification \
  --base-model jhu-clsp/mmBERT-base \
  --input-fields sa \
  --lr 2e-5 --epochs 4 --batch-size 16 \
  --output-dir runs/mmbert_lp_seqcls
```

Training evaluates on the validation split each epoch, keeps the best checkpoint by accuracy, then scores both test splits. Each run directory ends up with:

```
runs/<name>/
  run_config.json                 # every setting, for reproducibility
  predictions/test_ua.csv         # per-instance gold and predictions
  predictions/test_ua_metrics.json
  predictions/test_uq.csv
  predictions/test_uq_metrics.json
  summary.json                    # accuracy / macro-F1 / QWK per split
```

Add `--train-frac 0.01 --epochs 1` for a fast smoke run, and `--subjects bio physik` to restrict subjects.

Metrics are macro-F1, quadratic weighted kappa and accuracy, overall and broken down per subject. QWK is computed on the global level scheme, the only indexing on which the ordinal scale is meaningful.

## Inference

`src/predict.py` reads `run_config.json` back, so a checkpoint scores a split with exactly the input format it was trained with:

```bash
python src/predict.py --checkpoint runs/mmbert_lp_ar --split test_uq
```

## Zero-shot prompting

`src/zero_shot_llm.py` reproduces the prompting setup from the paper appendix: a German system prompt with a task-specific FOCUS line, a user message carrying the rubric set and optional context, and a JSON reply with a short justification and one integer score.

Three input configurations:

| `--input-config` | Prompt contains |
|---|---|
| `full` | rubric set + question + sample solution + answer |
| `norub` | rubric set + answer |
| `labels` | level names only (no descriptions) + answer |

```bash
# inspect the exact prompt without spending tokens
python src/zero_shot_llm.py --task lp --model gpt-4o-mini \
  --input-config full --output-dir /tmp/x --dry-run

export OPENAI_API_KEY=...
python src/zero_shot_llm.py \
  --task lp --split test_uq --model gpt-4o-mini --input-config full \
  --output-dir runs/zeroshot_gpt4omini_lp_full
```

`--backend openai` speaks the chat-completions API, so `--base-url` points it at any compatible server (vLLM, Ollama) to score open-weight models. Replies that cannot be parsed into a valid level fall back to level 0 and are counted in `unparsed_replies` in the metrics file.

## Reported results

Macro-F1 on the unseen-question split, best model per approach. Full tables, QWK, per-subject breakdowns and ablations are in the paper.

| Approach | Best model | LP | KE | SK |
|---|---|---|---|---|
| Rubric-retrieval | Llama-3.2-3B(-Instruct) | **69.5** | **60.0** | **65.4** |
| Sequence classification | Llama-3.2-3B-Instruct | 66.4 | 52.7 | 60.6 |
| Zero-shot | GPT-5-mini | 66.1 | 59.9 | 59.5 |

Fine-tuned encoders benchmarked in the paper: XLM-RoBERTa-Long, mmBERT, Llama-3.2-1B/3B and their Instruct variants. Zero-shot: Mistral-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, GPT-4o-mini, GPT-5-mini.

These numbers are not directly comparable to published baselines associated with the earlier ALICE-LP 1.1 release.

## Repository layout

```
alice_data_public.zip       the dataset
src/
  data_utils/
    data_loader.py          loading, joining, rubric rendering (stdlib only)
    data_prep.py            input formatting, tokenisation, collation
  modelling/
    modelling_rubret.py     the rubric-retrieval model
    modelling_utils.py      shared model/dataset wiring
  metrics.py                macro-F1, QWK, accuracy, per-subject breakdown
  utils.py                  seeding, device selection, prediction I/O
  dataset_stats.py          statistics, and a check that the release parses
  train.py                  fine-tuning, both approaches
  predict.py                inference from a checkpoint
  zero_shot_llm.py          zero-shot prompting
examples/                   ready-to-edit run scripts
assets/                     figures used in this README
```

Run the entry points from the repo root as `python src/<script>.py`; Python puts `src/` on the path, so the `data_utils` / `modelling` imports resolve with no setup.

## Licence and terms of use

- **Data** (`alice_data_public.zip`): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — see [LICENSE-DATA](LICENSE-DATA).
- **Code**: MIT — see [LICENSE](LICENSE).

The data consists of responses written by K-12 students in routine classroom assessment, collected under educational governance and privacy protections; identifiers were removed and responses anonymised before annotation. Do not attempt to re-identify students, teachers, classes or schools, and do not redistribute the data in a form that reverses the anonymisation.

Automated scoring can affect educational outcomes. Models built on ALICE should be treated as tools that support instructional practice, not as replacements for expert human judgment. Please consider bias, fairness and interpretability when reporting on this benchmark.

## Citation

```bibtex
@inproceedings{alice-benchmark,
  title     = {{ALICE}: A Large-Scale German Benchmark for Rubric-Based
               Multi-Dimensional Automatic Short Answer Scoring},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
