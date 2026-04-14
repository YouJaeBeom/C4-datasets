# C4 Political-Topic Subset with LLM Persona Annotations

A public dataset and reproducible data pipeline that filters politically-sensitive documents from the C4 corpus (`allenai/c4`) across 15 political topics and annotates each document with political orientation / stance labels using multiple LLMs under four distinct political personas.

This repository accompanies the following paper:

```bibtex
@inproceedings{you2026data,
  title={From Data to Model in Bias: A Statistical Analysis of Political Bias in the C4 Corpus and Its Impact on LLMs},
  author={You, Jaebeom and Lee, Jaewon and Lee, Sehun and Kwon, Hyuk-Yoon},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={860--870},
  year={2026}
}
```

## Overview

Large-scale web corpora such as C4 are the primary training source for modern LLMs, yet their political leaning has not been systematically measured at the document level. This project builds a topic-balanced political subset of C4 and attaches high-quality persona-conditioned annotations so that:

1. **Researchers** can study political bias distributions inside the training data itself.
2. **Model builders** can quantify how source-corpus bias propagates into downstream model behavior.
3. **The community** can reuse the released artifacts for further fairness, debiasing, and alignment research.

The end-to-end workflow is:

```
C4 (streaming)  →  Rule/NLI filtering  →  Topic-balanced raw subset  →  4-persona × multi-LLM annotation  →  Annotated public dataset  →  Statistical bias analysis
```

## Repository Structure

```
C4 datasets/
├── C4_datat_collection/        # Stage 1: streaming collection + NLI filtering
│   └── c4_huggingface_nli.py
├── LLM_based_annotation/       # Stage 2: multi-persona annotation via GPT / Claude
│   ├── gpt_annotation.py
│   ├── chatgpt/chatgpt_request.py
│   ├── claude/claude_request.py
│   └── prompt/                 # Role prompts + content prompt + few-shot samples
├── Bias_detection/             # Stage 3: statistical bias analysis (TOST, etc.)
│   ├── c4_huggingface_nli.py   # (referenced from Stage 1)
│   ├── bias_detection_test.py
│   ├── integrated_bias_analysis.py
│   └── sample_size_calculation.py
├── datasets/                   # Raw topic-balanced subset of C4 (per-topic CSVs)
├── result_datasets/            # Final annotated dataset (per-topic CSVs)
└── topics-questions.csv        # 15 topics × probe questions used downstream
```

## Data Collection Pipeline (Stage 1)

File: `C4_datat_collection/c4_huggingface_nli.py`

The collector streams the English split of `allenai/c4` directly from Hugging Face Datasets and funnels each document through a four-stage filter. Streaming avoids materializing the full 300+ GB corpus, and the filter targets high-precision political documents per topic.

**Target topics (15).** tax policy, trade policy, free-market, civil liberties, gun control, death penalty, abortion, LGBTQ, drug policy, immigration, gender equality, bioethics, nationalism, multiculturalism, climate change.

**Per topic** we maintain a curated list of 9–10 expansion keywords (e.g., *abortion* → `pro choice`, `pro life`, `reproductive rights`, `bodily autonomy`, …).

**Four-stage filter.**

| Stage | Check | Purpose |
|-------|-------|---------|
| 1 | Any topic keyword present | Cheap prefilter on the streaming iterator |
| 2 | `len(text) ≥ 300` and `url` starts with `https://` + URL deduplication | Remove stubs and duplicates |
| 3 | Topic keyword appears `≥ 3` times with word-boundary regex | Enforce topical density, not incidental mentions |
| 4 | `facebook/bart-large-mnli` zero-shot NLI with `This text is about {topic}`, threshold `0.7` | Semantic topicality, chunked for long texts |

Documents that survive all four stages are scored, sorted by NLI score, and the top `samples_per_topic = 3000` per topic are saved to `datasets/<topic>_<timestamp>.csv` with metadata (URL, keyword frequencies, text length, NLI score).

**Why NLI on top of keywords.** Keyword hits alone return many off-topic pages (e.g., real-estate listings mentioning "tax"). The zero-shot entailment score forces each surviving document to semantically entail the topic, which is what downstream bias analysis depends on.

**Key design properties.**

- **Streaming only** — `load_dataset(..., streaming=True)` + `.filter()` so memory stays constant regardless of corpus size.
- **Batched GPU inference** — `pipeline(..., device=0)` with `batch_size = 32`, chunking texts into `max_chunk_length = 1800` character windows for BART's input limit.
- **URL deduplication** — `seen_urls` set prevents the same crawled page from appearing twice.
- **Deterministic artifacts** — every approved row is serialized with reproducible metadata (`total_keyword_frequency`, `max_keyword`, `nli_topic_score`) so filtering decisions are auditable.
- **Resumable topic-by-topic loop** with per-topic logs (stage pass counts, duplicate counts, success rate, average NLI score).

Raw outputs are in `datasets/`. Typical yield per topic is a few thousand high-confidence documents (CSVs of 50k–125k rows of streaming stats + 3000 kept samples each).

## LLM-Based Annotation Pipeline (Stage 2)

File: `LLM_based_annotation/gpt_annotation.py` (+ `chatgpt/`, `claude/`, `prompt/`)

Each surviving document is annotated by **multiple LLMs under four political personas**, providing a bias-stress-test view of the same text.

**Personas (four):**

- `opp_left` — opposed, left-leaning
- `opp_right` — opposed, right-leaning
- `sup_left` — supportive, left-leaning
- `sup_right` — supportive, right-leaning

Each persona is injected as a system / role prompt from `prompt/prompt_role_*.txt`, while the content prompt (`prompt/prompt_input.txt`) supplies the `topic` and the document `text`. Few-shot exemplars live in `prompt/few_shot_sample.txt`.

**Models.** The orchestrator accepts arbitrary Claude and ChatGPT model lists. The default configuration in `__main__` uses `claude-sonnet-4-20250514` and `gpt-4.1`, giving `2 models × 4 personas = 8 annotations per document`.

**Structured output.** Each call must return a JSON object with:

```json
{
  "Political": { "label": "...", "score": 0.0 },
  "Stance":    { "label": "...", "score": 0.0 },
  "Reasoning": "..."
}
```

Malformed or failed responses are replaced by a neutral `"Undecided"` record via `create_empty_result_json()` so downstream analysis never breaks on NaNs.

**Orchestration highlights.**

- **Resumable, idempotent writes.** The output CSV is re-read and only missing `<model>_<persona>` cells are filled, so the pipeline can be interrupted and restarted without re-billing already-completed calls.
- **Per-model rate limiting.** Random `1–3 s` jitter between calls (`time.sleep(random.uniform(1, 3))`) avoids provider throttling.
- **Column auto-provisioning.** `ensure_columns_exist` materializes one column per `model_version_persona` pair before annotation.
- **Content-column autodetection.** Accepts `Article_Content`, `content`, or `text` so the annotator can run on collectors other than this repo's.
- **Row-level checkpointing.** The CSV is flushed after every row update.

Annotated outputs land in `result_datasets/annotated_<topic>_<timestamp>.csv`.

## Bias Detection (Stage 3)

Folder: `Bias_detection/`

After annotation, statistical tests quantify whether the corpus is skewed toward one political side. Tools include:

- `integrated_bias_analysis.py` — end-to-end bias analysis across all 15 topics.
- `bias_detection_test.py` — Two One-Sided Tests (TOST) for equivalence testing.
- `sample_size_calculation.py` — statistical power / sample-size planning.

These scripts consume `result_datasets/annotated_*.csv` and output the per-topic statistics reported in the paper.

## Released Artifacts

- `datasets/*.csv` — topic-filtered raw subset of C4 (per-topic, dated).
- `result_datasets/annotated_*.csv` — same documents with `model_persona` annotation columns.
- `topics-questions.csv` — probe questions used to evaluate downstream LLMs on the same 15 topics.
- `Bias_detection/analysis_results/` — precomputed statistical outputs.

All data rows carry the original C4 `url`, letting consumers re-verify source provenance against the public C4 release.

## Reproducing the Pipeline

### 1. Environment

Python 3.10+ with:

```bash
pip install datasets transformers torch pandas tqdm openai anthropic python-dotenv
```

A CUDA-capable GPU is strongly recommended for Stage 1 (`facebook/bart-large-mnli` zero-shot NLI over millions of streaming rows).

### 2. API keys for Stage 2

Create a `.env` (ignored by git) with:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Never commit real keys. The repository only ships prompt templates and orchestration code.

### 3. Run

```bash
# Stage 1 — collect topic-filtered subset from C4 (streaming)
cd C4_datat_collection
python c4_huggingface_nli.py

# Stage 2 — annotate with multi-LLM × 4 personas
cd ../LLM_based_annotation
python gpt_annotation.py

# Stage 3 — statistical bias analysis
cd ../Bias_detection
python integrated_bias_analysis.py
```

Each stage is independent — you can point Stage 2 at any pre-existing CSV with a `url` column and one of `text` / `content` / `Article_Content`.

## Intended Use and Limits

- The annotations reflect **LLM judgments**, not ground-truth human labels. They are designed as a *relative* bias signal across personas and models, not an absolute political ground truth.
- The collection targets English C4 only and 15 explicitly political topics — it is not a general-purpose corpus.
- Downstream users should cite the paper above and respect the original `allenai/c4` license.

## License

Code in this repository is released for research use. The underlying text originates from `allenai/c4` and remains subject to C4's own terms.
