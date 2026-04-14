# Bias Detection & Visualization

Statistical bias detection and **publication-quality visualization** pipeline for the annotated C4 political subset. This module consumes the LLM-persona-annotated data produced by the upstream `C4_datat_collection` + `LLM_based_annotation` stages and produces the statistical tests and figures reported in:

```bibtex
@inproceedings{you2026data,
  title={From Data to Model in Bias: A Statistical Analysis of Political Bias in the C4 Corpus and Its Impact on LLMs},
  author={You, Jaebeom and Lee, Jaewon and Lee, Sehun and Kwon, Hyuk-Yoon},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={860--870},
  year={2026}
}
```

## Scope

This folder answers one question: **given the LLM-annotated political subset of C4, does the corpus exhibit a statistically significant political lean, and how should that lean be *shown* to a reader?**

The pipeline therefore has two sides:

1. **Statistical inference** — TOST equivalence testing, confidence intervals, multiple-comparison correction, power / sample-size planning.
2. **Data visualization** — a single publication-quality figure that renders 15 political topics × 2 indicators (Political leaning, Stance) × 2 LLMs in one composition.

## Files

| File | Role |
|------|------|
| `integrated_bias_analysis.py` | End-to-end analysis over all 15 topics. Loads annotated CSVs, parses per-persona JSON labels, aggregates by topic, runs statistical tests, and renders the final publication figure. |
| `bias_detection_test.py` | TOST (Two One-Sided Tests) equivalence testing against multiple neutrality thresholds (`δ ∈ {0.05, 0.10, 0.15, 0.20}`) with 99% confidence. |
| `sample_size_calculation.py` | Prospective sample-size calculation (Cochran / finite population), validating that the 3,000-samples-per-topic collection meets the required statistical power. |
| `analysis_results/` | CSV artifacts — per-topic summaries, TOST outputs, sample size tables. |
| `fig/` | Publication figures (`publication_bias_analysis_latest.png` + timestamped snapshot, 600 DPI). |

## Statistical Methodology

### 1. Data model
Each annotated document carries up to `|models| × |personas| = 2 × 4 = 8` structured JSON annotations. Every annotation contributes two bounded scores:

- **Political** ∈ [−1, +1] — left vs right lean
- **Stance** ∈ [−1, +1] — opposed vs supportive

The per-document score is the signed mean across the 8 persona × model combinations, which cancels persona-induced variance and isolates the corpus-level signal.

### 2. TOST equivalence testing

Traditional t-tests can only *reject* the null hypothesis — they cannot prove neutrality. To actually claim "this topic is neutral in C4", we use **Two One-Sided Tests** against a neutrality corridor `δ`:

```
H₀₁ : μ ≤ −δ   (lower bound test)
H₀₂ : μ ≥ +δ   (upper bound test)
H₁  : −δ < μ < +δ   (equivalent / neutral)
```

If *both* null hypotheses are rejected at `α = 0.01`, the topic is statistically neutral at threshold `δ`. Otherwise, **bias is detected** and its direction is read off the CI. Thresholds `δ ∈ {0.05, 0.10, 0.15, 0.20}` are swept to show robustness.

### 3. Multiple-comparison correction

Because 15 topics × 2 indicators = 30 simultaneous tests are run, p-values are corrected via `statsmodels.stats.multitest.multipletests` to control family-wise error.

### 4. Sample size validation

`sample_size_calculation.py` estimates the minimum `n` required for the observed standard deviations at 95% / 99% confidence and ±3% / ±5% margins of error, then reports whether the current 3,000-per-topic budget is adequate.

## Visualization — why this is the portfolio artifact

`integrated_bias_analysis.py::create_publication_figure` renders a single **12×12 inch, 600 DPI** figure that is the actual figure in the WSDM 2026 paper. Notable techniques:

- **`matplotlib` + `seaborn`** with custom `rcParams` (font family, sizes, spines) overridden inside the method so the rest of the notebook environment is not polluted.
- **Dual-indicator layout** — Political (circles) and Stance (squares) for every topic on a shared axis, with CI bars connecting them. A grey/blue alternating row band encodes topic groups without adding a legend key.
- **Manual legend composition** using `Line2D` / `Rectangle` proxies, letting the legend mix marker types, bar colors, and background-band swatches in one block.
- **Dual output** — a timestamped archive (`publication_bias_analysis_<timestamp>.png`) plus a stable `publication_bias_analysis_latest.png` alias, so downstream LaTeX `\includegraphics` paths never need editing when the analysis is rerun.
- **Publication-grade DPI** — `dpi=600` with `bbox_inches='tight'` and an explicit white facecolor, producing a figure that holds up under print scaling.

## Running It

### Prerequisites

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
```

The scripts assume the upstream stage has already produced `../annotation_datasets/annotated_<topic>_<timestamp>.csv`.

### Commands

```bash
# Full 15-topic analysis + publication figure (~3 s on CPU)
python integrated_bias_analysis.py

# TOST equivalence test on a single topic (edit base_filename at the top)
python bias_detection_test.py

# Prospective sample-size validation
python sample_size_calculation.py
```

## Results Snapshot

Regenerated artifacts (latest run on 2026-04-14):

### Statistical output

- `analysis_results/integrated_bias_analysis_20260414_183845.csv`
- `analysis_results/topic_summary_20260414_183845.csv`
- `analysis_results/sample_sizes_20260414_183845.csv`
- `analysis_results/bias_detection_pilot_based_20260414_183858.csv`
- `analysis_results/sample_size_calculation_20260414_183858.csv`
- `analysis_results/sample_statistics_20260414_183911.csv`
- `analysis_results/sample_size_detailed_20260414_183911.csv`
- `analysis_results/sample_size_recommendations_20260414_183911.csv`

### Figure

- `fig/publication_bias_analysis_latest.png` (1.2 MB, 600 DPI)
- `fig/publication_bias_analysis_20260414_183840.png` (timestamped snapshot)

### Headline numbers (latest run, δ = ±0.20, α = 0.01)

```
Total tests:        30   (15 topics × 2 indicators)
Bias detected:      25
Bias detection rate: 83.3 %
```

Per-topic highlights (mean ± 99% CI):

| Topic            | Indicator | Mean    | 99% CI              | Verdict        |
|------------------|-----------|---------|---------------------|----------------|
| LGBTQ            | Political | −0.6285 | [−0.6436, −0.6134]  | Bias detected  |
| LGBTQ            | Stance    | +0.7925 | [+0.7800, +0.8050]  | Bias detected  |
| Gender Equality  | Political | −0.5043 | [−0.5358, −0.4728]  | Bias detected  |
| Gender Equality  | Stance    | +0.8136 | [+0.7905, +0.8366]  | Bias detected  |
| Climate Change   | Political | −0.4416 | [−0.4735, −0.4097]  | Bias detected  |
| Tax              | Political | +0.0934 | [+0.0814, +0.1053]  | Neutral        |
| Tax              | Stance    | −0.0588 | [−0.0738, −0.0438]  | Neutral        |
| Trade            | Political | −0.0359 | [−0.0482, −0.0237]  | Neutral        |
| Nationalism      | Stance    | −0.1421 | [−0.1637, −0.1205]  | Neutral        |

The consistent sign pattern (Political negative, Stance positive) across socially progressive topics is what the paper interprets as **systematic left-leaning bias in C4-derived LLM training data**.

## Directory Layout

```
Bias_detection/
├── README.md                       ← this file
├── integrated_bias_analysis.py     ← 15-topic analysis + publication figure
├── bias_detection_test.py          ← TOST equivalence test
├── sample_size_calculation.py      ← power / sample size planning
├── analysis_results/               ← CSV statistical outputs
└── fig/                            ← publication-grade figures
```
