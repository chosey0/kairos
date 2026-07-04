# AGENTS.md

## Scope

This directory holds the notebook sources for the candlestick-shape-quantization
research. Follow, in order:

1. Repository root [AGENTS.md](../../AGENTS.md) — research rules, leakage rules,
   testing rules.
2. [notebooks/AGENTS.md](../AGENTS.md) — run/figure layout, config hash rules,
   required run files.
3. This file — notebook roles and the reference papers this research is built on.

The research design contract is
[docs/candlestick-shape-quantization/08-research-roadmap.md](../../docs/candlestick-shape-quantization/08-research-roadmap.md).
Do not change feature definitions, split protocol, or decision gates inside a
notebook; change the roadmap first.

## Notebook Sequence

Notebooks are numbered by execution order and map to roadmap phases:

```text
00_data_protocol.ipynb            Phase 0  data manifest, cleaning, split spec (D1 first)
01_shape_feature_validation.ipynb Phase 1  shape core distributions, boundary-candle A/B
02_tokenizer_baselines.ipynb      Phase 1  k-means / GMM / hand-crafted bins
03_vq_tokenizer.ipynb             Phase 1  VQ-VAE (+ FSQ/LFQ variants), K sweep
04_token_atlas.ipynb              Phase 1  prototype table, shape atlas, usage stats
05_motif_vocabulary.ipynb         Phase 3  BPE/Unigram/Sequitur + surrogate tests
```

Rules:

- A notebook may consume only artifacts produced by lower-numbered notebooks or
  by `kairos/` code. Never reach forward.
- Import feature logic from `kairos.features` (`extract_shape`,
  `extract_features`, `ShapeFeatures`). Do not reimplement lambda/logit math,
  ATR, or volume statistics inline; if a notebook needs new reusable logic, move
  it into `kairos/` with unit tests first.
- `00_data_protocol` must pin the D1 symbol, split boundary dates, and cleaning
  rules before any later notebook fits anything. Later notebooks read the split
  spec from the phase-00 run artifacts instead of redefining it.
- Record the roadmap decision gates in the run's `metrics.json` when a notebook
  answers one (for example `vq_beats_kmeans`, `effective_vocab_ratio`,
  `surrogate_test_passed`).

## Reference Papers

Local PDFs live in `references/` at the repository root; the links below are
the canonical sources. Cite them by arXiv id in notebook markdown and in
`07-evidence-and-references.md` when a design decision depends on them.

### Feature representation and quantization

| Paper | Link | Role in this research |
| --- | --- | --- |
| Dimension reduction of OHLC data based on pseudo-PCA | https://arxiv.org/abs/2103.16908 | Constraint-free bijective OHLC representation; direct basis for the `(logit lambda_o, logit lambda_c)` shape core (notebook 01) |
| Kronos: A Foundation Model for the Language of Financial Markets | https://arxiv.org/abs/2508.02739 (code: https://github.com/shiyu-coder/Kronos) | K-line tokenizer (BSQ), coarse/fine hierarchical tokens, instance z-score + clip preprocessing, data-cleaning pipeline (notebooks 00, 03, 05); pretrained zero-shot baseline for Phase 5 |
| Neural Discrete Representation Learning (VQ-VAE) | https://arxiv.org/abs/1711.00937 | Discrete latent codebook learning; the base tokenizer model (notebook 03) |
| TimeVQVAE: Vector Quantized Time Series Generation with a Bidirectional Prior | https://arxiv.org/abs/2303.04743 | Coarse/fine codebook separation and VQ evaluation practice for time series (notebook 03) |
| A Novel Trend Symbolic Aggregate Approximation (TSAX) | https://arxiv.org/abs/1905.00421 | Why symbolization must keep direction/trend, not only level; SAX/TSAX baselines in Phase 5 |
| TS2Vec: Towards Universal Representation of Time Series | https://arxiv.org/abs/2106.10466 | Continuous contrastive baseline; measures information lost by quantization (Phase 5) |
| VQ-AR: Vector Quantized Autoregressive Probabilistic Time Series Forecasting | https://arxiv.org/abs/2205.15894 | Evidence that discrete representations can drive forecasting; comparison framing for Phase 5 |

### Candlestick pattern recognition and interpretability

| Paper | Link | Role in this research |
| --- | --- | --- |
| Encoding Candlesticks as Images for Pattern Classification Using CNNs (GAF-CNN) | https://arxiv.org/abs/1901.05237 | The 8-pattern taxonomy used as ground truth for token/motif vs traditional-pattern overlap (Phase 4) |
| Explainable Deep Convolutional Candlestick Learner | https://arxiv.org/abs/2001.02767 | Perturbation-based analysis of which shape components drive assignments; motif sensitivity checks (Phase 4) |
| Dynamic Deep Convolutional Candlestick Learner | https://arxiv.org/abs/2201.08669 | Variable-length pattern localization perspective for motif boundary stability (Phase 4) |
| Using Deep Learning and Candlestick Chart Representation to Predict Stock Market | https://arxiv.org/abs/1903.12258 | Image-CNN prediction baseline; window-length and volume-inclusion ablation design (Phase 5) |
| Deep Stock Representation Learning: From Candlestick Charts to Investment Decisions | https://arxiv.org/abs/1709.03803 | Unsupervised candle representation feeding downstream decisions; evaluation structure precedent (Phase 5) |

Caution: arXiv `2508.02739` is the Kronos paper, not a VLM candlestick benchmark.
Do not mislabel it in citations.

## Research-Specific Reminders

- yfinance symbol mapping: request `^GSPC` for S&P 500 (`^SPX` is not a Yahoo
  symbol); record every mapping in the run config.
- Zero-range candles (`high == low`) never enter tokenizer fitting; count them in
  `metrics.json`. Boundary candles default to winsorize + boundary flags
  (`boundary_policy: "winsorize_flag"`), with the exclusion policy run as the
  A/B comparison arm.
- All trailing statistics (ATR, volume median) come from `kairos.features` and
  use data up to `t-1` only; warmup rows carry `None` channels — filter and
  count them, never impute.
- Seeds: at least `7`, `17`, `37`; one `run-*` directory per seed under the same
  `cfg-*` hash.
