# AGENTS.md

## Project Summary

`kairos` is a local-first research project for quantizing OHLC candles into
interpretable discrete shape tokens, then learning multi-candle motif
vocabularies from token sequences.

The research design source of truth is
[docs/candlestick-shape-quantization/08-research-roadmap.md](docs/candlestick-shape-quantization/08-research-roadmap.md).
If code and documentation conflict, follow the roadmap. If the plan itself must
change, update the roadmap first, then code and notebooks.

Do **not** add trading/order execution, live portfolio management, strategy
execution, or production brokerage workflows unless explicitly requested for a
concrete feature. Broker integrations in this repository are read-only data
access helpers for research.

## Repository Layout

```text
kairos/   Research pipeline package:
            core/                pure contracts, feature transforms, metrics,
                                 tokenizer model, and training helpers
            experiments/         research protocols, artifact helpers, and
                                 step-specific experiment logic
            sources/             read-only external market-data access helpers
tests/    Focused unit tests for leakage-sensitive pure functions
notebooks/ Experimental research notebooks. Outputs live under
            notebooks/runs/<experiment-name>/
docs/     Research design and evidence documents
references/ Reference papers and local research material
main.py   Lightweight broker-modules import smoke example
pyproject.toml Dependency metadata managed by uv
```

There is no local `modules/` package in this repository. If stale references to
`modules.*` appear in code or notebooks, treat them as migration debt and prefer
repo-local `kairos.*` contracts or the external `broker-modules` package.

## Architecture Rules

### Research Core (`kairos/`)

- Keep the research core pure and deterministic where possible.
- Prefer small dataclasses and pure functions for candle contracts, feature
  extraction, split logic, metrics, and token transforms.
- Never hide time ordering in helper functions. Inputs that depend on sequence
  order must require timestamp-sorted candles or validate sorting explicitly.
- Do not let notebooks become the only implementation of reusable logic. Move
  stable logic into `kairos/` and keep notebooks as experiment orchestration and
  reporting surfaces.
- Do not add database, broker, or credential dependencies to tokenizer/model
  code. Data loading should stay at the edge.

### Broker Modules

`broker-modules` is an external dependency, not owned by this repository. Import
broker SDKs through the package namespace:

```python
from brokers.kis import KisClient
from brokers.kiwoom import KiwoomClient
from brokers.krx import KrxClient
from brokers.toss import TossClient
```

Use broker SDKs only for read-only market/account-data experiments unless a task
explicitly says otherwise. Do not modify broker SDK internals from `kairos`; make
changes in the `broker-modules` repository instead.

Unit tests must not call real broker APIs. Use fixtures, sample rows, or mocked
responses.

## Development Commands

Prefer `uv`:

```bash
uv sync
uv run python main.py
uv run --with pytest pytest -q
uv run --with ruff ruff check .
```

Python 3.12 or newer is required. Core research dependencies include `torch`,
`scikit-learn`, `plotly`, and `matplotlib`. `yfinance` may remain available for
ad hoc comparison, but it is not the primary research data source.

For package changes, update `pyproject.toml` through `uv add` / `uv remove` when
practical, then refresh `uv.lock`.

## Data and Local Files

- Do not commit secrets, account numbers, API keys, tokens, local `.env` values,
  raw broker responses, downloaded market datasets, model checkpoints, local DB
  files, or large generated artifacts unless explicitly requested.
- Keep durable experiment summaries small and reviewable: `experiment_config.json`,
  `metrics.json`, and selected figures/tables under
  `notebooks/runs/<experiment-name>/`.
- Research index data source is fixed: domestic Korean index data comes from
  Kiwoom, and overseas index data comes from KIS. Record broker method names,
  broker symbols/codes, and credential environment assumptions in the run
  config. Use yfinance only as an explicitly labeled fallback or comparison
  source.
- If extracting PDF text locally, prefer Python tooling such as `pypdf`; do not
  assume poppler is installed.

## Research Rules

These are roadmap decisions and must not be changed casually.

- **Leakage prevention is the top priority.** No model, scaler, tokenizer, or
  vocabulary may be fit before time-based train/validation/test boundaries are
  fixed. Fit only on the train window. ATR, volume median, z-score, and related
  normalization statistics must use train-only statistics or rolling data up to
  `t-1`. Labels may use only information after `t+1` as designed. Random split
  is forbidden.
- **Feature definition.** Tokenizer input shape core is
  `(logit lambda_o, logit lambda_c)`, where
  `lambda_o = (open - low) / range` and
  `lambda_c = (close - low) / range`, with epsilon winsorization. Body, upper,
  lower, and center features are derived reporting/visualization fields only.
  `rel_range`, `gap`, and `vol_spike` are separate continuous side channels and
  must not be mixed into the shape token.
- **Stage expansion.** Follow D1(single stock index, each index as its own
  dataset) -> D2(country-level stock index groups, such as KOSPI/KOSDAQ or
  NASDAQ/NYSE/AMEX representative indexes) -> D3(global major stock indexes as a
  combined dataset). Feature definitions, split protocol, and evaluation metrics
  fixed in D1 must carry forward. If they change, rerun from D1. Keep VIX out of
  the main set.
- **Exceptional candles.** `high == low` zero-range candles get a special token
  and must be counted. Boundary candles such as marubozu use winsorize +
  boundary flags by default, with exclusion compared and recorded when relevant.
- **Decision gates.** If VQ-family tokenizers do not beat k-means on at least one
  meaningful criterion such as reconstruction error, transfer, or stability,
  choose k-means and continue. If motif learning does not pass a Markov-1
  surrogate test, continue on the single-token path and document the result.

## Experiment Records

- Every experiment leaves `experiment_config.json` and `metrics.json` in
  `notebooks/runs/<experiment-name>/`.
- Tokenizer experiments must record reconstruction error, `token_entropy`,
  `effective_vocab_size`, `dead_token_count`, and excluded/special-row counts
  and ratios.
- Run at least three seeds, such as `7`, `17`, and `37`, and report prototype
  stability across seeds.
- Report results by index and regime, not only as a global average.
- Notebook changes should be rerun end to end when practical. If not practical,
  state the exact cells or validations that were run.

## Testing Rules

- Add or update unit tests for pure functions in `kairos/`, especially feature
  transforms, split boundaries, metric calculations, and leakage-sensitive
  rolling statistics.
- Tests for split logic must cover boundary dates and embargo behavior when that
  behavior is present.
- Tests for side channels must prove the current candle never contributes to its
  own trailing statistic.
- Keep broker and yfinance access mocked in tests. No network calls in unit
  tests.
- Before claiming a code change is complete, run the smallest relevant test set;
  for broad changes, run `uv run --with pytest pytest -q` unless `pytest` has
  been added as a project dependency.

## Documentation Rules

- Keep research documents in Korean narrative with English code blocks where
  useful.
- Preserve top-level and related-document links at the top of each research
  document.
- When adding or deleting docs, update
  [docs/candlestick-shape-quantization/README.md](docs/candlestick-shape-quantization/README.md)
  and [docs/README.md](docs/README.md).
- When citing a new paper, record how it supports or changes a design decision in
  [docs/candlestick-shape-quantization/07-evidence-and-references.md](docs/candlestick-shape-quantization/07-evidence-and-references.md).
- Document only implemented commands and verified experiment outputs as available.

## Git and PR Rules

- Keep changes focused and reviewable. Separate research-plan edits, code edits,
  notebook output updates, and dependency changes when they can be reviewed
  independently.
- Commit only intentional source, test, docs, and small experiment summary files.
- Include verification evidence in final reports: tests run, notebook cells or
  scripts run, and any known gaps.
