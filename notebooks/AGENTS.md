# AGENTS.md

## Notebook Scope

This directory is for research notebooks and notebook-generated experiment
artifacts. Follow the repository root [AGENTS.md](../AGENTS.md) first; this file
adds notebook-specific structure and artifact rules.

Notebook work must preserve the research roadmap contract:
[../docs/candlestick-shape-quantization/08-research-roadmap.md](../docs/candlestick-shape-quantization/08-research-roadmap.md).

## Directory Roles

Keep notebook source, raw experiment outputs, and selected publication figures
separate.

```text
notebooks/
  candlestick-shape-quantization/
    00_data_protocol.ipynb
    01_shape_feature_validation.ipynb
    02_tokenizer_baselines.ipynb
    03_vq_tokenizer.ipynb
    04_token_atlas.ipynb
    05_motif_vocabulary.ipynb

  runs/
    candlestick-shape-quantization/
      phase-00-data-protocol/
      phase-01-shape-tokenizer/
      phase-02-motif-vocabulary/

  figures/
    candlestick-shape-quantization/
      phase-00-data-protocol/
      phase-01-shape-tokenizer/
      phase-02-motif-vocabulary/
      paper/
```

- `notebooks/<research-name>/` contains notebook source only. Do not store run
  outputs beside notebook files.
- `notebooks/runs/` is the reproducible source of experiment outputs.
- `notebooks/figures/` contains only selected figures promoted for docs, papers,
  reports, or presentations.
- Existing flat run folders, such as `notebooks/runs/00-vq-vae-tokenizer-validation/`,
  are legacy layout. Do not expand the legacy pattern for new experiments.

## Run Layout

New experiment outputs should use this path shape:

```text
notebooks/runs/
  <research-name>/
    <phase-id>-<phase-slug>/
      <step-id>-<step-slug>/
        <dataset-stage>_<symbol-set>_<interval>/
          cfg-<config-hash>/
            config.json
            config_readable.yaml
            run-<yyyymmdd-hhmmss>_seed-<seed>/
              experiment_config.json
              metrics.json
              manifest.json
              tables/
              figures/
              logs/
```

Example:

```text
notebooks/runs/candlestick-shape-quantization/
  phase-01-shape-tokenizer/
    step-03-vqvae/
      d1_ixic_daily/
        cfg-a13f9c2/
          config.json
          config_readable.yaml
          run-20260704-153000_seed-7/
            experiment_config.json
            metrics.json
            manifest.json
            tables/
            figures/
```

Use lowercase slugs and ASCII path names. Prefer stable names over long natural
language names.

## Figure Layout

Figure management follows the same research, phase, step, dataset, and config
hierarchy as `runs/`.

```text
notebooks/figures/
  <research-name>/
    <phase-id>-<phase-slug>/
      <step-id>-<step-slug>/
        <dataset-stage>_<symbol-set>_<interval>/
          cfg-<config-hash>/
            selected/
              <figure>.png
              <figure>.svg
            README.md
    paper/
      01-data/
      02-tokenizer/
      03-motif/
```

- Keep all generated figures inside each run's `figures/` directory first.
- Promote only reviewed figures to `notebooks/figures/.../selected/`.
- `paper/` is for final narrative figures that may combine or rename selected
  figures across runs.
- If a figure is promoted, record its source run path and config hash in the
  nearest `README.md` or manifest.

## Config and Hash Rules

Any user-defined variable or hyperparameter that can change experiment results
must be represented in the config object used to derive `cfg-<config-hash>`.

Include at least:

```json
{
  "research": "candlestick-shape-quantization",
  "phase": "phase-01-shape-tokenizer",
  "step": "step-03-vqvae",
  "dataset_stage": "D1",
  "symbols": ["^IXIC"],
  "interval": "1d",
  "data_source_policy": {
    "domestic_index_provider": "kiwoom",
    "overseas_index_provider": "kis",
    "fallback_provider": null
  },
  "split": {
    "train_end": "YYYY-MM-DD",
    "validation_end": "YYYY-MM-DD",
    "embargo_days": 0
  },
  "feature": {
    "eps": 0.001,
    "atr_period": 14,
    "include_volume": false
  },
  "tokenizer": {
    "type": "vqvae",
    "num_codes": 16,
    "embedding_dim": 2,
    "commitment_cost": 0.25
  },
  "user_vars": {
    "max_rows": null,
    "boundary_policy": "winsorize_flag"
  }
}
```

Config hash rules:

- Hash canonical JSON with sorted keys.
- Exclude volatile fields such as wall-clock start time, hostname, absolute local
  paths, runtime duration, and random seed.
- Include dataset selection, broker source mapping, interval, split boundaries,
  feature parameters, model parameters, tokenizer parameters, filtering rules,
  and user-defined variables.
- Store the canonical machine-readable config as `config.json`.
- Store a human-friendly copy as `config_readable.yaml` when practical.
- Seeds create separate `run-*` directories under the same `cfg-*` directory.

## Required Run Files

Every completed run must write:

```text
experiment_config.json
metrics.json
manifest.json
```

Use these meanings:

- `experiment_config.json` records the full run config including seed and runtime
  metadata.
- `metrics.json` records scalar and structured metrics required by the root
  AGENTS experiment rules.
- `manifest.json` records source notebook, git commit when available, input data
  summary, output files, and promoted figure destinations if any.

Tokenizer runs must include reconstruction error, `token_entropy`,
`effective_vocab_size`, `dead_token_count`, and excluded/special-row counts and
ratios.

## Naming Rules

Recommended dataset folder:

```text
<dataset-stage>_<symbol-set>_<interval>
```

Examples:

```text
d1_kospi_daily
d1_ixic_daily
d2_kr-kospi-kosdaq_daily
d2_us-nasdaq-nyse-amex_daily
d3_global-major_daily
d3_global-major_weekly
```

Use the data stages consistently:

- `d1_*`: one stock index per dataset/run.
- `d2_*`: n stock indexes from one country or market group per dataset/run.
- `d3_global-major_*`: all selected global major stock indexes in one dataset.

Data source naming is fixed for this research:

- Domestic Korean indexes use Kiwoom broker-modules APIs.
- Overseas indexes use KIS broker-modules APIs.
- Any yfinance use must be labeled as fallback/comparison and must not replace
  the primary source in run configs.

Recommended figure filename:

```text
<metric-or-view>__<dataset>__<model-or-comparison>__<key-params>__seed-<seed>.<ext>
```

Examples:

```text
token-atlas__d1_ixic_daily__vqvae__k16__seed-7.png
reconstruction-error__d1_ixic_daily__kmeans-vs-vqvae__cfg-a13f9c2.png
split-boundaries__d1_ixic_daily.png
```

Do not encode secrets, private local paths, or full parameter dumps in filenames.

## Notebook Execution Rules

- Notebook cells should be restart-and-run-all friendly.
- Move reusable logic into `kairos/`; notebooks should orchestrate, visualize,
  and record experiments.
- Set random seeds explicitly in each experiment run.
- Write outputs through path helpers or constants near the top of the notebook.
- Do not overwrite an existing `run-*` directory. Create a new timestamped run.
- Do not make network calls in tests. Notebook data downloads are allowed only
  when the experiment config records source, symbol mapping, date range, and row
  counts.

## Promotion Rules

A figure can be promoted from `runs/.../figures/` to `notebooks/figures/...`
only when:

- the source run has `experiment_config.json`, `metrics.json`, and `manifest.json`;
- the figure is connected to a documented result or decision;
- the promoted path preserves research, phase, step, dataset, and config context;
- the source run path and config hash are recorded near the promoted figure.

Do not promote exploratory scratch figures.

## Cleanup Rules

- Keep generated caches such as `__pycache__/` out of notebooks.
- Keep large raw downloads, checkpoints, and temporary exports out of git unless
  explicitly requested.
- If a run is superseded, prefer adding a note to its manifest over deleting it.
- If deleting generated outputs is necessary, delete only generated files and
  preserve notebook source and documented metrics.
