from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from kairos.core.shape_metrics import token_utilization
from kairos.experiments.artifacts import latest_run, read_json
from kairos.experiments.protocol import (
    PHASE_01_ID,
    RESEARCH_NAME,
    STEP_01_FEATURE_ID,
    STEP_02_BASELINE_ID,
    SplitProtocol,
    split_name,
    split_protocol_config_for_interval,
)


@dataclass(frozen=True, slots=True)
class FeatureInput:
    dataset_id: str
    feature_cfg_hash: str
    symbol: str
    provider: Literal["kiwoom", "kis"]


FEATURE_INPUTS = (
    FeatureInput("d1_kospi_daily", "5628106a", "KOSPI", "kiwoom"),
    FeatureInput("d1_kospi_1m", "d7a389ba", "KOSPI", "kiwoom"),
    FeatureInput("d1_kosdaq_daily", "d17b71e2", "KOSDAQ", "kiwoom"),
    FeatureInput("d1_kosdaq_1m", "b2699d03", "KOSDAQ", "kiwoom"),
    FeatureInput("d1_nasdaq_daily", "a8489a0e", "NASDAQ", "kis"),
    FeatureInput("d1_spx_daily", "e28b4a09", "SPX", "kis"),
)


@dataclass(frozen=True, slots=True)
class MergedFeatureInput:
    dataset_id: str
    stage: str
    components: tuple[FeatureInput, ...]

    @property
    def provider(self) -> str:
        providers = {component.provider for component in self.components}
        return providers.pop() if len(providers) == 1 else "mixed"


def _feature_input(dataset_id: str) -> FeatureInput:
    return next(item for item in FEATURE_INPUTS if item.dataset_id == dataset_id)


MERGED_FEATURE_INPUTS = (
    MergedFeatureInput(
        "d2_kr-kospi-kosdaq_daily",
        "D2",
        (_feature_input("d1_kospi_daily"), _feature_input("d1_kosdaq_daily")),
    ),
    MergedFeatureInput(
        "d2_kr-kospi-kosdaq_1m",
        "D2",
        (_feature_input("d1_kospi_1m"), _feature_input("d1_kosdaq_1m")),
    ),
)


def dataset_interval(dataset_id: str) -> str:
    return "1m" if dataset_id.endswith("_1m") else "1d"

SEEDS = (7, 17, 37)
CODEBOOK_SIZES = (8, 16, 32)
HANDCRAFTED_BINS_PER_AXIS = 4
MIN_TRAIN_ROWS = max(CODEBOOK_SIZES)
BOUNDARY_POLICIES = ("include_boundary", "exclude_boundary")


def latest_feature_run(runs_root: Path, feature_input: FeatureInput) -> Path:
    cfg_dir = (
        runs_root
        / PHASE_01_ID
        / STEP_01_FEATURE_ID
        / feature_input.dataset_id
        / f"cfg-{feature_input.feature_cfg_hash}"
    )
    return latest_run(cfg_dir, seed=7)


def component_feature_runs(runs_root: Path, merged: MergedFeatureInput) -> dict[str, Path]:
    return {
        component.dataset_id: latest_feature_run(runs_root, component)
        for component in merged.components
    }


def read_merged_shape_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, run_dir in run_dirs.items():
        for row in read_shape_rows(run_dir):
            rows.append(row | {"source_dataset_id": dataset_id})
    rows.sort(key=lambda row: (row["timestamp"], row["symbol"]))
    return rows


def read_shape_rows(run_dir: Path) -> list[dict[str, Any]]:
    table_path = run_dir / "tables" / "shape_sample.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "symbol": row["symbol"],
                    "s1": float(row["s1"]),
                    "s2": float(row["s2"]),
                    "lambda_o": float(row["lambda_o"]),
                    "lambda_c": float(row["lambda_c"]),
                    "is_zero_range": row["is_zero_range"] == "True",
                    "is_boundary": row["is_boundary"] == "True",
                }
            )
    return rows


def finite_shape_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row["is_zero_range"]
        and math.isfinite(row["s1"])
        and math.isfinite(row["s2"])
        and math.isfinite(row["lambda_o"])
        and math.isfinite(row["lambda_c"])
    ]


def shape_rows_for_boundary_policy(
    rows: list[dict[str, Any]],
    *,
    boundary_policy: Literal["include_boundary", "exclude_boundary"],
) -> list[dict[str, Any]]:
    finite_rows = finite_shape_rows(rows)
    if boundary_policy == "include_boundary":
        return finite_rows
    if boundary_policy == "exclude_boundary":
        return [row for row in finite_rows if not row["is_boundary"]]
    raise ValueError(f"unsupported boundary policy: {boundary_policy}")


def shape_vectors(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[row["s1"], row["s2"]] for row in rows], dtype=float)


def split_masks(
    rows: list[dict[str, Any]], *, split: SplitProtocol | None = None
) -> dict[str, np.ndarray]:
    names = np.array([split_name(row["timestamp"], split) for row in rows], dtype=object)
    return {name: names == name for name in ("train", "validation", "test")}


def reconstruction_mse(
    vectors: np.ndarray, reconstructions: np.ndarray
) -> float | None:
    if len(vectors) == 0:
        return None
    return float(np.mean(np.sum((vectors - reconstructions) ** 2, axis=1)))


def token_stats(tokens: np.ndarray, *, codebook_size: int) -> dict[str, Any]:
    utilization = token_utilization(
        (int(token) for token in tokens.tolist()), codebook_size=codebook_size
    )
    return {
        "token_entropy": utilization.entropy,
        "effective_vocab_size": utilization.effective_vocab_size,
        "utilized_token_count": utilization.utilized_count,
        "dead_token_count": utilization.dead_count,
        "dead_token_ratio": utilization.dead_ratio,
        "histogram": {
            str(token): utilization.histogram.get(token, 0)
            for token in range(codebook_size)
        },
    }


def split_reconstruction_metrics(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    reconstructions: np.ndarray,
    tokens: np.ndarray,
    *,
    codebook_size: int,
    split: SplitProtocol | None = None,
) -> dict[str, Any]:
    masks = split_masks(rows, split=split)
    by_split = {}
    for name, mask in masks.items():
        split_tokens = tokens[mask]
        by_split[name] = {
            "row_count": int(mask.sum()),
            "reconstruction_mse": reconstruction_mse(
                vectors[mask], reconstructions[mask]
            ),
            **token_stats(split_tokens, codebook_size=codebook_size),
        }
    return by_split


def fit_kmeans(
    train_vectors: np.ndarray,
    all_vectors: np.ndarray,
    *,
    codebook_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_vectors)
    train_scaled = scaler.transform(train_vectors)
    all_scaled = scaler.transform(all_vectors)
    model = KMeans(n_clusters=codebook_size, random_state=seed, n_init=20)
    model.fit(train_scaled)
    tokens = model.predict(all_scaled)
    recon_scaled = model.cluster_centers_[tokens]
    recon = scaler.inverse_transform(recon_scaled)
    return tokens.astype(int), recon


def fit_gmm(
    train_vectors: np.ndarray,
    all_vectors: np.ndarray,
    *,
    codebook_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_vectors)
    train_scaled = scaler.transform(train_vectors)
    all_scaled = scaler.transform(all_vectors)
    model = GaussianMixture(
        n_components=codebook_size,
        covariance_type="full",
        random_state=seed,
        reg_covar=1e-6,
        max_iter=300,
    )
    model.fit(train_scaled)
    tokens = model.predict(all_scaled)
    recon_scaled = model.means_[tokens]
    recon = scaler.inverse_transform(recon_scaled)
    return tokens.astype(int), recon


def handcrafted_lambda_bins(
    rows: list[dict[str, Any]], *, bins_per_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    centers = (np.arange(bins_per_axis) + 0.5) / bins_per_axis
    tokens = []
    reconstructions = []
    for row in rows:
        lo = min(max(int(row["lambda_o"] * bins_per_axis), 0), bins_per_axis - 1)
        lc = min(max(int(row["lambda_c"] * bins_per_axis), 0), bins_per_axis - 1)
        tokens.append(lo * bins_per_axis + lc)
        reconstructions.append([float(_logit(centers[lo])), float(_logit(centers[lc]))])
    return np.array(tokens, dtype=int), np.array(reconstructions, dtype=float)


def evaluate_baselines(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    codebook_sizes: tuple[int, ...] = CODEBOOK_SIZES,
    bins_per_axis: int = HANDCRAFTED_BINS_PER_AXIS,
    split: SplitProtocol | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vectors = shape_vectors(rows)
    masks = split_masks(rows, split=split)
    train_vectors = vectors[masks["train"]]
    baseline_metrics: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for codebook_size in codebook_sizes:
        for model_name, fitter in (("kmeans", fit_kmeans), ("gmm", fit_gmm)):
            tokens, recon = fitter(
                train_vectors, vectors, codebook_size=codebook_size, seed=seed
            )
            utilization = token_stats(tokens, codebook_size=codebook_size)
            split_metrics = split_reconstruction_metrics(
                rows, vectors, recon, tokens, codebook_size=codebook_size, split=split
            )
            row = {
                "model": model_name,
                "codebook_size": codebook_size,
                "seed": seed,
                "reconstruction_mse": reconstruction_mse(vectors, recon),
                **{
                    key: value
                    for key, value in utilization.items()
                    if key != "histogram"
                },
            }
            baseline_metrics.append(
                row
                | {
                    "histogram": utilization["histogram"],
                    "split_metrics": split_metrics,
                }
            )
            baseline_rows.append(row)

    hand_tokens, hand_recon = handcrafted_lambda_bins(rows, bins_per_axis=bins_per_axis)
    hand_codebook_size = bins_per_axis**2
    hand_utilization = token_stats(hand_tokens, codebook_size=hand_codebook_size)
    hand_split = split_reconstruction_metrics(
        rows, vectors, hand_recon, hand_tokens, codebook_size=hand_codebook_size, split=split
    )
    hand_row = {
        "model": "handcrafted_lambda_bins",
        "codebook_size": hand_codebook_size,
        "seed": seed,
        "reconstruction_mse": reconstruction_mse(vectors, hand_recon),
        **{key: value for key, value in hand_utilization.items() if key != "histogram"},
    }
    baseline_metrics.append(
        hand_row
        | {"histogram": hand_utilization["histogram"], "split_metrics": hand_split}
    )
    baseline_rows.append(hand_row)
    return baseline_metrics, baseline_rows


def _baseline_config_common(interval: str) -> dict[str, Any]:
    return {
        "research": RESEARCH_NAME,
        "phase": PHASE_01_ID,
        "step": STEP_02_BASELINE_ID,
        "split": split_protocol_config_for_interval(interval),
        "preprocessing": {
            "fit_scaler_on_train_only": True,
            "zero_range_policy": "exclude_from_fit_and_count_as_special_candidate",
            "boundary_policies": BOUNDARY_POLICIES,
            "shape_vector": ["s1", "s2"],
        },
        "baselines": {
            "kmeans": {"codebook_sizes": CODEBOOK_SIZES},
            "gmm": {"codebook_sizes": CODEBOOK_SIZES},
            "handcrafted_lambda_bins": {"bins_per_axis": HANDCRAFTED_BINS_PER_AXIS},
        },
        "user_vars": {"raw_ohlcv_persisted": False},
    }


def build_baseline_config(
    feature_input: FeatureInput, input_run_dir: Path
) -> dict[str, Any]:
    input_metrics = read_json(input_run_dir / "metrics.json")
    return _baseline_config_common(dataset_interval(feature_input.dataset_id)) | {
        "dataset": asdict(feature_input),
        "input": {
            "source_step": STEP_01_FEATURE_ID,
            "feature_cfg_hash": feature_input.feature_cfg_hash,
            "feature_run_dataset_id": input_metrics["dataset_id"],
            "feature_row_count": input_metrics["row_count"],
            "feature_date_range": input_metrics["date_range"],
            "feature_requested_start_date": input_metrics.get("data_request", {}).get(
                "requested_start_date"
            ),
        },
    }


def build_merged_baseline_config(
    merged: MergedFeatureInput, input_run_dirs: dict[str, Path]
) -> dict[str, Any]:
    interval = dataset_interval(merged.dataset_id)
    component_inputs = {}
    for component in merged.components:
        input_metrics = read_json(input_run_dirs[component.dataset_id] / "metrics.json")
        component_inputs[component.dataset_id] = {
            "feature_cfg_hash": component.feature_cfg_hash,
            "feature_run_dataset_id": input_metrics["dataset_id"],
            "feature_row_count": input_metrics["row_count"],
            "feature_date_range": input_metrics["date_range"],
        }
    return _baseline_config_common(interval) | {
        "dataset": {
            "dataset_id": merged.dataset_id,
            "stage": merged.stage,
            "interval": interval,
            "merge_rule": "concatenate_component_shape_rows_sorted_by_timestamp",
            "components": [asdict(component) for component in merged.components],
        },
        "input": {
            "source_step": STEP_01_FEATURE_ID,
            "components": component_inputs,
        },
    }


def _logit(probability: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.asarray(probability) / (1.0 - np.asarray(probability)))
