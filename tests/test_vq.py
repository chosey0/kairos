import numpy as np
import pytest

from kairos.experiments.shape_tokenizer.vq import (
    BSQConfig,
    CoarseFineConfig,
    FSQConfig,
    LatentClusteringConfig,
    bsq_quantize_numpy,
    coarse_body_quantile_thresholds,
    coarse_class_ids,
    fit_bsq,
    fit_coarse_fine,
    fit_fsq,
    fit_vqvae_latent_clustering,
    fsq_quantize_numpy,
    token_share_by_symbol,
)


def shape_row(
    day: int,
    s1: float,
    s2: float,
    lambda_o: float,
    lambda_c: float,
    *,
    zero: bool = False,
    boundary: bool = False,
    symbol: str = "TEST",
) -> dict:
    return {
        "timestamp": f"2016-01-{day:02d}",
        "symbol": symbol,
        "s1": s1,
        "s2": s2,
        "lambda_o": lambda_o,
        "lambda_c": lambda_c,
        "is_zero_range": zero,
        "is_boundary": boundary,
    }


def test_vqvae_latent_clustering_assigns_all_rows_with_boundary_tokens() -> None:
    rows = [
        shape_row(day, float(day % 2), float((day + 1) % 2), 0.35 + (day % 2) * 0.2, 0.45)
        for day in range(1, 13)
    ]
    rows += [
        shape_row(13, -6.9, 6.9, 0.0, 1.0, boundary=True),
        shape_row(14, 0.0, 0.0, 0.5, 0.5, zero=True),
    ]
    config = LatentClusteringConfig(
        codebook_size=2,
        hidden_dim=8,
        latent_dim=2,
        epochs=2,
        batch_size=4,
    )

    result = fit_vqvae_latent_clustering(rows, seed=7, config=config)

    assert result["vocabulary_size"] == 11
    assert len(result["tokens"]) == len(rows)
    assert result["train_interior_row_count"] == 12
    assert result["boundary_token_count"] == 1
    assert result["zero_range_token_count"] == 1
    assert result["interior_reconstruction_mse"] is not None
    assert np.all(result["tokens"] >= 0)


def test_token_share_by_symbol_returns_sparse_normalized_distribution() -> None:
    rows = [
        shape_row(1, 0.0, 0.0, 0.5, 0.5, symbol="A"),
        shape_row(2, 0.0, 0.0, 0.5, 0.5, symbol="A"),
        shape_row(3, 0.0, 0.0, 0.5, 0.5, symbol="B"),
    ]

    shares = token_share_by_symbol(rows, np.array([0, 1, 1]), vocabulary_size=3)

    assert shares["A"] == {"0": pytest.approx(0.5), "1": pytest.approx(0.5)}
    assert shares["B"] == {"1": pytest.approx(1.0)}


def test_fsq_rounding_stays_inside_levels() -> None:
    latents = np.array([[-2.0, -0.2], [0.0, 0.4], [2.0, 2.0]])

    codes, quantized = fsq_quantize_numpy(latents, levels=(6, 5))

    assert codes.min() >= 0
    assert codes.max() < 30
    assert len(set(codes.tolist())) <= 30
    assert np.all(quantized >= -1.0)
    assert np.all(quantized <= 1.0)
    assert len(np.unique(quantized[:, 0])) <= 6
    assert len(np.unique(quantized[:, 1])) <= 5


def test_bsq_codes_are_binary_spherical_vertices() -> None:
    latents = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0], [1.0, -1.0, 1.0, -1.0, 1.0]])

    codes, quantized = bsq_quantize_numpy(latents)

    expected_abs = 1 / np.sqrt(5)
    assert codes.min() >= 0
    assert codes.max() < 32
    assert len(set(codes.tolist())) <= 32
    assert set(np.unique(np.round(np.abs(quantized), 12)).tolist()) == {round(expected_abs, 12)}
    assert set(np.unique(np.sign(quantized)).tolist()) <= {-1.0, 1.0}


def candidate_rows() -> list[dict]:
    rows = [
        shape_row(
            day,
            float(day % 3) / 2.0,
            float((day + 1) % 3) / 2.0,
            0.25 + (day % 4) * 0.1,
            0.35 + (day % 3) * 0.1,
        )
        for day in range(1, 25)
    ]
    rows += [
        shape_row(25, -6.9, 6.9, 0.0, 1.0, boundary=True),
        shape_row(26, 0.0, 0.0, 0.5, 0.5, zero=True),
    ]
    return rows


@pytest.mark.parametrize(
    ("fit_fn", "config", "expected_codebook_size"),
    [
        (
            fit_fsq,
            FSQConfig(levels=(3, 2), hidden_dim=8, epochs=2, batch_size=4),
            6,
        ),
        (
            fit_bsq,
            BSQConfig(bits=3, hidden_dim=8, epochs=2, batch_size=4),
            8,
        ),
        (
            fit_coarse_fine,
            CoarseFineConfig(body_bins=4, fine_per_coarse=2),
            16,
        ),
    ],
)
def test_candidates_assign_every_row_and_preserve_boundary_wrapper(
    fit_fn, config, expected_codebook_size
) -> None:
    rows = candidate_rows()

    result = fit_fn(rows, seed=7, config=config)

    assert result["codebook_size"] == expected_codebook_size
    assert result["vocabulary_size"] == expected_codebook_size + 9
    assert len(result["tokens"]) == len(rows)
    assert np.all(result["tokens"] >= 0)
    assert result["tokens"][-2] in result["vocabulary"].boundary_tokens.values()
    assert result["tokens"][-1] == result["vocabulary"].zero_range_token
    assert result["boundary_token_count"] == 1
    assert result["zero_range_token_count"] == 1


def test_coarse_fine_class_boundaries_and_degenerate_empty_train_class() -> None:
    train_rows = [
        shape_row(day, 0.0, 0.0, 0.20, 0.20 + day * 0.02)
        for day in range(1, 9)
    ]
    test_row = {**shape_row(9, 0.0, 0.0, 0.80, 0.10), "timestamp": "2021-01-01"}
    rows = train_rows + [test_row]

    thresholds = coarse_body_quantile_thresholds(train_rows, body_bins=4)
    classes = coarse_class_ids(rows, thresholds=thresholds, body_bins=4)
    result = fit_coarse_fine(
        rows,
        seed=7,
        config=CoarseFineConfig(body_bins=4, fine_per_coarse=4),
    )

    assert classes.min() >= 0
    assert classes.max() <= 7
    assert result["interior_tokens"].min() >= 0
    assert result["interior_tokens"].max() <= 31
    assert result["config"]["degenerate_classes"]
    assert len(result["tokens"]) == len(rows)
