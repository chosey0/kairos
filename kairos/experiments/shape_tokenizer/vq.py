from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kairos.core.model import require_torch
from kairos.experiments.protocol import SplitProtocol
from kairos.experiments.shape_tokenizer.baselines import (
    BoundaryAwareVocabulary,
    assign_boundary_aware_tokens,
    boundary_aware_fit_rows,
    build_boundary_aware_vocabulary,
    dataset_interval,
    reconstruction_mse,
    shape_vectors,
    split_masks,
    split_protocol_config_for_interval,
    token_stats,
)


@dataclass(frozen=True, slots=True)
class LatentClusteringConfig:
    codebook_size: int = 32
    hidden_dim: int = 32
    latent_dim: int = 4
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    eps: float = 1e-3

    def validate(self) -> None:
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class FSQConfig:
    levels: tuple[int, ...] = (6, 5)
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    eps: float = 1e-3

    @property
    def latent_dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        size = 1
        for level in self.levels:
            size *= level
        return size

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("levels must not be empty")
        if any(level <= 1 for level in self.levels):
            raise ValueError("all FSQ levels must be greater than 1")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class BSQConfig:
    bits: int = 5
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    eps: float = 1e-3

    @property
    def codebook_size(self) -> int:
        return 2**self.bits

    def validate(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class CoarseFineConfig:
    body_bins: int = 4
    fine_per_coarse: int = 4
    eps: float = 1e-3

    @property
    def coarse_class_count(self) -> int:
        return 2 * self.body_bins

    @property
    def codebook_size(self) -> int:
        return self.coarse_class_count * self.fine_per_coarse

    def validate(self) -> None:
        if self.body_bins <= 1:
            raise ValueError("body_bins must be greater than 1")
        if self.fine_per_coarse <= 0:
            raise ValueError("fine_per_coarse must be positive")
        if not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")


def _autoencoder_class(hidden_dim: int, latent_dim: int):
    torch, nn = require_torch()

    class ShapeAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, inputs):
            latent = self.encoder(inputs)
            reconstruction = self.decoder(latent)
            return reconstruction, latent

    return torch, ShapeAutoencoder


def _train_autoencoder(
    train_scaled: np.ndarray,
    *,
    config: LatentClusteringConfig,
    seed: int,
):
    torch, ShapeAutoencoder = _autoencoder_class(config.hidden_dim, config.latent_dim)
    torch.manual_seed(seed)
    inputs = torch.tensor(train_scaled, dtype=torch.float32)
    model = ShapeAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(inputs.size(0))
        for start in range(0, inputs.size(0), config.batch_size):
            batch = inputs[permutation[start : start + config.batch_size]]
            reconstruction, _latent = model(batch)
            loss = torch.mean((reconstruction - batch) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return model, final_loss


def _encode_decode(model, vectors_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(vectors_scaled, dtype=torch.float32)
        reconstruction, latent = model(tensor)
    return latent.cpu().numpy(), reconstruction.cpu().numpy()


def _decode_latents(model, latents: np.ndarray) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(latents, dtype=torch.float32)
        reconstruction = model.decoder(tensor)
    return reconstruction.cpu().numpy()


def fsq_quantize_numpy(
    latents: np.ndarray, *, levels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    bounded = np.clip(latents, -1.0, 1.0)
    level_array = np.array(levels, dtype=float)
    indices = np.rint(((bounded + 1.0) / 2.0) * (level_array - 1.0)).astype(int)
    indices = np.clip(indices, 0, level_array.astype(int) - 1)
    quantized = (indices / (level_array - 1.0)) * 2.0 - 1.0
    codes = _mixed_radix_codes(indices, tuple(int(level) for level in levels))
    return codes.astype(int), quantized.astype(float)


def bsq_quantize_numpy(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latents.ndim != 2:
        raise ValueError("latents must be a 2D array")
    dim = latents.shape[1]
    if dim <= 0:
        raise ValueError("latents must have at least one dimension")
    norm = np.linalg.norm(latents, axis=1, keepdims=True)
    normalized = latents / np.maximum(norm, 1e-12)
    quantized = np.where(normalized >= 0.0, 1.0, -1.0) / np.sqrt(dim)
    bits = (quantized > 0.0).astype(int)
    powers = 2 ** np.arange(dim, dtype=int)
    codes = bits @ powers
    return codes.astype(int), quantized.astype(float)


def _mixed_radix_codes(indices: np.ndarray, levels: tuple[int, ...]) -> np.ndarray:
    multipliers = np.ones(len(levels), dtype=int)
    for index in range(len(levels) - 2, -1, -1):
        multipliers[index] = multipliers[index + 1] * levels[index + 1]
    return indices @ multipliers


def _fsq_quantize_torch(latents, levels: tuple[int, ...]):
    torch, _ = require_torch()
    level_tensor = torch.tensor(levels, dtype=latents.dtype, device=latents.device)
    bounded = torch.tanh(latents)
    indices = torch.round(((bounded + 1.0) / 2.0) * (level_tensor - 1.0))
    indices = torch.clamp(indices, min=0.0)
    indices = torch.minimum(indices, level_tensor - 1.0)
    quantized = (indices / (level_tensor - 1.0)) * 2.0 - 1.0
    quantized_st = bounded + (quantized - bounded).detach()
    return quantized_st


def _bsq_quantize_torch(latents):
    torch, _ = require_torch()
    normalized = latents / torch.clamp(torch.linalg.norm(latents, dim=1, keepdim=True), min=1e-12)
    quantized = torch.where(normalized >= 0.0, 1.0, -1.0) / (latents.shape[1] ** 0.5)
    return normalized + (quantized - normalized).detach()


def _train_quantized_autoencoder(
    train_scaled: np.ndarray,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    quantizer: str,
    fsq_levels: tuple[int, ...] | None = None,
):
    torch, ShapeAutoencoder = _autoencoder_class(hidden_dim, latent_dim)
    torch.manual_seed(seed)
    inputs = torch.tensor(train_scaled, dtype=torch.float32)
    model = ShapeAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(inputs.size(0))
        for start in range(0, inputs.size(0), batch_size):
            batch = inputs[permutation[start : start + batch_size]]
            latent = model.encoder(batch)
            if quantizer == "fsq":
                if fsq_levels is None:
                    raise ValueError("fsq_levels is required for FSQ training")
                quantized = _fsq_quantize_torch(latent, fsq_levels)
            elif quantizer == "bsq":
                quantized = _bsq_quantize_torch(latent)
            else:
                raise ValueError(f"unsupported quantizer: {quantizer}")
            reconstruction = model.decoder(quantized)
            loss = torch.mean((reconstruction - batch) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return model, final_loss


def _encode_latents(model, vectors_scaled: np.ndarray) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(vectors_scaled, dtype=torch.float32)
        latent = model.encoder(tensor)
    return latent.cpu().numpy()


def fit_vqvae_latent_clustering(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    split: SplitProtocol | None = None,
    config: LatentClusteringConfig | None = None,
) -> dict[str, Any]:
    active_config = config or LatentClusteringConfig()
    active_config.validate()
    vocabulary = build_boundary_aware_vocabulary(active_config.codebook_size)
    interior_rows = boundary_aware_fit_rows(rows, eps=active_config.eps)
    masks = split_masks(interior_rows, split=split)
    train_mask = masks["train"]
    if int(train_mask.sum()) < active_config.codebook_size:
        raise ValueError(
            "not enough train interior rows for VQ-VAE latent clustering: "
            f"{int(train_mask.sum())} < {active_config.codebook_size}"
        )

    interior_vectors = shape_vectors(interior_rows)
    train_vectors = interior_vectors[train_mask]
    scaler = StandardScaler().fit(train_vectors)
    train_scaled = scaler.transform(train_vectors)
    all_scaled = scaler.transform(interior_vectors)

    model, final_loss = _train_autoencoder(train_scaled, config=active_config, seed=seed)
    latents, ae_recon_scaled = _encode_decode(model, all_scaled)
    train_latents = latents[train_mask]

    clusterer = KMeans(
        n_clusters=active_config.codebook_size,
        random_state=seed,
        n_init=20,
    )
    clusterer.fit(train_latents)
    interior_tokens = clusterer.predict(latents).astype(int)
    clustered_recon_scaled = _decode_latents(model, clusterer.cluster_centers_[interior_tokens])

    ae_recon = scaler.inverse_transform(ae_recon_scaled)
    clustered_recon = scaler.inverse_transform(clustered_recon_scaled)
    all_tokens = assign_boundary_aware_tokens(
        rows,
        interior_tokens=interior_tokens,
        vocabulary=vocabulary,
        eps=active_config.eps,
    )

    split_metrics = _interior_split_metrics(
        interior_rows,
        interior_vectors,
        clustered_recon,
        interior_tokens,
        vocabulary=vocabulary,
        split=split,
    )
    utilization = token_stats(all_tokens, codebook_size=vocabulary.size)
    boundary_count = int(
        np.isin(
            all_tokens,
            np.array(list(vocabulary.boundary_tokens.values()), dtype=int),
        ).sum()
    )
    zero_range_count = int((all_tokens == vocabulary.zero_range_token).sum())

    return {
        "model": "vqvae_latent_kmeans",
        "seed": seed,
        "codebook_size": active_config.codebook_size,
        "vocabulary_size": vocabulary.size,
        "config": asdict(active_config),
        "train_interior_row_count": int(train_mask.sum()),
        "interior_row_count": len(interior_rows),
        "row_count": len(rows),
        "final_train_loss": final_loss,
        "interior_reconstruction_mse": reconstruction_mse(interior_vectors, clustered_recon),
        "interior_autoencoder_mse": reconstruction_mse(interior_vectors, ae_recon),
        "boundary_token_count": boundary_count,
        "boundary_token_ratio": boundary_count / len(rows) if rows else None,
        "zero_range_token_count": zero_range_count,
        "zero_range_token_ratio": zero_range_count / len(rows) if rows else None,
        "token_usage": utilization,
        "split_metrics": split_metrics,
        "tokens": all_tokens,
        "interior_tokens": interior_tokens,
        "vocabulary": vocabulary,
    }


def fit_fsq(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    split: SplitProtocol | None = None,
    config: FSQConfig | None = None,
) -> dict[str, Any]:
    active_config = config or FSQConfig()
    active_config.validate()
    vocabulary = build_boundary_aware_vocabulary(active_config.codebook_size)
    interior_rows = boundary_aware_fit_rows(rows, eps=active_config.eps)
    masks = split_masks(interior_rows, split=split)
    train_mask = masks["train"]
    if int(train_mask.sum()) == 0:
        raise ValueError("not enough train interior rows for FSQ: 0")

    interior_vectors = shape_vectors(interior_rows)
    train_vectors = interior_vectors[train_mask]
    scaler = StandardScaler().fit(train_vectors)
    train_scaled = scaler.transform(train_vectors)
    all_scaled = scaler.transform(interior_vectors)

    model, final_loss = _train_quantized_autoencoder(
        train_scaled,
        hidden_dim=active_config.hidden_dim,
        latent_dim=active_config.latent_dim,
        epochs=active_config.epochs,
        batch_size=active_config.batch_size,
        learning_rate=active_config.learning_rate,
        seed=seed,
        quantizer="fsq",
        fsq_levels=active_config.levels,
    )
    latents = np.tanh(_encode_latents(model, all_scaled))
    interior_tokens, quantized = fsq_quantize_numpy(latents, levels=active_config.levels)
    recon_scaled = _decode_latents(model, quantized)
    recon = scaler.inverse_transform(recon_scaled)
    return _candidate_result(
        model_name="fsq",
        rows=rows,
        interior_rows=interior_rows,
        interior_vectors=interior_vectors,
        recon=recon,
        interior_tokens=interior_tokens,
        vocabulary=vocabulary,
        seed=seed,
        train_mask=train_mask,
        split=split,
        config=asdict(active_config)
        | {"latent_dim": active_config.latent_dim, "codebook_size": active_config.codebook_size},
        final_train_loss=final_loss,
        eps=active_config.eps,
    )


def fit_bsq(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    split: SplitProtocol | None = None,
    config: BSQConfig | None = None,
) -> dict[str, Any]:
    active_config = config or BSQConfig()
    active_config.validate()
    vocabulary = build_boundary_aware_vocabulary(active_config.codebook_size)
    interior_rows = boundary_aware_fit_rows(rows, eps=active_config.eps)
    masks = split_masks(interior_rows, split=split)
    train_mask = masks["train"]
    if int(train_mask.sum()) == 0:
        raise ValueError("not enough train interior rows for BSQ: 0")

    interior_vectors = shape_vectors(interior_rows)
    train_vectors = interior_vectors[train_mask]
    scaler = StandardScaler().fit(train_vectors)
    train_scaled = scaler.transform(train_vectors)
    all_scaled = scaler.transform(interior_vectors)

    model, final_loss = _train_quantized_autoencoder(
        train_scaled,
        hidden_dim=active_config.hidden_dim,
        latent_dim=active_config.bits,
        epochs=active_config.epochs,
        batch_size=active_config.batch_size,
        learning_rate=active_config.learning_rate,
        seed=seed,
        quantizer="bsq",
    )
    latents = _encode_latents(model, all_scaled)
    interior_tokens, quantized = bsq_quantize_numpy(latents)
    recon_scaled = _decode_latents(model, quantized)
    recon = scaler.inverse_transform(recon_scaled)
    return _candidate_result(
        model_name="bsq",
        rows=rows,
        interior_rows=interior_rows,
        interior_vectors=interior_vectors,
        recon=recon,
        interior_tokens=interior_tokens,
        vocabulary=vocabulary,
        seed=seed,
        train_mask=train_mask,
        split=split,
        config=asdict(active_config) | {"codebook_size": active_config.codebook_size},
        final_train_loss=final_loss,
        eps=active_config.eps,
    )


def coarse_body_quantile_thresholds(
    rows: list[dict[str, Any]], *, body_bins: int = 4
) -> np.ndarray:
    if body_bins <= 1:
        raise ValueError("body_bins must be greater than 1")
    body = np.array([abs(float(row["lambda_c"]) - float(row["lambda_o"])) for row in rows])
    if len(body) == 0:
        raise ValueError("rows must not be empty")
    return np.quantile(body, np.arange(1, body_bins) / body_bins)


def coarse_class_ids(
    rows: list[dict[str, Any]],
    *,
    thresholds: np.ndarray,
    body_bins: int = 4,
) -> np.ndarray:
    classes: list[int] = []
    for row in rows:
        direction = 1 if float(row["lambda_c"]) >= float(row["lambda_o"]) else 0
        body = abs(float(row["lambda_c"]) - float(row["lambda_o"]))
        body_bin = int(np.searchsorted(thresholds, body, side="right"))
        body_bin = min(max(body_bin, 0), body_bins - 1)
        classes.append(direction * body_bins + body_bin)
    return np.array(classes, dtype=int)


def fit_coarse_fine(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    split: SplitProtocol | None = None,
    config: CoarseFineConfig | None = None,
) -> dict[str, Any]:
    active_config = config or CoarseFineConfig()
    active_config.validate()
    vocabulary = build_boundary_aware_vocabulary(active_config.codebook_size)
    interior_rows = boundary_aware_fit_rows(rows, eps=active_config.eps)
    masks = split_masks(interior_rows, split=split)
    train_mask = masks["train"]
    if int(train_mask.sum()) == 0:
        raise ValueError("not enough train interior rows for coarse_fine: 0")

    interior_vectors = shape_vectors(interior_rows)
    train_rows = [row for row, is_train in zip(interior_rows, train_mask, strict=True) if is_train]
    thresholds = coarse_body_quantile_thresholds(
        train_rows, body_bins=active_config.body_bins
    )
    coarse_ids = coarse_class_ids(
        interior_rows,
        thresholds=thresholds,
        body_bins=active_config.body_bins,
    )
    scaler = StandardScaler().fit(interior_vectors[train_mask])
    scaled_vectors = scaler.transform(interior_vectors)

    interior_tokens = np.full(len(interior_rows), -1, dtype=int)
    recon_scaled = np.full_like(scaled_vectors, np.nan, dtype=float)
    degenerate_classes: list[int] = []
    for coarse_id in range(active_config.coarse_class_count):
        all_indices = np.flatnonzero(coarse_ids == coarse_id)
        if len(all_indices) == 0:
            continue
        train_indices = np.flatnonzero(train_mask & (coarse_ids == coarse_id))
        token_offset = coarse_id * active_config.fine_per_coarse
        if len(train_indices) < active_config.fine_per_coarse:
            source_index = int(train_indices[0]) if len(train_indices) else int(all_indices[0])
            interior_tokens[all_indices] = token_offset
            recon_scaled[all_indices] = scaled_vectors[source_index]
            degenerate_classes.append(coarse_id)
            continue
        model = KMeans(
            n_clusters=active_config.fine_per_coarse,
            random_state=seed,
            n_init=20,
        )
        model.fit(scaled_vectors[train_indices])
        fine_ids = model.predict(scaled_vectors[all_indices]).astype(int)
        interior_tokens[all_indices] = token_offset + fine_ids
        recon_scaled[all_indices] = model.cluster_centers_[fine_ids]

    if np.any(interior_tokens < 0):
        raise RuntimeError("coarse_fine left interior rows unassigned")
    recon = scaler.inverse_transform(recon_scaled)
    return _candidate_result(
        model_name="coarse_fine",
        rows=rows,
        interior_rows=interior_rows,
        interior_vectors=interior_vectors,
        recon=recon,
        interior_tokens=interior_tokens,
        vocabulary=vocabulary,
        seed=seed,
        train_mask=train_mask,
        split=split,
        config=asdict(active_config)
        | {
            "codebook_size": active_config.codebook_size,
            "coarse_class_count": active_config.coarse_class_count,
            "body_quantile_thresholds": thresholds.tolist(),
            "degenerate_classes": degenerate_classes,
        },
        final_train_loss=None,
        eps=active_config.eps,
    )


def _candidate_result(
    *,
    model_name: str,
    rows: list[dict[str, Any]],
    interior_rows: list[dict[str, Any]],
    interior_vectors: np.ndarray,
    recon: np.ndarray,
    interior_tokens: np.ndarray,
    vocabulary: BoundaryAwareVocabulary,
    seed: int,
    train_mask: np.ndarray,
    split: SplitProtocol | None,
    config: dict[str, Any],
    final_train_loss: float | None,
    eps: float,
) -> dict[str, Any]:
    all_tokens = assign_boundary_aware_tokens(
        rows,
        interior_tokens=interior_tokens,
        vocabulary=vocabulary,
        eps=eps,
    )
    split_metrics = _interior_split_metrics(
        interior_rows,
        interior_vectors,
        recon,
        interior_tokens,
        vocabulary=vocabulary,
        split=split,
    )
    utilization = token_stats(all_tokens, codebook_size=vocabulary.size)
    boundary_count = int(
        np.isin(
            all_tokens,
            np.array(list(vocabulary.boundary_tokens.values()), dtype=int),
        ).sum()
    )
    zero_range_count = int((all_tokens == vocabulary.zero_range_token).sum())
    return {
        "model": model_name,
        "seed": seed,
        "codebook_size": vocabulary.continuous_codebook_size,
        "vocabulary_size": vocabulary.size,
        "config": config,
        "train_interior_row_count": int(train_mask.sum()),
        "interior_row_count": len(interior_rows),
        "row_count": len(rows),
        "final_train_loss": final_train_loss,
        "interior_reconstruction_mse": reconstruction_mse(interior_vectors, recon),
        "boundary_token_count": boundary_count,
        "boundary_token_ratio": boundary_count / len(rows) if rows else None,
        "zero_range_token_count": zero_range_count,
        "zero_range_token_ratio": zero_range_count / len(rows) if rows else None,
        "token_usage": utilization,
        "split_metrics": split_metrics,
        "tokens": all_tokens,
        "interior_tokens": interior_tokens,
        "vocabulary": vocabulary,
    }


def _interior_split_metrics(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    reconstructions: np.ndarray,
    tokens: np.ndarray,
    *,
    vocabulary: BoundaryAwareVocabulary,
    split: SplitProtocol | None = None,
) -> dict[str, Any]:
    masks = split_masks(rows, split=split)
    by_split: dict[str, Any] = {}
    for name, mask in masks.items():
        split_tokens = tokens[mask]
        by_split[name] = {
            "interior_row_count": int(mask.sum()),
            "interior_reconstruction_mse": reconstruction_mse(
                vectors[mask], reconstructions[mask]
            ),
            **token_stats(
                split_tokens,
                codebook_size=vocabulary.continuous_codebook_size,
            ),
        }
    return by_split


def token_share_by_symbol(
    rows: list[dict[str, Any]],
    tokens: np.ndarray,
    *,
    vocabulary_size: int,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = {}
    for row, token in zip(rows, tokens, strict=True):
        grouped.setdefault(str(row["symbol"]), []).append(int(token))
    return {
        symbol: {
            str(index): float(count / len(symbol_tokens))
            for index, count in enumerate(
                np.bincount(symbol_tokens, minlength=vocabulary_size).tolist()
            )
            if count
        }
        for symbol, symbol_tokens in grouped.items()
    }


def build_vq_config(dataset_id: str, config: LatentClusteringConfig | None = None) -> dict[str, Any]:
    active_config = config or LatentClusteringConfig()
    active_config.validate()
    interval = dataset_interval(dataset_id)
    return {
        "dataset_id": dataset_id,
        "split": split_protocol_config_for_interval(interval),
        "model": "vqvae_latent_kmeans",
        "boundary_handling": {
            "continuous_codebook_size": active_config.codebook_size,
            "boundary_discrete_tokens": 8,
            "zero_range_special_tokens": 1,
            "total_vocabulary_size": active_config.codebook_size + 9,
            "fit_rows": "train split and interior x interior only",
        },
        "training": asdict(active_config),
        "leakage_checks": {
            "time_split_fixed_before_fit": True,
            "autoencoder_fit_on_train_interior_only": True,
            "scaler_fit_on_train_interior_only": True,
            "latent_kmeans_fit_on_train_interior_only": True,
            "random_split_forbidden": True,
        },
    }


def build_step03_model_configs(
    dataset_id: str,
    *,
    vq_config: LatentClusteringConfig | None = None,
    fsq_config: FSQConfig | None = None,
    bsq_config: BSQConfig | None = None,
    coarse_fine_config: CoarseFineConfig | None = None,
) -> dict[str, Any]:
    vq_active = vq_config or LatentClusteringConfig()
    fsq_active = fsq_config or FSQConfig()
    bsq_active = bsq_config or BSQConfig()
    coarse_active = coarse_fine_config or CoarseFineConfig()
    vq_active.validate()
    fsq_active.validate()
    bsq_active.validate()
    coarse_active.validate()
    interval = dataset_interval(dataset_id)
    split = split_protocol_config_for_interval(interval)
    boundary = {
        "boundary_discrete_tokens": 8,
        "zero_range_special_tokens": 1,
        "fit_rows": "train split and interior x interior only",
    }
    return {
        "dataset_id": dataset_id,
        "split": split,
        "models": {
            "kmeans_boundary_aware": {
                "continuous_codebook_size": 32,
                "total_vocabulary_size": 41,
                "boundary_handling": boundary,
            },
            "vqvae_latent_kmeans": build_vq_config(dataset_id, vq_active),
            "fsq": {
                "model": "fsq",
                "continuous_codebook_size": fsq_active.codebook_size,
                "total_vocabulary_size": fsq_active.codebook_size + 9,
                "training": asdict(fsq_active)
                | {
                    "latent_dim": fsq_active.latent_dim,
                    "codebook_size": fsq_active.codebook_size,
                },
                "boundary_handling": boundary,
            },
            "bsq": {
                "model": "bsq",
                "continuous_codebook_size": bsq_active.codebook_size,
                "total_vocabulary_size": bsq_active.codebook_size + 9,
                "training": asdict(bsq_active)
                | {"codebook_size": bsq_active.codebook_size},
                "boundary_handling": boundary,
            },
            "coarse_fine": {
                "model": "coarse_fine",
                "continuous_codebook_size": coarse_active.codebook_size,
                "total_vocabulary_size": coarse_active.codebook_size + 9,
                "training": asdict(coarse_active)
                | {
                    "coarse_class_count": coarse_active.coarse_class_count,
                    "codebook_size": coarse_active.codebook_size,
                },
                "boundary_handling": boundary,
            },
        },
        "leakage_checks": {
            "time_split_fixed_before_fit": True,
            "scaler_fit_on_train_interior_only": True,
            "model_fit_on_train_interior_only": True,
            "random_split_forbidden": True,
        },
    }
