from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .features import ShapeFeatures
from .model import VQVAEConfig, require_torch


@dataclass(frozen=True, slots=True)
class TrainConfig:
    output_dir: Path
    model: VQVAEConfig = VQVAEConfig()
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42

    def validate(self) -> None:
        self.model.validate()
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class TrainResult:
    checkpoint_path: Path
    epochs: int
    final_loss: float


def train(features: Sequence[ShapeFeatures], *, config: TrainConfig) -> TrainResult:
    """Train a small VQ-VAE tokenizer on the 2D shape core and write a safe checkpoint."""
    config.validate()
    if not features:
        raise ValueError("features must not be empty")
    if any(feature.is_zero_range for feature in features):
        raise ValueError(
            "zero-range candles must be excluded (or assigned a special token) before training"
        )

    torch, _ = require_torch()
    from .model import VQVAE  # noqa: PLC0415

    if VQVAE is None:  # defensive; require_torch already covers this branch
        raise RuntimeError("VQVAE is unavailable")

    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = torch.tensor(
        [feature.as_tuple() for feature in features], dtype=torch.float32
    )
    model = VQVAE(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    final_loss = 0.0
    for _ in range(config.epochs):
        permutation = torch.randperm(inputs.size(0))
        for start in range(0, inputs.size(0), config.batch_size):
            batch = inputs[permutation[start : start + config.batch_size]]
            reconstruction, z_e, _z_q_st, z_q, _indices = model(batch)
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            codebook_loss = torch.mean((z_q - z_e.detach()) ** 2)
            commitment_loss = torch.mean((z_e - z_q.detach()) ** 2)
            loss = (
                reconstruction_loss
                + codebook_loss
                + config.model.commitment_cost * commitment_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    checkpoint_path = config.output_dir / "tokenizer.pt"
    torch.save(
        {
            "format_version": 2,
            "config": asdict(config.model),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return TrainResult(
        checkpoint_path=checkpoint_path, epochs=config.epochs, final_loss=final_loss
    )
