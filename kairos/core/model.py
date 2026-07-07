from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - exercised when optional dependency is installed
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - default lightweight environment
    torch = None
    nn = None


@dataclass(frozen=True, slots=True)
class VQVAEConfig:
    input_dim: int = 2
    hidden_dim: int = 64
    latent_dim: int = 16
    codebook_size: int = 32
    commitment_cost: float = 0.25

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if self.commitment_cost < 0:
            raise ValueError("commitment_cost must be non-negative")


def require_torch():
    if torch is None or nn is None:
        raise RuntimeError(
            "research.tokenizers VQ-VAE requires the optional 'tokenizers' dependencies "
            "(for example: uv sync --extra tokenizers)."
        )
    return torch, nn


if nn is not None:  # pragma: no cover - optional ML path

    class Encoder(nn.Module):
        def __init__(self, config: VQVAEConfig) -> None:
            super().__init__()
            config.validate()
            self.net = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.latent_dim),
            )

        def forward(self, inputs):
            return self.net(inputs)


    class Decoder(nn.Module):
        def __init__(self, config: VQVAEConfig) -> None:
            super().__init__()
            config.validate()
            self.net = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.input_dim),
            )

        def forward(self, inputs):
            return self.net(inputs)


    class VectorQuantizer(nn.Module):
        def __init__(self, config: VQVAEConfig) -> None:
            super().__init__()
            config.validate()
            self.codebook_size = config.codebook_size
            self.embedding = nn.Embedding(config.codebook_size, config.latent_dim)
            self.embedding.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)

        def forward(self, z_e):
            distances = (
                z_e.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1)
            )
            indices = distances.argmin(dim=1)
            z_q = self.embedding(indices)
            z_q_st = z_e + (z_q - z_e).detach()
            return z_q_st, z_q, indices


    class VQVAE(nn.Module):
        def __init__(self, config: VQVAEConfig | None = None) -> None:
            super().__init__()
            self.config = config or VQVAEConfig()
            self.config.validate()
            self.encoder = Encoder(self.config)
            self.quantizer = VectorQuantizer(self.config)
            self.decoder = Decoder(self.config)

        def forward(self, inputs):
            z_e = self.encoder(inputs)
            z_q_st, z_q, indices = self.quantizer(z_e)
            reconstruction = self.decoder(z_q_st)
            return reconstruction, z_e, z_q_st, z_q, indices

else:
    Encoder = None
    Decoder = None
    VectorQuantizer = None
    VQVAE = None
