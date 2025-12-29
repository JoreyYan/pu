"""Different methods for positional embeddings. These are not essential for understanding DDPMs, but are relevant for the ablation study."""

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class FourierPositionEncoding(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        include_input: bool = False,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                min_freq, max_freq, steps=N_freqs
            )  # (nf,)
        else:
            freq_bands = torch.linspace(
                2.0**min_freq, 2.0**max_freq, steps=N_freqs
            )  # (nf,)

        assert (
            freq_bands.isfinite().all()
        ), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.numel() * 2

    def forward(
        self,
        pos: torch.Tensor,
    ):
        """
        Get the positional encoding for each coordinate.
        Args:
            pos:
                (*, in_dim)
        Returns:
            out:
                (*, in_dimitional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (*, in_dim)

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        out = torch.cat(out, dim=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return out
