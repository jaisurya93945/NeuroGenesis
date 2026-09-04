"""Concept encoders. Deliberately small: the science is about grounding, not scale."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """LeNet-scale CNN over 28x28 greyscale digits (~62k params).

    Chosen because it trains to published MNIST-Addition concept accuracy in
    minutes on a CPU core, which is what makes a multi-hundred-run experiment
    matrix honest rather than aspirational.
    """

    def __init__(self, k: int, in_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 28, 28) -> (B, k)`` logits."""
        return self.net(x)


class MLPEncoder(nn.Module):
    """MLP over synthetic vector inputs (Tier S), where perceptual difficulty is a knob."""

    def __init__(self, k: int, in_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim) -> (B, k)`` logits."""
        return self.net(x)


def build_encoder(kind: str, k: int, **kwargs) -> nn.Module:
    """Factory so configs can name an encoder without importing torch classes."""
    if kind == "cnn":
        return SmallCNN(k=k, **kwargs)
    if kind == "mlp":
        return MLPEncoder(k=k, **kwargs)
    raise ValueError(f"unknown encoder {kind!r}")
