"""模倣学習の方策（多層パーセプトロン）と、観測の正規化。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .env import MAX_DELTA, OBS_DIM


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """環境へ渡す行動（手先の変位 [m] 3 次元、把持 ±1）を返す。"""
        y = self.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])


def to_target(act: np.ndarray) -> np.ndarray:
    """環境の行動を、学習用の目標（各次元が約 -1〜1）に直す。"""
    y = act.copy()
    y[:, :3] /= MAX_DELTA
    return y


def load_policy(path: str) -> MLPPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = MLPPolicy()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
