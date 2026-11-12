"""ResNet-style MLP (skip connections), one network per output group: (id, is), ig, ib.

Architecture: Linear(20 -> 256), 3 residual blocks h = h + Linear(SiLU(Linear(SiLU(h)))), SiLU, Linear(256 -> outputs).
Training: common/nets.py (Huber 0.05 decades, AdamW, cosine LR, batch 128, 500 epochs, best validation epoch).
Idea from the IIITH ResNet-MLP leakage work on PTM 22nm HP (Agarwal/Jain/Amuru/Abbas).
"""
import numpy as np
from torch import nn

from common.config import GROUPS, MODEL_SEED
from common.nets import train_net


class ResMLP(nn.Module):
    def __init__(self, n_in, n_out, width=256, blocks=3):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(blocks)
        ])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, n_out))

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class ResNetMLPModel:
    name = "resnet_mlp"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed, self.params = case, seed, params

    def fit(self, X_train, y_train, X_val, y_val, df_train=None, df_val=None):
        self.nets = {
            group: train_net(ResMLP, X_train, y_train[:, cols], X_val, y_val[:, cols], self.seed + i, self.params)
            for i, (group, cols) in enumerate(GROUPS.items())
        }
        return self

    def predict(self, X, df=None):
        out = np.zeros((len(X), 4))
        for group, cols in GROUPS.items():
            out[:, cols] = self.nets[group].predict(X)
        return out
