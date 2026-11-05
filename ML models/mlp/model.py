"""Multilayer perceptron: one network predicting log10|I| for all four terminals.

Shared estimator interface:
    fit(X_train, y_train, X_val, y_val, signs_train) -> self
    predict(X) -> array of shape (n_samples, 4)
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from common.config import SEED

MLP_PARAMS = {
    "hidden": (256, 128, 64),
    "epochs": 500,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "huber_beta": 0.05,  # in decades of current
    "kcl_weight": 0.02,
}


class _Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden, y_mean, y_std):
        super().__init__()
        layers, size = [], n_inputs
        for width in hidden:
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        layers.append(nn.Linear(size, n_outputs))
        self.layers = nn.Sequential(*layers)
        # The network learns standardised values, but its output is in decades, so
        # the Huber threshold keeps its physical meaning.
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, x):
        return self.y_mean + self.y_std * self.layers(x)


def _kirchhoff_penalty(log_pred, signs):
    """Mean of |sum I| / sum |I|, computed relative to each row's largest current."""
    magnitudes = 10.0 ** (log_pred - log_pred.max(dim=1, keepdim=True).values.detach())
    return ((signs * magnitudes).sum(dim=1).abs() / magnitudes.sum(dim=1)).mean()


class MLPModel:
    """One network for all four terminals, trained with a Kirchhoff penalty."""

    name = "mlp"

    def __init__(self, params=None, seed=SEED):
        self.params = {**MLP_PARAMS, **(params or {})}
        self.seed = seed

    def fit(self, X_train, y_train, X_val, y_val, signs_train):
        p = self.params
        torch.manual_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def tensor(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

        X, y, signs = tensor(X_train), tensor(y_train), tensor(signs_train)
        X_v, y_v = tensor(X_val), tensor(y_val)

        net = _Network(
            X.shape[1], y.shape[1], p["hidden"], y_train.mean(axis=0), y_train.std(axis=0)
        ).to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p["epochs"])

        best_loss, best_state = float("inf"), None
        for _ in range(p["epochs"]):
            net.train()
            for batch in torch.randperm(len(X), device=device).split(p["batch_size"]):
                pred = net(X[batch])
                loss = smooth_l1_loss(pred, y[batch], beta=p["huber_beta"])
                loss = loss + p["kcl_weight"] * _kirchhoff_penalty(pred, signs[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                val_loss = smooth_l1_loss(net(X_v), y_v, beta=p["huber_beta"]).item()
            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(net.state_dict())

        net.load_state_dict(best_state)
        self.net = net.cpu().eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            return self.net(x).numpy().astype(float)
