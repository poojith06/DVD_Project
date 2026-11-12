"""Shared neural-network trainer (same recipe as Final_models and the benchmark).

Huber loss (beta 0.05 decades) on a standardised log10|I| target, AdamW lr 1e-3 weight decay 1e-4,
cosine learning rate, batch 128, 500 epochs; keeps the epoch with the lowest validation loss.
Trains on the GPU if there is one; the trained network is kept on the CPU for prediction and saving.
"""
import copy

import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET_PARAMS = {"epochs": 500, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "huber_beta": 0.05}


def mlp(n_in, n_out, hidden=(256, 128, 64), act=nn.SiLU):
    layers, size = [], n_in
    for width in hidden:
        layers += [nn.Linear(size, width), act()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, n_out))


class TrainedNet:
    """A trained network plus the target scaling: predict(X) returns log10|I| (decades)."""

    def __init__(self, net, y_mean, y_std):
        self.net, self.y_mean, self.y_std = net.cpu().eval(), y_mean, y_std

    def predict(self, X):
        with torch.no_grad():
            z = self.net(torch.as_tensor(np.asarray(X, float), dtype=torch.float32)).numpy()
        return self.y_mean + self.y_std * z


def train_net(make, X_train, y_train, X_val, y_val, seed=0, params=None):
    """Fit make(n_inputs, n_outputs) to y_train (decades, shape (rows, outputs))."""
    p = {**NET_PARAMS, **(params or {})}
    torch.manual_seed(seed)
    mu, sd = y_train.mean(0), y_train.std(0) + 1e-12

    def tensor(a):
        return torch.tensor(np.asarray(a, float), dtype=torch.float32, device=DEVICE)

    X, y, Xv, yv, MU, SD = tensor(X_train), tensor((y_train - mu) / sd), tensor(X_val), tensor(y_val), tensor(mu), tensor(sd)
    net = make(X.shape[1], y.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p["epochs"])
    best, best_state = float("inf"), None
    for _ in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            loss = nn.functional.smooth_l1_loss(MU + SD * net(X[idx]), MU + SD * y[idx], beta=p["huber_beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        with torch.no_grad():
            val_loss = nn.functional.smooth_l1_loss(MU + SD * net(Xv), yv, beta=p["huber_beta"]).item()
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return TrainedNet(net, mu, sd)
