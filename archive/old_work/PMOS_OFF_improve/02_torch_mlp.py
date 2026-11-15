"""PyTorch MLP variants for PMOS OFF (log10|I| targets), validation set only. Saves validation predictions per config.
Usage: python 02_torch_mlp.py <config_name> [seeds]"""
import sys, time, json
import numpy as np, torch
from torch import nn
from common_setup import load, summ, HERE, TARGET_COLUMNS
torch.set_num_threads(4)

CONFIGS = {
  # name: hidden layers, epochs, lr, weight decay, loss, batch, which outputs
  "repo_like":   dict(hidden=(256, 128, 64), epochs=500, lr=1e-3, wd=1e-4, loss="huber", bs=128),
  "wide_long":   dict(hidden=(512, 512, 256, 128), epochs=1500, lr=2e-3, wd=1e-5, loss="mse", bs=256),
  "deep_long":   dict(hidden=(256,) * 6, epochs=1500, lr=2e-3, wd=1e-5, loss="mse", bs=256),
  "wide_long_l1": dict(hidden=(512, 512, 256, 128), epochs=1500, lr=2e-3, wd=1e-5, loss="l1", bs=256),
}

def train(cfg, X, Y, Xv, seed, out_idx=None):
    torch.manual_seed(seed); np.random.seed(seed)
    out_idx = list(range(Y.shape[1])) if out_idx is None else out_idx
    y = Y[:, out_idx]; mu, sd = y.mean(0), y.std(0)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    layers, n = [], X.shape[1]
    for h in cfg["hidden"]: layers += [nn.Linear(n, h), nn.SiLU()]; n = h
    net = nn.Sequential(*layers, nn.Linear(n, len(out_idx)))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * int(np.ceil(len(X) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    lossf = {"mse": nn.MSELoss(), "huber": nn.HuberLoss(delta=0.05), "l1": nn.L1Loss()}[cfg["loss"]]
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["bs"]):
            b = perm[i:i + cfg["bs"]]; opt.zero_grad(); l = lossf(net(Xt[b]), yt[b]); l.backward(); opt.step(); sched.step()
    with torch.no_grad(): return net(torch.tensor(Xv, dtype=torch.float32)).numpy() * sd + mu

if __name__ == "__main__":
    name = sys.argv[1]; seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    X, Y, I, _, _ = load("remeasured"); cfg = CONFIGS[name]
    preds = []
    for s in range(seeds):
        t0 = time.time(); preds.append(train(cfg, X["tr"], Y["tr"], X["va"], seed=s))
        S = summ(preds[-1], I["va"]); E = summ(np.mean(preds, 0), I["va"])
        print(f"{name} seed {s} ({time.time()-t0:.0f}s): single " + "  ".join(f"{t} {S[t][0]:.2f}/{S[t][1]:.1f}" for t in TARGET_COLUMNS)
              + f" | ensemble of {len(preds)}: " + "  ".join(f"{t} {E[t][0]:.2f}/{E[t][1]:.1f}" for t in TARGET_COLUMNS), flush=True)
    np.save(HERE / "val_preds" / f"torch_{name}_x{seeds}.npy", np.mean(preds, 0))
