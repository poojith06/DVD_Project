"""Train an ensemble of E small MLPs at once on the GPU (batched matrix multiplies, one weight set per network).
Same recipe as 03_mlp_ensemble.py: Huber loss (beta in target units), AdamW lr 1e-3 wd 1e-4, cosine LR, batch 128,
each network keeps its own best epoch on the validation loss. Falls back to the CPU if no GPU is available.

    from gpu_ensemble import train_ensemble
    val_pred, test_pred = train_ensemble(Xtr, ytr, Xva, yva, Xte, E=16, hidden=(256, 128, 64), epochs=500, seed=0)
    # predictions have shape [E, rows, outputs]; average over axis 0 for the ensemble
"""
import time
import numpy as np, torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BatchedMLP(nn.Module):
    def __init__(self, E, sizes, gen):
        super().__init__()
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for a, c in zip(sizes[:-1], sizes[1:]):
            bound = 1 / np.sqrt(a)   # same scale as nn.Linear's default init
            self.W.append(nn.Parameter((torch.rand(E, a, c, generator=gen) * 2 - 1) * bound))
            self.b.append(nn.Parameter((torch.rand(E, 1, c, generator=gen) * 2 - 1) * bound))
    def forward(self, x):   # x: [E, rows, in] -> [E, rows, out]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.W) - 1: x = nn.functional.silu(x)
        return x

def train_ensemble(Xtr, ytr, Xva, yva, Xte, E=16, hidden=(256, 128, 64), epochs=500, seed=0, beta=0.05, lr=1e-3, wd=1e-4, bs=128, verbose=True):
    gen = torch.Generator().manual_seed(seed); t0 = time.time()
    ytr, yva = np.asarray(ytr, float).reshape(len(ytr), -1), np.asarray(yva, float).reshape(len(yva), -1)
    mu, sd = ytr.mean(0), ytr.std(0)
    T = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32, device=DEVICE)
    Xt, yt, Xv, yv, Xs = T(Xtr), T((ytr - mu) / sd), T(Xva), T(yva), T(Xte)
    MU, SD = T(mu), T(sd)
    net = BatchedMLP(E, [Xt.shape[1], *hidden, yt.shape[1]], gen).to(DEVICE)
    out = lambda x: MU + SD * net(x)                       # predictions in target units
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = torch.full((E,), float("inf"), device=DEVICE)
    best_state = [p.detach().clone() for p in net.parameters()]
    n = len(Xt); g = torch.Generator(device=DEVICE).manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.argsort(torch.rand(E, n, device=DEVICE, generator=g), dim=1)   # independent shuffle per network
        for i in range(0, n, bs):
            idx = perm[:, i:i + bs]
            pred = MU + SD * net(Xt[idx])
            loss = nn.functional.smooth_l1_loss(pred, MU + SD * yt[idx], beta=beta, reduction="none").mean(dim=(1, 2)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(out(Xv.expand(E, -1, -1)), yv.expand(E, -1, -1), beta=beta, reduction="none").mean(dim=(1, 2))
            better = vl < best; best = torch.where(better, vl, best)
            for p, s in zip(net.parameters(), best_state):
                s[better] = p.detach()[better]
    with torch.no_grad():
        for p, s in zip(net.parameters(), best_state): p.copy_(s)
        pv, pt = out(Xv.expand(E, -1, -1)).cpu().numpy(), out(Xs.expand(E, -1, -1)).cpu().numpy()
    if verbose: print(f"trained {E} networks on {DEVICE} in {time.time()-t0:.0f}s", flush=True)
    return pv, pt
