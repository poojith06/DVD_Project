"""Gaussian process regression: exact GP with an ARD kernel (one length-scale per feature) + signal variance + noise.

Per current (id, ig, is, ib) separately:
  1. hyper-parameters: maximise the log marginal likelihood (Adam, 300 steps, float64) on 2,000 random training rows
  2. condition on ALL 7,000 training rows with those hyper-parameters (Cholesky, float64); target standardised
  3. RBF and Matern-5/2 kernels are both fitted; the one with the lower validation MAE (decades) is kept
Runs on the GPU if there is one (float64 is slow on laptop GPUs: about 40-60 min per case on an RTX 3050).
"""
import numpy as np
import torch

from common.config import MODEL_SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KERNELS = ("rbf", "matern52")


class ExactGP:
    def __init__(self, kernel, seed=MODEL_SEED):
        self.kernel, self.seed = kernel, seed

    def _k(self, A, B, ls, sf):
        A, B = A / ls, B / ls
        r2 = ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T).clamp_min(0)
        if self.kernel == "rbf":
            return sf * torch.exp(-0.5 * r2)
        r = torch.sqrt(r2 + 1e-12) * np.sqrt(5.0)  # Matern 5/2
        return sf * (1 + r + r * r / 3) * torch.exp(-r)

    def fit(self, X, y):
        def T(a):
            return torch.tensor(a, dtype=torch.float64, device=DEVICE)

        self.mu, self.sd = y.mean(), y.std()
        yn = (y - self.mu) / self.sd
        idx = np.random.default_rng(self.seed).choice(len(X), min(2000, len(X)), replace=False)
        Xs, ys = T(X[idx]), T(yn[idx])
        p = {n: torch.zeros(s, dtype=torch.float64, device=DEVICE, requires_grad=True)
             for n, s in (("lls", X.shape[1]), ("lsf", 1), ("lsn", 1))}
        with torch.no_grad():
            p["lls"] += np.log(np.sqrt(X.shape[1]))
            p["lsn"] -= 6
        opt = torch.optim.Adam(p.values(), lr=0.05)
        eye = torch.eye(len(Xs), dtype=torch.float64, device=DEVICE)
        for _ in range(300):
            K = self._k(Xs, Xs, p["lls"].exp(), p["lsf"].exp()) + (p["lsn"].exp() + 1e-8) * eye
            Lc = torch.linalg.cholesky(K)
            a = torch.cholesky_solve(ys[:, None], Lc)
            nll = 0.5 * (ys[:, None] * a).sum() + torch.log(torch.diagonal(Lc)).sum()
            opt.zero_grad()
            nll.backward()
            opt.step()
        ls, sf, sn = (p[n].detach().exp() for n in ("lls", "lsf", "lsn"))
        Xt = T(X)
        K = self._k(Xt, Xt, ls, sf) + (sn + 1e-8) * torch.eye(len(X), dtype=torch.float64, device=DEVICE)
        alpha = torch.cholesky_solve(T(yn)[:, None], torch.linalg.cholesky(K))
        # keep on the CPU so the model can be saved and loaded anywhere
        self.ls, self.sf, self.Xt, self.alpha = ls.cpu(), sf.cpu(), Xt.cpu(), alpha.cpu()
        return self

    def predict(self, X):
        ls, sf, Xt, alpha = (t.to(DEVICE) for t in (self.ls, self.sf, self.Xt, self.alpha))
        out = []
        for i in range(0, len(X), 2000):
            Ks = self._k(torch.tensor(X[i:i + 2000], dtype=torch.float64, device=DEVICE), Xt, ls, sf)
            out.append((Ks @ alpha)[:, 0].cpu().numpy())
        return np.concatenate(out) * self.sd + self.mu


class GPRModel:
    name = "gpr"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed = case, seed

    def fit(self, X_train, y_train, X_val, y_val, df_train=None, df_val=None):
        self.gps, self.kernels = [], []
        for j in range(y_train.shape[1]):
            best = (np.inf, None, None)
            for kernel in KERNELS:
                gp = ExactGP(kernel, self.seed).fit(X_train, y_train[:, j])
                mae = float(np.mean(np.abs(gp.predict(X_val) - y_val[:, j])))
                if mae < best[0]:
                    best = (mae, kernel, gp)
            self.gps.append(best[2])
            self.kernels.append(best[1])
        return self

    def predict(self, X, df=None):
        return np.column_stack([gp.predict(X) for gp in self.gps])
