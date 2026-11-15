"""NMOS ON - FINAL model.

Data   : Datasets/data/nmos_on.csv, split 70/15/15 seed 42 (repo's common.data.split)
Inputs : the repo's build_features(df, "nmos") (ML models/common/features.py), standardised
Target : log10|I| for each current; sign restored by physics: ig > 0, ib < 0, sign(id) = sign(vd-vs), sign(is) = -sign(vd-vs)
Model  : one MLP per output group - (id, is), ig, ib - 3 networks in total
         256-128-64 SiLU, Huber loss (beta 0.05 decades), AdamW lr 1e-3 wd 1e-4, cosine LR, batch 128, 500 epochs,
         keeps the epoch with the lowest validation loss
Metrics: relative error |pred - SPICE| / |SPICE| -> MAE %, median %, p95 %; R2 on log10|I| and on amps
Output : results/final_metrics.csv, model/nmos_on_final.pt (weights + scaler + target scaling)
Usage  : python nmos_on_final.py
"""
import sys, time, copy, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "ML models"))
from common.data import load_case, split
from common.features import build_features
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

torch.manual_seed(0); np.random.seed(0)
CURRENTS = ["id", "ig", "is", "ib"]
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}
EPOCHS, BS, BETA = 500, 128, 0.05

def make_net(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(),
                         nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, n_out))

def train(Xtr, ytr, Xva, yva):
    mu, sd = ytr.mean(0), ytr.std(0)                       # scale the target, predict in decades
    Xt, yt, Xv, yv = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, (ytr - mu) / sd, Xva, yva))
    MU, SD = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    net = make_net(Xt.shape[1], yt.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, best_state = float("inf"), None
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            loss = nn.functional.smooth_l1_loss(MU + SD * net(Xt[idx]), MU + SD * yt[idx], beta=BETA)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(MU + SD * net(Xv), yv, beta=BETA).item()
        if vl < best: best, best_state = vl, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return net, mu, sd

def sign_rule(d):
    s = np.sign(d.vd - d.vs).replace(0, 1).to_numpy()
    return np.column_stack([s, np.ones(len(d)), -s, -np.ones(len(d))])

df = load_case("nmos_on", REPO / "Datasets" / "data")
tr, va, te = split(df, (0.70, 0.15, 0.15), 42)
SPLITS = {"train": tr, "validation": va, "test": te}
sc = StandardScaler().fit(build_features(tr, "nmos"))
X = {k: sc.transform(build_features(d, "nmos")) for k, d in SPLITS.items()}
Y = {k: np.log10(np.abs(d[CURRENTS].to_numpy())) for k, d in SPLITS.items()}
print(f"{X['train'].shape[1]} input features | {len(tr)}/{len(va)}/{len(te)} rows | 1 network per group")

P = {k: np.zeros((len(d), 4)) for k, d in SPLITS.items()}   # predicted log10|I|
saved = {"feature_mean": sc.mean_, "feature_scale": sc.scale_, "groups": GROUPS, "nets": {}}
for g, cols in GROUPS.items():
    t0 = time.time()
    net, mu, sd = train(X["train"], Y["train"][:, cols], X["validation"], Y["validation"][:, cols])
    with torch.no_grad():
        for k in SPLITS: P[k][:, cols] = mu + sd * net(torch.tensor(X[k], dtype=torch.float32)).numpy()
    saved["nets"][g] = {"state_dict": net.state_dict(), "target_mean": mu, "target_std": sd}
    print(f"  trained group {g:6s} in {time.time()-t0:.0f}s", flush=True)
(HERE / "model").mkdir(exist_ok=True); torch.save(saved, HERE / "model" / "nmos_on_final.pt")

rows = []
for k, d in SPLITS.items():
    true, pred = d[CURRENTS].to_numpy(), sign_rule(d) * 10 ** P[k]
    for j, c in enumerate(CURRENTS):
        e = 100 * np.abs(pred[:, j] - true[:, j]) / np.abs(true[:, j])
        rows.append({"split": k, "current": c, "MAE %": e.mean(), "median %": np.median(e), "p95 %": np.percentile(e, 95),
                     "R2 (log)": r2_score(np.log10(np.abs(true[:, j])), P[k][:, j]), "R2 (I)": r2_score(true[:, j], pred[:, j])})
R = pd.DataFrame(rows); (HERE / "results").mkdir(exist_ok=True); R.to_csv(HERE / "results" / "final_metrics.csv", index=False)
fmt = {"MAE %": "{:.3f}".format, "median %": "{:.3f}".format, "p95 %": "{:.2f}".format, "R2 (log)": "{:.5f}".format, "R2 (I)": "{:.5f}".format}
for k in SPLITS:
    print(f"\n=== {k.upper()} ==="); print(R[R.split == k].drop(columns="split").to_string(index=False, formatters=fmt))
