"""ig specialists. ig is symmetric in (vd, vs) (checked in SPICE: swapped ig matches, p95 0.01%), so inputs are put in canonical
order (higher of vd/vs first) before building the features. Target: 'log' = log10|ig| (sign from a classifier) or
'asinh' = asinh(ig / 1e-16), a signed target that is log-like for large currents and linear through zero.
Same recipe as 03 (Huber, AdamW, cosine LR, best epoch on validation).  Usage: python 07_ig_specialist.py <log|asinh> <seed> [...]"""
import sys, time, copy, importlib
import numpy as np, torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
from common_setup import load, HERE, CURRENT_FLOOR
from sklearn.preprocessing import StandardScaler
ext_features = importlib.import_module("03_mlp_ensemble_feats").ext_features
torch.set_num_threads(3)
S0 = 1e-16
def canon(d):
    d = d.copy(); hi, lo = np.maximum(d.vd, d.vs), np.minimum(d.vd, d.vs); d["vd"], d["vs"] = hi, lo; return d
target = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]]
X, Y, I, (tr, va, te), _ = load("remeasured")
sc = StandardScaler().fit(ext_features(canon(tr))); Xc = {k: sc.transform(ext_features(canon(d))) for k, d in (("tr", tr), ("va", va), ("te", te))}
ig = {k: v[:, 1] for k, v in I.items()}
y = {k: (np.log10(np.maximum(np.abs(v), CURRENT_FLOOR)) if target == "log" else np.arcsinh(v / S0)) for k, v in ig.items()}
mu, sd = y["tr"].mean(), y["tr"].std(); beta = 0.05 if target == "log" else 0.05 * np.log(10)
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
Xt, yt, Xv, yv, Xte = T(Xc["tr"]), T(y["tr"])[:, None], T(Xc["va"]), T(y["va"])[:, None], T(Xc["te"])
for seed in seeds:
    t0 = time.time(); torch.manual_seed(seed)
    core = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
    net = lambda x: mu + sd * core(x)
    opt = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    best, state = 1e9, None
    for ep in range(500):
        core.train()
        for b in torch.randperm(len(Xt)).split(128):
            loss = smooth_l1_loss(net(Xt[b]), yt[b], beta=beta); opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); core.eval()
        with torch.no_grad(): vl = smooth_l1_loss(net(Xv), yv, beta=beta).item()
        if vl < best: best, state = vl, copy.deepcopy(core.state_dict())
    core.load_state_dict(state)
    with torch.no_grad(): pv, pt = net(Xv).numpy().ravel(), net(Xte).numpy().ravel()
    tag = f"ig_canon_{target}_s{seed}"; np.save(HERE / "val_preds" / f"{tag}.npy", pv); np.save(HERE / "test_preds" / f"{tag}.npy", pt)
    if target == "log": e = 100 * np.abs(10 ** pv - np.abs(ig["va"])) / np.abs(ig["va"]); kind = "magnitude"
    else: e = 100 * np.abs(S0 * np.sinh(pv) - ig["va"]) / np.abs(ig["va"]); kind = "signed"
    print(f"{tag} ({time.time()-t0:.0f}s): ig {kind} error median {np.median(e):.2f}%  p95 {np.percentile(e, 95):.1f}%", flush=True)
