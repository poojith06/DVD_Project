"""ig with a physics-based structure (BSIM4 gate-overlap tunnelling: I ~ W*Vox^2*exp(-B*toxe*(a - b*Vox)) per side).
 A 'plainphys': plain MLP on extended features + physics terms for both sides
 B 'struct'   : |ig| = exp(g(drain side)) + exp(g(source side)), same network g for both sides (ig is symmetric in vd/vs);
               each side gets its own Vox physics terms plus the other side's voltage and the shared parameters.
Huber loss on log10|ig|, AdamW, cosine LR, best epoch on validation.  Usage: python 08_ig_physics.py <A|B> <weight_decay> <seed> [...]"""
import sys, time, copy, importlib
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
from common_setup import load, HERE, CURRENT_FLOOR
from sklearn.preprocessing import StandardScaler
ext_features = importlib.import_module("03_mlp_ensemble_feats").ext_features
torch.set_num_threads(10)

def side(d, v_this, v_other):
    vox = (d.vg - v_this).clip(lower=1e-4); vox_o = (d.vg - v_other).clip(lower=1e-4)
    return pd.DataFrame({"log_vox": np.log(vox), "vox": vox, "tox_vox": d.toxe * 1e9 * vox, "tox_vox2": d.toxe * 1e9 * vox ** 2,
                         "log_tox": np.log(d.toxe), "tox": d.toxe * 1e9, "log_w": np.log(d.w), "temp": d.temp, "vdd": d.vdd,
                         "vth0": d.vth0, "ndep": d.ndep / 1e18, "u0": d.u0 * 100, "xj": d.xj * 1e9, "vox_other": vox_o, "log_vox_other": np.log(vox_o)})

mode, wd = sys.argv[1], float(sys.argv[2]); seeds = [int(s) for s in sys.argv[3:]]
X, Y, I, (tr, va, te), _ = load("remeasured")
y = {k: v[:, 1] for k, v in Y.items()}; ig = {k: v[:, 1] for k, v in I.items()}
T = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32)
if mode == "A":
    feat = lambda d: pd.concat([ext_features(d).reset_index(drop=True), side(d, d.vd, d.vs).add_suffix("_d").reset_index(drop=True), side(d, d.vs, d.vd).add_suffix("_s").reset_index(drop=True)], axis=1)
    sc = StandardScaler().fit(feat(tr)); F = {k: T(sc.transform(feat(d))) for k, d in (("tr", tr), ("va", va), ("te", te))}
else:
    sc = StandardScaler().fit(pd.concat([side(tr, tr.vd, tr.vs), side(tr, tr.vs, tr.vd)]))
    F = {k: (T(sc.transform(side(d, d.vd, d.vs))), T(sc.transform(side(d, d.vs, d.vd)))) for k, d in (("tr", tr), ("va", va), ("te", te))}
yt, yv = T(y["tr"])[:, None], T(y["va"])[:, None]; mu, sd = float(y["tr"].mean()), float(y["tr"].std())
for seed in seeds:
    t0 = time.time(); torch.manual_seed(seed)
    nin = (F["tr"].shape[1] if mode == "A" else F["tr"][0].shape[1])
    core = nn.Sequential(nn.Linear(nin, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
    if mode == "A": net = lambda f: mu + sd * core(f)
    else: net = lambda f: torch.logsumexp(torch.cat([core(f[0]), core(f[1])], 1), 1, keepdim=True) * sd / np.log(10) * np.log(10) + mu if False else \
                         (mu + sd * torch.logsumexp(torch.cat([core(f[0]), core(f[1])], 1), 1, keepdim=True))
    batch = lambda f, b: f[b] if mode == "A" else (f[0][b], f[1][b])
    opt = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=wd); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    best, state, best_ep = 1e9, None, 0
    for ep in range(500):
        core.train()
        for b in torch.randperm(len(yt)).split(128):
            loss = smooth_l1_loss(net(batch(F["tr"], b)), yt[b], beta=0.05); opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); core.eval()
        with torch.no_grad(): vl = smooth_l1_loss(net(F["va"]), yv, beta=0.05).item()
        if vl < best: best, state, best_ep = vl, copy.deepcopy(core.state_dict()), ep
    core.load_state_dict(state)
    with torch.no_grad(): pv, pt, ptr = net(F["va"]).numpy().ravel(), net(F["te"]).numpy().ravel(), net(F["tr"]).numpy().ravel()
    tag = f"ig_phys{mode}_wd{wd:g}_s{seed}"; np.save(HERE / "val_preds" / f"{tag}.npy", pv); np.save(HERE / "test_preds" / f"{tag}.npy", pt)
    e = lambda p, i: np.median(100 * np.abs(10 ** p - np.abs(i)) / np.abs(i))
    print(f"{tag} ({time.time()-t0:.0f}s, best epoch {best_ep}): ig magnitude error  train {e(ptr, ig['tr']):.2f}%  validation {e(pv, ig['va']):.2f}%", flush=True)
