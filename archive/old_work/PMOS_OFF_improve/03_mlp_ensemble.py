"""Repo-recipe MLP (Huber in decades, AdamW, cosine LR, keep the best epoch on validation), trained with several seeds so the
networks can be averaged. Saves each network's validation AND test predictions (test is only looked at in the final step).
Usage: python 03_mlp_ensemble.py <feature_set: base|ext> <hidden e.g. 256-128-64> <epochs> <seed> [<seed> ...] [--out 0,2]
--out picks which currents the network predicts (0=id, 1=ig, 2=is, 3=ib); default all four."""
import sys, time, copy
import numpy as np, torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
from common_setup import load, summ, HERE, TARGET_COLUMNS
from common.features import build_features, BOLTZMANN_OVER_Q
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(3)

def ext_features(d):
    """repo features + per-terminal biases: gate-drain and gate-source voltage, body-drain/body-source voltage, each also in units of Vt"""
    f = build_features(d, "pmos").copy(); vt = BOLTZMANN_OVER_Q * (d["temp"] + 273.15)
    f["vgd"] = d.vg - d.vd; f["vgs"] = d.vg - d.vs; f["vgd_vt"] = f.vgd / vt; f["vgs_vt"] = f.vgs / vt
    f["vmin"] = np.minimum(d.vd, d.vs); f["vds_vt"] = (d.vd - d.vs) / vt; f["sign_vds"] = np.sign(d.vd - d.vs)
    return f

args = sys.argv[1:]; OUT = [0, 1, 2, 3]
if "--out" in args: k = args.index("--out"); OUT = [int(v) for v in args[k + 1].split(",")]; args = args[:k] + args[k + 2:]
fs, hidden, epochs = args[0], tuple(int(h) for h in args[1].split("-")), int(args[2]); seeds = [int(s) for s in args[3:]]
X, Y, I, (tr, va, te), _ = load("remeasured")
if fs == "ext":
    sc = StandardScaler().fit(ext_features(tr)); X = {k: sc.transform(ext_features(d)) for k, d in (("tr", tr), ("va", va), ("te", te))}
Y = {k: v[:, OUT] for k, v in Y.items()}
mu, sd = Y["tr"].mean(0), Y["tr"].std(0)
T = lambda a: torch.tensor(a, dtype=torch.float32)
Xt, yt, Xv, yv, Xte = T(X["tr"]), T(Y["tr"]), T(X["va"]), T(Y["va"]), T(X["te"])
for seed in seeds:
    t0 = time.time(); torch.manual_seed(seed)
    layers, n = [], Xt.shape[1]
    for h in hidden: layers += [nn.Linear(n, h), nn.SiLU()]; n = h
    core = nn.Sequential(*layers, nn.Linear(n, len(OUT))); MU, SD = T(mu), T(sd)
    net = lambda x: MU + SD * core(x)
    opt = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = 1e9, None
    for ep in range(epochs):
        core.train()
        for b in torch.randperm(len(Xt)).split(128):
            loss = smooth_l1_loss(net(Xt[b]), yt[b], beta=0.05); opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); core.eval()
        with torch.no_grad(): vl = smooth_l1_loss(net(Xv), yv, beta=0.05).item()
        if vl < best: best, state = vl, copy.deepcopy(core.state_dict())
    core.load_state_dict(state)
    with torch.no_grad(): pv, pt = net(Xv).numpy(), net(Xte).numpy()
    tag = f"ens_{fs}_{args[1]}_e{epochs}_s{seed}" + ("" if OUT == [0, 1, 2, 3] else "_out" + "".join(map(str, OUT)))
    np.save(HERE / "val_preds" / f"{tag}.npy", pv); np.save(HERE / "test_preds" / f"{tag}.npy", pt)
    full = np.full((len(pv), 4), np.nan); full[:, OUT] = pv; S = summ(np.nan_to_num(full, nan=-30), I["va"])
    print(f"{tag} ({time.time()-t0:.0f}s): " + "  ".join(f"{TARGET_COLUMNS[j]} {S[TARGET_COLUMNS[j]][0]:.2f}/{S[TARGET_COLUMNS[j]][1]:.1f}" for j in OUT), flush=True)
