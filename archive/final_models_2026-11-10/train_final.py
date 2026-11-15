"""FINAL models for the four transistor cases (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) - one recipe for all.

Data   : Datasets/data/<case>.csv (PMOS OFF ig/ib read from BSIM4, see Datasets/netlists/pmos_off_ngspice.cir)
         split 70/15/15 seed 42 (repo's common.data.split)
Inputs : the repo's build_features(df, polarity) (ML models/common/features.py): the 10 varying CSV columns
         + 10 physics features, standardised. (vg and vb are constant or equal to vdd, so they are not raw inputs.)
Target : log10|I| of each current (repo floor 1e-18 A for exact zeros)
Model  : one MLP per output group - (id, is), ig, ib - so 3 networks per case
         256-128-64 SiLU, Huber loss (beta 0.05 decades), AdamW lr 1e-3 wd 1e-4, cosine LR, batch 128, 500 epochs,
         keeps the epoch with the lowest validation loss
Sign   : ON cases - physics rule for id/is: sign(id) = sign(vd-vs), sign(is) = -sign(vd-vs); ig/ib have one fixed sign
         OFF cases - effective drain (higher of vd/vs for NMOS, lower for PMOS) has a fixed sign (+1 NMOS, -1 PMOS);
                     the effective source's sign comes from one gradient-boosting classifier (inputs + KCL margin)
Metrics: repo's relative_error_percent |pred - SPICE| / |SPICE| -> MAE %, median %, p95 %; R2 on log10|I| and on amps
Output : models/<case>.joblib (scaler + 3 networks + sign models), results/<case>_metrics.csv
Usage  : python train_final.py                 # all four cases (--eval: re-score saved models only)
         python train_final.py nmos_off        # one case
Predict: from train_final import load, predict;  currents = predict(load("nmos_on"), df)   # df has the 12 input columns
"""
import sys, time, copy, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from torch import nn
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "ML models"))
from common.config import CASES, CURRENT_FLOOR
from common.data import load_case, split
from common.features import build_features
from common.surrogate import SignModel, log_magnitude
from common.metrics import relative_error_percent
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score

CURRENTS = ["id", "ig", "is", "ib"]
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}
EPOCHS, BS, BETA, HIDDEN = 500, 128, 0.05, (256, 128, 64)

def make_net(n_in, n_out):
    layers, a = [], n_in
    for h in HIDDEN: layers += [nn.Linear(a, h), nn.SiLU()]; a = h
    return nn.Sequential(*layers, nn.Linear(a, n_out))

def train_net(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    mu, sd = ytr.mean(0), ytr.std(0)                           # train on a standardised target, measure loss in decades
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
    return {"state_dict": best_state, "target_mean": mu, "target_std": sd, "n_out": yt.shape[1]}

def load(case):
    return joblib.load(HERE / "models" / f"{case}.joblib")

def predict_log(bundle, df):
    """log10|I| for id, ig, is, ib."""
    X = torch.tensor(bundle["scaler"].transform(build_features(df, bundle["polarity"])), dtype=torch.float32)
    out = np.zeros((len(df), 4))
    for g, cols in GROUPS.items():
        n = bundle["nets"][g]; net = make_net(X.shape[1], n["n_out"]); net.load_state_dict(n["state_dict"]); net.eval()
        with torch.no_grad(): out[:, cols] = n["target_mean"] + n["target_std"] * net(X).numpy()
    return out

def off_parts(bundle, df, log_pred):
    """OFF cases. Effective drain = the terminal the channel current flows into (higher of vd/vs for NMOS, lower for
    PMOS): channel current and leakage add there, so its sign is always +1 (NMOS) / -1 (PMOS) - 100 % of rows.
    Only the effective source can change sign; the classifier gets one extra input, the KCL margin
    log10(|ig|+|ib|) - log10|I_eff_drain| (> 0 means gate+body leakage exceeds the drain current)."""
    s0 = 1.0 if bundle["polarity"] == "nmos" else -1.0
    d_is_drain = ((df.vd > df.vs) if bundle["polarity"] == "nmos" else (df.vd < df.vs)).to_numpy()
    A = 10.0 ** log_pred
    margin = np.log10(A[:, 1] + A[:, 3]) - np.log10(np.where(d_is_drain, A[:, 0], A[:, 2]))
    return s0, d_is_drain, margin

def predict_sign(bundle, df, log_pred=None):
    """+1/-1 per current. ig, ib: one fixed sign per case (SignModel stores it as a constant).
    ON cases, id/is: physics rule - the channel current flows from the higher of vd/vs to the lower,
    so sign(id) = sign(vd-vs), sign(is) = -sign(vd-vs) (99.9 % of rows).
    OFF cases, id/is: effective drain from physics, effective source from one classifier (see off_parts)."""
    X = bundle["scaler"].transform(build_features(df, bundle["polarity"]))
    s = bundle["sign_model"].predict(X)
    if bundle["case"].endswith("_on"):
        vds = np.sign(df["vd"] - df["vs"]).replace(0, 1).to_numpy()
        s[:, 0], s[:, 2] = vds, -vds
    else:
        s0, d_is_drain, margin = off_parts(bundle, df, predict_log(bundle, df) if log_pred is None else log_pred)
        src = bundle["source_sign_clf"].predict(np.c_[X, margin])
        s[:, 0], s[:, 2] = np.where(d_is_drain, s0, src), np.where(d_is_drain, src, s0)
    return s

def predict(bundle, df):
    """Signed currents in amps, columns id, ig, is, ib."""
    L = predict_log(bundle, df)
    return predict_sign(bundle, df, L) * 10 ** L

def splits(case):
    df = load_case(case, REPO / "Datasets" / "data")
    tr, va, te = split(df, (0.70, 0.15, 0.15), 42)
    return {"train": tr, "validation": va, "test": te}

def run(case):
    polarity = CASES[case][0]; t0 = time.time()
    SPLITS = splits(case); tr = SPLITS["train"]
    sc = StandardScaler().fit(build_features(tr, polarity))
    X = {k: sc.transform(build_features(d, polarity)) for k, d in SPLITS.items()}
    Y = {k: log_magnitude(d[CURRENTS].to_numpy()) for k, d in SPLITS.items()}
    signs = np.where(tr[CURRENTS].to_numpy() >= 0, 1.0, -1.0)
    bundle = {"case": case, "polarity": polarity, "scaler": sc, "sign_model": SignModel().fit(X["train"], signs), "nets": {}}
    for i, (g, cols) in enumerate(GROUPS.items()):
        bundle["nets"][g] = train_net(X["train"], Y["train"][:, cols], X["validation"], Y["validation"][:, cols], seed=i)
        print(f"  [{case}] trained group {g:6s} ({time.time()-t0:.0f}s)", flush=True)
    if case.endswith("_off"):   # effective-source sign classifier, fit after the nets because it uses their KCL margin
        _, d_is_drain, margin = off_parts(bundle, tr, predict_log(bundle, tr))
        src_true = np.where(d_is_drain, signs[:, 2], signs[:, 0])
        bundle["source_sign_clf"] = HistGradientBoostingClassifier(random_state=42).fit(np.c_[X["train"], margin], src_true)
    (HERE / "models").mkdir(exist_ok=True); joblib.dump(bundle, HERE / "models" / f"{case}.joblib")
    evaluate(case, bundle, SPLITS)

def evaluate(case, bundle=None, SPLITS=None):
    bundle = bundle or load(case); SPLITS = SPLITS or splits(case)
    rows = []
    for k, d in SPLITS.items():
        true, pred = d[CURRENTS].to_numpy(), predict(bundle, d)
        err = relative_error_percent(true, pred); lt, lp = log_magnitude(true), log_magnitude(pred)
        for j, c in enumerate(CURRENTS):
            rows.append({"case": case, "split": k, "current": c, "MAE %": err[:, j].mean(), "median %": np.median(err[:, j]),
                         "p95 %": np.percentile(err[:, j], 95), "R2 (log)": r2_score(lt[:, j], lp[:, j]),
                         "R2 (I)": r2_score(true[:, j], pred[:, j]),
                         "sign acc %": 100 * (np.sign(pred[:, j]) == np.where(true[:, j] >= 0, 1, -1)).mean()})
    R = pd.DataFrame(rows); (HERE / "results").mkdir(exist_ok=True); R.to_csv(HERE / "results" / f"{case}_metrics.csv", index=False)
    fmt = {"MAE %": "{:.3f}".format, "median %": "{:.3f}".format, "p95 %": "{:.2f}".format,
           "R2 (log)": "{:.5f}".format, "R2 (I)": "{:.5f}".format, "sign acc %": "{:.2f}".format}
    print(f"\n##### {case.upper()} ({' / '.join(str(len(d)) for d in SPLITS.values())} rows)")
    print(R.drop(columns="case").to_string(index=False, formatters=fmt), flush=True)

if __name__ == "__main__":
    args = sys.argv[1:]; only_eval = "--eval" in args          # --eval: re-score the saved models, no training
    for case in ([a for a in args if a != "--eval"] or list(CASES)):
        evaluate(case) if only_eval else run(case)
