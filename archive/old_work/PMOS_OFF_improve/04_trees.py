"""Tree models (Extra Trees, Random Forest, CatBoost) on all four log10|I| targets with the extended features.
Saves validation and test predictions for later blending.  Usage: python 04_trees.py"""
import importlib, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common_setup import load, summ, HERE, TARGET_COLUMNS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
ext_features = importlib.import_module("03_mlp_ensemble_feats").ext_features
X, Y, I, (tr, va, te), _ = load("remeasured")
sc = StandardScaler().fit(ext_features(tr)); X = {k: sc.transform(ext_features(d)) for k, d in (("tr", tr), ("va", va), ("te", te))}
MODELS = {"et": lambda: ExtraTreesRegressor(800, max_features=1.0, n_jobs=2, random_state=0),
          "rf": lambda: RandomForestRegressor(500, max_features=0.8, n_jobs=2, random_state=0),
          "cat": lambda: CatBoostRegressor(iterations=6000, learning_rate=0.03, depth=8, verbose=0, random_seed=0, thread_count=2)}
for name, make in MODELS.items():
    t0 = time.time(); pv = np.zeros_like(Y["va"]); pt = np.zeros_like(Y["te"])
    for j in range(4):
        m = make().fit(X["tr"], Y["tr"][:, j]); pv[:, j] = m.predict(X["va"]); pt[:, j] = m.predict(X["te"])
    np.save(HERE / "val_preds" / f"tree_{name}_ext.npy", pv); np.save(HERE / "test_preds" / f"tree_{name}_ext.npy", pt)
    S = summ(pv, I["va"]); print(f"{name} ext ({time.time()-t0:.0f}s): " + "  ".join(f"{t} {S[t][0]:.2f}/{S[t][1]:.1f}" for t in TARGET_COLUMNS), flush=True)
