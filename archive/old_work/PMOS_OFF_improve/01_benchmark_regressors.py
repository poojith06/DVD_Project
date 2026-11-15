"""Benchmark many regression model families on PMOS OFF (log10|I| target, one model per terminal), VALIDATION set only.
Error = relative error of the predicted magnitude (sign is handled by the repo's sign classifier, scored separately)."""
import time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common_setup import load, summ, HERE, TARGET_COLUMNS
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

X, Y, I, _, _ = load("remeasured")
MODELS = {
  "Linear (ridge)":            lambda: RidgeCV(alphas=np.logspace(-6, 2, 20)),
  "Polynomial ridge deg 3":    lambda: make_pipeline(PolynomialFeatures(3), StandardScaler(), RidgeCV(alphas=np.logspace(-6, 2, 20))),
  "Polynomial ridge deg 4":    lambda: make_pipeline(PolynomialFeatures(4), StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 15))),
  "Kernel ridge RBF":          lambda: KernelRidge(kernel="rbf", alpha=1e-4, gamma=0.02),
  "Kernel ridge Laplacian":    lambda: KernelRidge(kernel="laplacian", alpha=1e-3, gamma=0.02),
  "SVR RBF":                   lambda: SVR(C=30, epsilon=0.002, gamma="scale"),
  "KNN (k=5, distance)":       lambda: KNeighborsRegressor(5, weights="distance"),
  "Random Forest":             lambda: RandomForestRegressor(500, max_features=0.8, min_samples_leaf=1, n_jobs=-1, random_state=0),
  "Extra Trees":               lambda: ExtraTreesRegressor(800, max_features=1.0, n_jobs=-1, random_state=0),
  "HistGradientBoosting":      lambda: HistGradientBoostingRegressor(max_iter=3000, learning_rate=0.05, max_leaf_nodes=63, early_stopping=False, random_state=0),
  "XGBoost (tuned up)":        lambda: XGBRegressor(n_estimators=4000, learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, n_jobs=-1, random_state=0),
  "LightGBM":                  lambda: LGBMRegressor(n_estimators=4000, learning_rate=0.02, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0),
  "CatBoost":                  lambda: CatBoostRegressor(iterations=4000, learning_rate=0.05, depth=8, verbose=0, random_seed=0, thread_count=-1),
  "MLP (sklearn 256-128-64)":  lambda: MLPRegressor((256, 128, 64), learning_rate_init=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=50, random_state=0),
}
rows = []
for name, make in MODELS.items():
    t0 = time.time(); P = np.zeros_like(Y["va"])
    for j in range(4):
        m = make().fit(X["tr"], Y["tr"][:, j]); P[:, j] = m.predict(X["va"])
    s = summ(P, I["va"]); dt = time.time() - t0
    rows.append({"model": name, "sec": round(dt, 1), **{f"{t}_med": s[t][0] for t in TARGET_COLUMNS}, **{f"{t}_p95": s[t][1] for t in TARGET_COLUMNS}})
    print(f"{name:26s} {dt:6.1f}s  " + "  ".join(f"{t} {s[t][0]:6.2f}/{s[t][1]:7.1f}" for t in TARGET_COLUMNS), flush=True)
    np.save(HERE / "val_preds" / f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}.npy", P) if (HERE / "val_preds").exists() else None
pd.DataFrame(rows).to_csv(HERE / "benchmark_validation.csv", index=False)
