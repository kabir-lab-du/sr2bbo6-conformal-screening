"""
V4 pipeline Stage 2B — refit GBDT stacking + MAPIE on the ground-state labels using the
HYPERPARAMETERS ALREADY FOUND by the v3 Optuna studies (optuna_studies/gbdt_v3.db).

No new search: the v3 search landscape was flat (the seeded CatBoost trial was never
beaten in 30 trials; the sibling project's log records 25/20/20 -> 200/200/200 giving
"essentially no improvement"). Carrying the params over is a documented deviation and
must be stated in the manuscript's methods.

Outputs: models/v4/{catboost,xgboost,lgbm,stacking,mapie}.pkl,
         models/v4/metrics_comparison.csv (with the v3 numbers alongside).
"""
import sys, pickle, os, json
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V4_STAGE_2B_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 2B cached — skipping"); sys.exit(0)

os.makedirs("models/v4", exist_ok=True)
# GBDT thread count. NOT the machine's core count: this training matrix is 4,856x55,
# which is far too small to feed 96 OpenMP threads. Measured on this box (bench, 100
# trees, LightGBM at the v3 best params): n_jobs=96 -> 732.85 s, n_jobs=8 -> 0.54 s,
# a 1357x slowdown from barrier spin-wait, made worse by the ~70-110 background load
# other users put on the 96 cores. CatBoost is 1.7x slower at 96 than at 8; XGBoost is
# flat. Raising this is a large pessimisation, not a speedup.
N_CORES = 8

with open("data/v4/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v4/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from mapie.regression import CrossConformalRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

STORAGE = "sqlite:///optuna_studies/gbdt_v3.db"
def best_params(study_name, fallback_json=None):
    try:
        st = optuna.load_study(study_name=study_name, storage=STORAGE)
        p = st.best_params
        print(f"{study_name}: best CV MAE {st.best_value:.4f} eV, params {p}")
        return p
    except Exception as e:
        if fallback_json and os.path.exists(fallback_json):
            with open(fallback_json) as f:
                p = json.load(f)
            print(f"{study_name}: study load failed ({e}); using fallback {fallback_json}")
            return p
        raise

cb_params   = best_params("catboost_v3", "reference/old_hyperparams/catboost_optuna_seed.json")
xgb_params  = best_params("xgboost_v3")
lgbm_params = best_params("lightgbm_v3", "reference/old_hyperparams/lgbm_optuna_seed.json")

train_df = pd.read_parquet("data/v4/train_featurized.parquet")
test_df  = pd.read_parquet("data/v4/test_featurized.parquet")
sr2_df   = pd.read_parquet("data/v4/sr2_holdout_featurized.parquet")

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

X_train = prep(train_df); y_train = train_df["band_gap"].values
X_test  = prep(test_df);  y_test  = test_df["band_gap"].values
X_sr2   = prep(sr2_df);   y_sr2   = sr2_df["band_gap"].values
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Sr2: {X_sr2.shape}")

# Per-model checkpoints: if this stage is interrupted (pause/kill/reboot), a relaunch
# reloads finished fits instead of redoing them. Removed after the stage completes.
def fit_or_load(ckpt, build):
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        print(f"[ckpt] reloaded {ckpt}")
        return m
    m = build()
    with open(ckpt, "wb") as f:
        pickle.dump(m, f)
    print(f"[ckpt] saved {ckpt}")
    return m

print("Fitting CatBoost (fixed params)...")
def _build_cb():
    m = CatBoostRegressor(**cb_params, task_type="CPU", thread_count=N_CORES, verbose=0)
    m.fit(X_train, y_train); return m
best_cb = fit_or_load("models/v4/ckpt_catboost.pkl", _build_cb)

print("Fitting XGBoost (fixed params)...")
def _build_xgb():
    m = XGBRegressor(**xgb_params, tree_method="hist", n_jobs=N_CORES, device="cpu")
    m.fit(X_train, y_train); return m
best_xgb = fit_or_load("models/v4/ckpt_xgboost.pkl", _build_xgb)

print("Fitting LightGBM (fixed params)...")
def _build_lgbm():
    m = LGBMRegressor(**lgbm_params, device="cpu", n_jobs=N_CORES, verbosity=-1)
    m.fit(X_train, y_train); return m
best_lgbm = fit_or_load("models/v4/ckpt_lgbm.pkl", _build_lgbm)

print("Fitting stacking ensemble...")
def _build_stack():
    m = StackingRegressor(
        estimators=[("catboost", best_cb), ("xgboost", best_xgb), ("lgbm", best_lgbm)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=False, n_jobs=1,
    )
    m.fit(X_train, y_train); return m
stacking = fit_or_load("models/v4/ckpt_stacking.pkl", _build_stack)

metrics = []
for name, model in [("CatBoost", best_cb), ("XGBoost", best_xgb),
                     ("LightGBM", best_lgbm), ("Stacking", stacking)]:
    yp_test = model.predict(X_test)
    yp_sr2  = model.predict(X_sr2)
    metrics.append({
        "model":     name,
        "global_r2": r2_score(y_test, yp_test),
        "global_mae": mean_absolute_error(y_test, yp_test),
        "sr2_r2":    r2_score(y_sr2,  yp_sr2),
        "sr2_mae":   mean_absolute_error(y_sr2,  yp_sr2),
    })
    print(f"{name}: Test R²={metrics[-1]['global_r2']:.4f}, MAE={metrics[-1]['global_mae']:.4f} eV"
          f" | Sr2 R²={metrics[-1]['sr2_r2']:.4f}, MAE={metrics[-1]['sr2_mae']:.4f} eV")

mdf = pd.DataFrame(metrics)
mdf.to_csv("models/v4/metrics_comparison.csv", index=False)

if os.path.exists("models/v2/metrics_comparison.csv"):
    v3m = pd.read_csv("models/v2/metrics_comparison.csv")
    print("\nv3 (arbitrary-entry labels) vs v4 (ground-state labels):")
    for _, r in mdf.iterrows():
        o = v3m[v3m["model"] == r["model"]]
        if len(o):
            o = o.iloc[0]
            print(f"  {r['model']:9s} test MAE {o['global_mae']:.4f} -> {r['global_mae']:.4f} | "
                  f"Sr2 MAE {o['sr2_mae']:.4f} -> {r['sr2_mae']:.4f}")

print("\nFitting MAPIE conformal predictor (cross-conformal, method='plus', cv=5)...")
mapie = CrossConformalRegressor(estimator=stacking, method="plus", cv=5, n_jobs=1)
mapie.fit_conformalize(X_train, y_train)
_, y_pis = mapie.predict_interval(X_test)
cov_t = np.mean((y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0]))
w_t   = np.median(y_pis[:, 1, 0] - y_pis[:, 0, 0])
_, y_pis_sr2 = mapie.predict_interval(X_sr2)
cov_s = np.mean((y_sr2 >= y_pis_sr2[:, 0, 0]) & (y_sr2 <= y_pis_sr2[:, 1, 0]))
w_s   = np.median(y_pis_sr2[:, 1, 0] - y_pis_sr2[:, 0, 0])
print(f"MAPIE: test coverage={cov_t:.3f} (width {w_t:.3f} eV), "
      f"Sr2 holdout coverage={cov_s:.3f} (width {w_s:.3f} eV) @90% nominal")

feat_importances = best_xgb.feature_importances_
total_imp = feat_importances.sum()
ranked = sorted(zip(feat_names, feat_importances), key=lambda t: -t[1])
print("\nXGBoost top-15 feature importances (corruption sanity check):")
for name, imp in ranked[:15]:
    frac = imp / total_imp if total_imp > 0 else 0
    flag = "  <-- FLAG: >40% of total, investigate" if frac > 0.40 else ""
    print(f"  {name:30s} {imp:.4f}  ({frac:.1%}){flag}")
src_share = sum(imp for name, imp in zip(feat_names, feat_importances)
                if name.startswith("src_")) / max(total_imp, 1e-12)
print(f"src_* combined importance: {src_share:.2%}")

with open("models/v4/mapie.pkl",    "wb") as f: pickle.dump(mapie, f)
with open("models/v4/stacking.pkl", "wb") as f: pickle.dump(stacking, f)
with open("models/v4/catboost.pkl", "wb") as f: pickle.dump(best_cb, f)
with open("models/v4/xgboost.pkl",  "wb") as f: pickle.dump(best_xgb, f)
with open("models/v4/lgbm.pkl",     "wb") as f: pickle.dump(best_lgbm, f)

for _p in ("models/v4/ckpt_catboost.pkl", "models/v4/ckpt_xgboost.pkl",
           "models/v4/ckpt_lgbm.pkl", "models/v4/ckpt_stacking.pkl"):
    if os.path.exists(_p):
        os.remove(_p)

open(FLAG, "w").close()
print("V4 Stage 2B complete.")
