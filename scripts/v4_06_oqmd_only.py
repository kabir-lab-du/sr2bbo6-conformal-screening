"""
V4 pipeline Stage 6 — single-convention ablation (Phase 1.1 of the plan).

Question: how much of the model's error is caused by MIXING label conventions across
databases? Train the same architecture on OQMD-labeled rows only (the majority
convention) and evaluate on OQMD-labeled test rows, then compare against the all-data
v4 model evaluated on those SAME rows. If the OQMD-only model wins on its own
convention, convention mixing is costing accuracy and fidelity-conditioning (Phase 1.2)
is justified; if not, the mixing is benign.

Output: models/v4/oqmd_ablation.txt (+ log lines).
"""
import os, sys, pickle
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V4_STAGE_6_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 6 cached — skipping"); sys.exit(0)

# See v4_02b_train_fixed.py: 96 OpenMP threads on this ~5k-row matrix is ~1300x slower
# than 8 for LightGBM. Keep these small.
N_CORES = 8
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

with open("data/v4/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v4/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v4/stacking.pkl",    "rb") as f: full_model = pickle.load(f)

STORAGE = "sqlite:///optuna_studies/gbdt_v3.db"
cbp  = optuna.load_study(study_name="catboost_v3", storage=STORAGE).best_params
xgp  = optuna.load_study(study_name="xgboost_v3",  storage=STORAGE).best_params
lgp  = optuna.load_study(study_name="lightgbm_v3", storage=STORAGE).best_params

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

train_df = pd.read_parquet("data/v4/train_featurized.parquet")
test_df  = pd.read_parquet("data/v4/test_featurized.parquet")

tr_o = train_df[train_df["source"] == "OQMD"].reset_index(drop=True)
te_o = test_df[test_df["source"] == "OQMD"].reset_index(drop=True)
print(f"OQMD-only: train {len(tr_o):,} (of {len(train_df):,}), "
      f"test {len(te_o):,} (of {len(test_df):,})")

X_tr, y_tr = prep(tr_o), tr_o["band_gap"].values
X_te, y_te = prep(te_o), te_o["band_gap"].values

cb = CatBoostRegressor(**cbp, task_type="CPU", thread_count=N_CORES, verbose=0)
xg = XGBRegressor(**xgp, tree_method="hist", n_jobs=N_CORES, device="cpu")
lg = LGBMRegressor(**lgp, device="cpu", n_jobs=N_CORES, verbosity=-1)
model = StackingRegressor(estimators=[("catboost", cb), ("xgboost", xg), ("lgbm", lg)],
                          final_estimator=Ridge(alpha=1.0), passthrough=False, n_jobs=1)
print("Fitting OQMD-only stacking...")
model.fit(X_tr, y_tr)

yp_o    = model.predict(X_te)
yp_full = full_model.predict(X_te)

lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("Single-convention ablation (OQMD-only vs all-data), identical test rows")
emit("=" * 74)
emit(f"OQMD test rows: {len(te_o):,}")
emit(f"  OQMD-only model : MAE {mean_absolute_error(y_te, yp_o):.4f} eV | "
     f"R² {r2_score(y_te, yp_o):.4f}")
emit(f"  All-data model  : MAE {mean_absolute_error(y_te, yp_full):.4f} eV | "
     f"R² {r2_score(y_te, yp_full):.4f}")
d = mean_absolute_error(y_te, yp_full) - mean_absolute_error(y_te, yp_o)
emit(f"  Δ(all-data − OQMD-only) = {d:+.4f} eV "
     f"({'convention mixing costs accuracy — fidelity conditioning justified' if d > 0.01 else 'mixing is benign at this scale'})")

te_x = test_df[test_df["source"] != "OQMD"].reset_index(drop=True)
if len(te_x):
    X_tx, y_tx = prep(te_x), te_x["band_gap"].values
    emit(f"\nTransfer check — non-OQMD test rows (n={len(te_x)}):")
    emit(f"  OQMD-only model : MAE {mean_absolute_error(y_tx, model.predict(X_tx)):.4f} eV")
    emit(f"  All-data model  : MAE {mean_absolute_error(y_tx, full_model.predict(X_tx)):.4f} eV")

with open("models/v4/oqmd_ablation.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

open(FLAG, "w").close()
print("V4 Stage 6 complete.")
