"""
V4 pipeline Stage 5 — grouped (LOCO-style) cross-validation.

Random row splits let near-identical chemistry sit on both sides of the split, so the
0.29 eV random-split MAE is an optimistic estimate for genuinely new B-site chemistry
(Meredig et al. 2018; Li et al. 2023). This stage reports 10-fold GroupKFold where the
group is the B-site element pair (fallback: the full cation set when the A/B assignment
is ambiguous), so no B-pair ever appears in both train and test folds.

Hyperparameters fixed (same as Stage 2B). ~10 stacking refits — runs for hours; per-fold
progress is printed for the monitor.

Output: models/v4/loco_results.csv + summary lines in the log.
"""
import os, sys, pickle, time
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V4_STAGE_5_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 5 cached — skipping"); sys.exit(0)

from pymatgen.core import Composition
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

with open("data/v4/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v4/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v4/stacking.pkl",    "rb") as f: stacking   = pickle.load(f)

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

pool = pd.concat([pd.read_parquet("data/v4/train_featurized.parquet"),
                  pd.read_parquet("data/v4/test_featurized.parquet")],
                 ignore_index=True)
X = prep(pool); y = pool["band_gap"].values
print(f"Grouped CV pool (non-Sr2): {len(pool):,} rows")

def group_key(formula_norm):
    try:
        comp = Composition(formula_norm).reduced_composition
        counts = {str(el): int(round(v)) for el, v in comp.items() if str(el) != "O"}
        ones = sorted([el for el, c in counts.items() if c == 1])
        if len(ones) == 2:          # A2 B B' O6 -> group by the B pair
            return "|".join(ones)
        return "|".join(sorted(counts))  # fallback: full cation set
    except Exception:
        return str(formula_norm)

groups = pool["formula_norm"].map(group_key).values
print(f"Distinct groups: {pd.Series(groups).nunique():,}")

gkf = GroupKFold(n_splits=10)
oof = np.full(len(y), np.nan)
fold_rows = []
for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
    t0 = time.time()
    model = clone(stacking)
    model.fit(X[tr], y[tr])
    yp = model.predict(X[te])
    oof[te] = yp
    mae = mean_absolute_error(y[te], yp)
    fold_rows.append({"fold": k, "n_test": len(te), "mae": mae,
                      "minutes": (time.time() - t0) / 60})
    print(f"[fold {k+1}/10] n_test={len(te):4d}  MAE={mae:.4f} eV  "
          f"({fold_rows[-1]['minutes']:.1f} min)", flush=True)

mae_all = mean_absolute_error(y, oof)
r2_all  = r2_score(y, oof)
base    = mean_absolute_error(y, np.full_like(y, y.mean()))
fdf = pd.DataFrame(fold_rows)
fdf.to_csv("models/v4/loco_results.csv", index=False)

print("\nGrouped-CV (leave-B-pair-out) result:")
print(f"  pooled MAE {mae_all:.4f} eV | R² {r2_all:.4f} | "
      f"per-fold MAE {fdf.mae.min():.3f}-{fdf.mae.max():.3f}")
print(f"  predict-the-mean baseline: {base:.4f} eV")
print("  compare: random-split test MAE (Stage 2B log) and Sr2 holdout MAE — the grouped")
print("  number is the honest estimate for genuinely new B-site chemistry.")

open(FLAG, "w").close()
print("V4 Stage 5 complete.")
