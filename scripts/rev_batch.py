#!/usr/bin/env python3
"""Review-response batch (M1 adaptive conformal + containment, M8 per-model preds,
M10 enumeration). Runs in ~/sr2bbo6_stacking_v3 under conda env sr2bbo6.
Thread budget: LGBM n_jobs=8 (hard server constraint), featurization n_jobs=16.
Outputs to rev_out/."""
import os, sys, pickle, itertools, json
import numpy as np, pandas as pd

os.chdir(os.path.expanduser("~/sr2bbo6_stacking_v3"))
sys.path.insert(0, "scripts")
os.makedirs("rev_out", exist_ok=True)
from preproc_utils import featurize_df, FEAT_LABELS
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

N = 8
LO, HI = 1.2, 1.8

with open("data/v4/imputer.pkl", "rb") as f: imputer = pickle.load(f)
with open("data/v4/scaler.pkl", "rb") as f: scaler = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
magpie_names = [c for c in feat_names if not c.startswith("src_")]
src_names = [c for c in feat_names if c.startswith("src_")]
magpie_idx = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

tr = pd.read_parquet("data/v4/train_featurized.parquet")
te = pd.read_parquet("data/v4/test_featurized.parquet")
sr = pd.read_parquet("data/v4/sr2_holdout_featurized.parquet")
Xtr, ytr = prep(tr), tr.band_gap.values
Xte, yte = prep(te), te.band_gap.values
Xsr, ysr = prep(sr), sr.band_gap.values
print("loaded:", len(tr), len(te), len(sr), flush=True)

# ---- A) per-model test predictions (paired-bootstrap fodder) -------------------
out = pd.DataFrame({"formula": te.formula_norm, "y": yte})
for name in ("catboost", "xgboost", "lgbm", "stacking"):
    with open(f"models/v4/{name}.pkl", "rb") as f: m = pickle.load(f)
    out[name] = m.predict(Xte)
out.to_csv("rev_out/per_model_test_preds.csv", index=False)
print("A done", flush=True)

# ---- B) adaptive conformal: split baseline vs CQR vs gap-binned Mondrian -------
itr, ical = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42)
LP = dict(n_estimators=1788, learning_rate=0.0338, num_leaves=73, max_depth=9,
          subsample=0.6301, colsample_bytree=0.7268, reg_alpha=0.8694,
          n_jobs=N, random_state=42, verbose=-1)
point = LGBMRegressor(**LP).fit(Xtr[itr], ytr[itr])
q05 = LGBMRegressor(objective="quantile", alpha=0.05, **LP).fit(Xtr[itr], ytr[itr])
q95 = LGBMRegressor(objective="quantile", alpha=0.95, **LP).fit(Xtr[itr], ytr[itr])
ycal = ytr[ical]
pc, l5c, u5c = (m.predict(Xtr[ical]) for m in (point, q05, q95))
E_abs = np.abs(ycal - pc)
E_cqr = np.maximum(l5c - ycal, ycal - u5c)
ncal = len(ycal)
k90 = int(np.ceil(0.90 * (ncal + 1))) - 1
q_abs = np.sort(E_abs)[min(k90, ncal - 1)]
q_cqr = np.sort(E_cqr)[min(k90, ncal - 1)]

def eval_sets(name, rows):
    res = []
    for sname, X, y in rows:
        p = point.predict(X); lo5, hi5 = q05.predict(X), q95.predict(X)
        cov_a = np.mean((y >= p - q_abs) & (y <= p + q_abs))
        w_a = 2 * q_abs
        lo_c, hi_c = lo5 - q_cqr, hi5 + q_cqr
        cov_c = np.mean((y >= lo_c) & (y <= hi_c))
        w_c = hi_c - lo_c
        # width of CQR intervals for rows predicted near the PV window
        m = (p >= 1.0) & (p <= 2.0)
        res.append(dict(set=sname, n=len(y), cov_abs=cov_a, width_abs=w_a,
                        cov_cqr=cov_c, width_cqr_mean=w_c.mean(),
                        width_cqr_median=np.median(w_c),
                        width_cqr_pvregion=w_c[m].mean() if m.any() else np.nan,
                        n_pvregion=int(m.sum())))
    pd.DataFrame(res).to_csv(f"rev_out/{name}.csv", index=False)
    return res

print(eval_sets("adaptive_eval", [("test", Xte, yte), ("sr2", Xsr, ysr)]), flush=True)

# gap-binned Mondrian on the calibration split (quartiles of point prediction)
bins = np.quantile(pc, [0, .25, .5, .75, 1])
bl = []
for i in range(4):
    m = (pc >= bins[i]) & (pc <= bins[i + 1] if i == 3 else pc < bins[i + 1])
    Eb = np.sort(E_abs[m]); nb = len(Eb)
    qb = Eb[min(int(np.ceil(0.90 * (nb + 1))) - 1, nb - 1)]
    bl.append(dict(bin=f"[{bins[i]:.2f},{bins[i+1]:.2f}]", n=nb, q90_halfwidth=qb, width=2*qb))
pd.DataFrame(bl).to_csv("rev_out/gap_binned_mondrian.csv", index=False)
print("B done", flush=True)

# ---- C) candidates + enumeration: predictions, intervals, containment ----------
cand = pd.read_csv("rev_out/cand_formulas.csv")     # 296 candidate formulas (uploaded)
ELS = ['Al','Bi','Co','Cr','Cu','Fe','Ga','Ge','Hf','In','Ir','Mn','Mo','Nb','Ni',
       'Pd','Re','Rh','Ru','Sb','Sc','Sn','Ta','Te','Ti','V','W','Y','Zn','Zr']
enum = pd.DataFrame({"formula": [f"Sr2{a}{b}O6" for a, b in itertools.combinations(ELS, 2)]})
enum["in_296"] = enum.formula.isin(set(cand.formula))
def featurize_block(df):
    raw = np.array(featurize_df(df, formula_col="formula", n_jobs=16))
    X_mag = imputer.transform(raw[:, magpie_idx])
    X_src = np.zeros((len(df), len(src_names)))
    X_src[:, src_names.index("src_OQMD")] = 1.0
    return scaler.transform(np.hstack([X_mag, X_src]))

for label, df in (("cand", cand), ("enum", enum)):
    X = featurize_block(df)
    p = point.predict(X); lo5, hi5 = q05.predict(X), q95.predict(X)
    d = df.copy()
    d["pred"] = p
    d["abs_lo"], d["abs_hi"] = p - q_abs, p + q_abs
    d["cqr_lo"], d["cqr_hi"] = lo5 - q_cqr, hi5 + q_cqr
    # containment confidence: max level a with finite-sample-valid interval inside window
    r_abs = np.minimum(p - LO, HI - p)
    d["contain_abs"] = np.where(r_abs > 0,
        (np.searchsorted(np.sort(E_abs), r_abs, side="right")) / (ncal + 1), 0.0)
    r_cqr = np.minimum(lo5 - LO, HI - hi5)
    d["contain_cqr"] = np.where(r_cqr > 0,
        (np.searchsorted(np.sort(E_cqr), r_cqr, side="right")) / (ncal + 1), 0.0)
    d.to_csv(f"rev_out/{label}_adaptive.csv", index=False)
    print(label, "max contain_abs=%.3f max contain_cqr=%.3f" %
          (d.contain_abs.max(), d.contain_cqr.max()), flush=True)

json.dump(dict(q_abs=float(q_abs), q_cqr=float(q_cqr), ncal=ncal,
               point_test_mae=float(np.abs(yte - point.predict(Xte)).mean())),
          open("rev_out/summary.json", "w"), indent=1)
print("ALL DONE", flush=True)
