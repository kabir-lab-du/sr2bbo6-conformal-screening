"""
V4 pipeline Stage 4 — Mondrian (class-conditional) conformal prediction.

Motivation (FABLE_REVIEW.md / DFT comparison): errors differ sharply by electronic
character — closed-shell (all B-site cations d0/d10 or main-group) vs open-shell
(partially filled d). A single global conformal quantile therefore over-covers one class
and under-covers the other, and the ~1.7 eV global intervals are useless for screening.
Mondrian conformal calibrates a separate quantile per class, giving valid per-class
coverage — the one credible route to a *usable* screen for the closed-shell subspace.

Method: 5-fold out-of-fold residuals of the v4 stacking on train; per-class conformal
quantile q_c at 90% with the (n+1) finite-sample correction; intervals = pred ± q_c.
Evaluated per class on the test set and the Sr2 holdout. Applied to the novel candidate
set and the 8 DFT-validated compounds.

Outputs: screening/v4/{mondrian_summary.txt, novel_mondrian.csv}
"""
import os, sys, pickle
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V4_STAGE_4_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 4 cached — skipping"); sys.exit(0)

N_CORES = 96
ALPHA = 0.10

from pymatgen.core import Composition, Element
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict
from preproc_utils import featurize_df, FEAT_LABELS

with open("data/v4/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v4/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v4/stacking.pkl",    "rb") as f: stacking   = pickle.load(f)

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]
magpie_idx   = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]

# ---- electronic-character classifier ------------------------------------------------
def classify_formula(formula):
    """'closed' if every cation is d0/d10/main-group in the guessed oxidation states,
    'open' if any has a partially filled d shell, 'unknown' if no oxidation guess."""
    try:
        comp = Composition(formula)
        guesses = comp.oxi_state_guesses(max_sites=-1)
        if not guesses:
            return "unknown"
        oxi = guesses[0]
        for el, ox in oxi.items():
            s = str(el)
            if s == "O":
                continue
            e = Element(s)
            if e.is_lanthanoid or e.is_actinoid:
                if s not in ("La", "Lu"):
                    return "open"
                continue
            if e.is_transition_metal:
                d = e.group - int(round(ox))
                if 0 < d < 10:
                    return "open"
        return "closed"
    except Exception:
        return "unknown"

def classify_many(formulas):
    return Parallel(n_jobs=N_CORES)(delayed(classify_formula)(f) for f in formulas)

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

train_df = pd.read_parquet("data/v4/train_featurized.parquet")
test_df  = pd.read_parquet("data/v4/test_featurized.parquet")
sr2_df   = pd.read_parquet("data/v4/sr2_holdout_featurized.parquet")

X_train = prep(train_df); y_train = train_df["band_gap"].values
X_test  = prep(test_df);  y_test  = test_df["band_gap"].values
X_sr2   = prep(sr2_df);   y_sr2   = sr2_df["band_gap"].values

print("Classifying electronic character (train/test/Sr2)...")
cls_train = np.array(classify_many(train_df["formula_norm"]))
cls_test  = np.array(classify_many(test_df["formula_norm"]))
cls_sr2   = np.array(classify_many(sr2_df["formula_norm"]))
for name, c in [("train", cls_train), ("test", cls_test), ("sr2", cls_sr2)]:
    v = pd.Series(c).value_counts().to_dict()
    print(f"  {name}: {v}")

print("Computing 5-fold OOF predictions of the stacking on train "
      "(5 stacking refits — the slow step)...")
oof = cross_val_predict(stacking, X_train, y_train, cv=5, n_jobs=1)
resid = np.abs(y_train - oof)

def conformal_q(scores):
    n = len(scores)
    if n < 20:
        return None
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(scores, level, method="higher"))

CLASSES = ("closed", "open")
q = {}
for c in CLASSES:
    m = cls_train == c
    # 'unknown' rows fold into 'open' (conservative)
    if c == "open":
        m = m | (cls_train == "unknown")
    q[c] = conformal_q(resid[m])
    print(f"  q_hat[{c}] (n={m.sum()}): {q[c]:.4f} eV  -> interval width {2*q[c]:.3f} eV")
q_global = conformal_q(resid)
print(f"  q_hat[global] (n={len(resid)}): {q_global:.4f} eV (width {2*q_global:.3f})")

lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("")
emit("Mondrian (class-conditional) conformal — v4 stacking, 90% nominal")
emit("=" * 74)

def eval_split(name, X, y, cls):
    yp = stacking.predict(X)
    for c in CLASSES:
        m = cls == c
        if c == "open":
            m = m | (cls == "unknown")
        if m.sum() == 0:
            continue
        cov = np.mean(np.abs(y[m] - yp[m]) <= q[c])
        emit(f"  {name:4s} {c:6s} (n={m.sum():4d}): coverage {cov:.3f} at width {2*q[c]:.3f} eV")

eval_split("test", X_test, y_test, cls_test)
eval_split("sr2",  X_sr2,  y_sr2,  cls_sr2)
emit(f"  (global-quantile width for comparison: {2*q_global:.3f} eV; "
     f"MAPIE cross-conformal width ~ see Stage 2B log)")

# ---- apply to novel candidates ------------------------------------------------------
novel = pd.read_csv("screening/v4/novel_tiered.csv")
cls_nov = np.array(classify_many(novel["formula"]))
novel["e_class"] = np.where(cls_nov == "closed", "closed", "open")
novel["mondrian_lo"] = np.maximum(
    novel["bg_pred"] - [q[c] for c in novel["e_class"]], 0.0)
novel["mondrian_hi"] = novel["bg_pred"] + [q[c] for c in novel["e_class"]]
PV_LO, PV_HI = 1.2, 1.8
overlaps = (novel["mondrian_lo"] <= PV_HI) & (novel["mondrian_hi"] >= PV_LO)
novel["mondrian_tier"] = np.where(~overlaps, "B",
    np.where(novel["bg_pred"].between(PV_LO, PV_HI), "A1", "A2"))
novel.to_csv("screening/v4/novel_mondrian.csv", index=False)

emit("")
emit("Novel candidates under Mondrian intervals:")
for c in ("closed", "open"):
    sub = novel[novel.e_class == c]
    emit(f"  {c:6s} (n={len(sub):3d}): width {2*q[c]:.3f} eV | tiers "
         f"A1={int((sub.mondrian_tier=='A1').sum())} "
         f"A2={int((sub.mondrian_tier=='A2').sum())} "
         f"B={int((sub.mondrian_tier=='B').sum())}")
closed_a1 = novel[(novel.e_class == "closed") & (novel.mondrian_tier == "A1")]
emit("")
emit("Closed-shell tier-A1 novel candidates (the usable-screen shortlist):")
for _, r in closed_a1.sort_values("dist_to_pv_centre").iterrows():
    emit(f"  {r.formula:14s} {r.bg_pred:5.3f} eV  Mondrian CI "
         f"[{r.mondrian_lo:5.3f}, {r.mondrian_hi:5.3f}]")

# ---- DFT-validated compounds --------------------------------------------------------
DFT = ["Sr2GaSbO6", "Sr2CrSbO6", "Sr2MnNbO6", "Sr2TeZnO6",
       "Sr2MnWO6", "Sr2TaReO6", "Sr2VHfO6", "Sr2TiMnO6"]
HSE = {"Sr2GaSbO6": 1.799, "Sr2CrSbO6": 0.580, "Sr2MnNbO6": 0.086,
       "Sr2TaReO6": 0.179, "Sr2VHfO6": 2.360, "Sr2TiMnO6": 2.470}
ddf = pd.DataFrame({"formula": DFT})
raw = np.array(featurize_df(ddf, formula_col="formula", n_jobs=8))
src = np.zeros((len(ddf), len(src_names)))
src[:, src_names.index("src_OQMD")] = 1.0
Xd = scaler.transform(np.hstack([imputer.transform(raw[:, magpie_idx]), src]))
ypd = stacking.predict(Xd)
cls_d = classify_many(DFT)
emit("")
emit("DFT compounds under Mondrian intervals (src_OQMD convention):")
for f, yp_i, c in zip(DFT, ypd, cls_d):
    cc = "closed" if c == "closed" else "open"
    lo_i, hi_i = max(yp_i - q[cc], 0.0), yp_i + q[cc]
    h = HSE.get(f)
    inside = "" if h is None else ("  HSE06 %.3f -> %s" % (h, "IN" if lo_i <= h <= hi_i else "OUT"))
    emit(f"  {f:12s} [{cc:6s}] pred {yp_i:5.3f}  CI [{lo_i:5.3f}, {hi_i:5.3f}]{inside}")

os.makedirs("screening/v4", exist_ok=True)
with open("screening/v4/mondrian_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

open(FLAG, "w").close()
print("V4 Stage 4 complete.")
