"""
V4 pipeline Stage 3 — screening with an EXPLICIT source convention, verified novelty,
and tiering restricted to the truly novel candidates.

Fixes deployed here (FABLE_REVIEW.md F1/F3 + §3):
  - Predictions are made under src_OQMD=1 (the majority training convention, 82% of
    labels) as PRIMARY, with all-zeros and src_MP=1 reported as a sensitivity analysis.
  - Novelty is enforced by pymatgen-normalized formula against the full merged database:
    B==B' entries are dropped AND every remaining candidate found in the databases is
    moved to a VALIDATION table (predicted vs known ground-state gap) instead of being
    reported as novel.
  - CI bounds are additionally reported clipped at 0 (a band gap cannot be negative).
  - The Bartel tau range of the final novel set is printed for the manuscript.

Outputs: screening/v4/{novel_tiered.csv, known_validation.csv, src_sensitivity.csv,
         tier_summary.txt}
"""
import os, sys, pickle, re
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V4_STAGE_3_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 3 cached — skipping"); sys.exit(0)

os.makedirs("screening/v4", exist_ok=True)
N_CORES = 96
PV_LO, PV_HI = 1.2, 1.8
PRIMARY = "OQMD"

from preproc_utils import featurize_df, FEAT_LABELS
from utils import norm_formula, compute_bartel_tau
from joblib import Parallel, delayed

with open("data/v4/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v4/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v4/mapie.pkl",       "rb") as f: mapie      = pickle.load(f)

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]
magpie_idx   = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]

screen_df = pd.read_parquet("data/processed/screening_set.parquet")
cand = screen_df[~screen_df["in_prior_screen"]].copy().reset_index(drop=True)

def split_bb(f):
    m = re.match(r"Sr2([A-Z][a-z]?)([A-Z][a-z]?)O6", str(f))
    return m.groups() if m else (None, None)
bb = [split_bb(f) for f in cand["formula"]]
cand["B_site"]  = [t[0] for t in bb]
cand["Bp_site"] = [t[1] for t in bb]
n0 = len(cand)
cand = cand[cand["B_site"] != cand["Bp_site"]].reset_index(drop=True)
print(f"Candidates: {n0} -> {len(cand)} after removing B==B' entries")

print("Featurizing candidates...")
raw = np.array(featurize_df(cand, formula_col="formula", n_jobs=N_CORES))
X_mag = imputer.transform(raw[:, magpie_idx])

def with_src(active):
    src = np.zeros((X_mag.shape[0], len(src_names)))
    if active is not None:
        src[:, src_names.index(f"src_{active}")] = 1.0
    return scaler.transform(np.hstack([X_mag, src]))

res = {}
for tag, active in [("zeros", None), ("OQMD", "OQMD"), ("MP", "MP")]:
    X = with_src(active)
    yp = mapie.predict(X)
    _, pis = mapie.predict_interval(X)
    res[tag] = (yp, pis[:, 0, 0], pis[:, 1, 0])
    print(f"  src={tag:5s}: pred mean {yp.mean():.4f}, min {yp.min():.4f}, max {yp.max():.4f}")

yp, lo, hi = res[PRIMARY]
cand["bg_pred"], cand["bg_ci_lo"], cand["bg_ci_hi"] = yp, lo, hi
cand["bg_ci_lo_clip"] = np.maximum(lo, 0.0)
for tag in ("zeros", "MP"):
    cand[f"bg_pred_{tag}"]  = res[tag][0]
    cand[f"bg_ci_lo_{tag}"] = res[tag][1]
    cand[f"bg_ci_hi_{tag}"] = res[tag][2]

sens = pd.DataFrame({
    "formula": cand["formula"],
    "pred_OQMD": res["OQMD"][0], "pred_zeros": res["zeros"][0], "pred_MP": res["MP"][0],
})
sens["d_zeros"] = (sens.pred_zeros - sens.pred_OQMD)
sens["d_MP"]    = (sens.pred_MP    - sens.pred_OQMD)
sens.to_csv("screening/v4/src_sensitivity.csv", index=False)
print(f"src sensitivity vs {PRIMARY}: mean|Δ| zeros {sens.d_zeros.abs().mean():.4f}, "
      f"MP {sens.d_MP.abs().mean():.4f} eV (max {sens.d_MP.abs().max():.4f})")

# ---- novelty against the merged databases ------------------------------------------
cand["formula_norm"] = [norm_formula(f) for f in cand["formula"]]
full = pd.read_parquet("data/v4/full_A2BBO6.parquet")[["formula_norm", "band_gap", "source"]]
full = full.rename(columns={"band_gap": "db_gap_v4", "source": "db_source"})
cand = cand.merge(full, on="formula_norm", how="left")
cand["known_in_db"] = cand["db_gap_v4"].notna()

known = cand[cand["known_in_db"]].copy()
novel = cand[~cand["known_in_db"]].copy()
print(f"\nNovelty: {len(known)} candidates already in the databases (-> validation set); "
      f"{len(novel)} truly novel")

# ---- tiering (novel set only, primary convention) ----------------------------------
def add_tiers(df):
    df = df.copy()
    df["ci_width"] = df["bg_ci_hi"] - df["bg_ci_lo"]
    overlaps = (df["bg_ci_lo"] <= PV_HI) & (df["bg_ci_hi"] >= PV_LO)
    df["tier"] = np.where(~overlaps, "B",
                  np.where(df["bg_pred"].between(PV_LO, PV_HI), "A1", "A2"))
    df["metal_risk"] = df["bg_pred"] <= 0.3
    df["dist_to_pv_centre"] = (df["bg_pred"] - (PV_LO + PV_HI) / 2).abs()
    return df.sort_values(["tier", "dist_to_pv_centre"]).reset_index(drop=True)

novel = add_tiers(novel)
known = add_tiers(known)

taus = Parallel(n_jobs=32)(delayed(compute_bartel_tau)(f) for f in novel["formula"])
tf = [t for t in taus if t < 900]
print(f"Bartel tau range of the novel set: [{min(tf):.3f}, {max(tf):.3f}]" if tf else
      "Bartel tau: no finite values")

# validation numbers on the known subset
err = (known["bg_pred"] - known["db_gap_v4"]).abs()
in_ci = ((known["db_gap_v4"] >= known["bg_ci_lo"]) &
         (known["db_gap_v4"] <= known["bg_ci_hi"]))
print(f"Known-subset validation (n={len(known)}): MAE {err.mean():.4f} eV, "
      f"median {err.median():.4f}, 90% CI coverage {in_ci.mean():.3f}")

cols = ["formula", "formula_norm", "tier", "bg_pred", "bg_ci_lo", "bg_ci_lo_clip",
        "bg_ci_hi", "ci_width", "dist_to_pv_centre", "metal_risk",
        "bg_pred_zeros", "bg_pred_MP"]
novel[[c for c in cols if c in novel.columns]].to_csv(
    "screening/v4/novel_tiered.csv", index=False)
kcols = cols + ["db_gap_v4", "db_source"]
known["abs_err"] = err
known["db_in_ci"] = in_ci
known[[c for c in kcols if c in known.columns] + ["abs_err", "db_in_ci"]].to_csv(
    "screening/v4/known_validation.csv", index=False)

lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("")
emit(f"Sr2BB'O6 V4 screening — src_{PRIMARY}=1 convention, 90% conformal, PV window "
     f"{PV_LO}-{PV_HI} eV")
emit("=" * 78)
emit(f"Truly novel candidates: {len(novel)}")
for t in ("A1", "A2", "B"):
    emit(f"  Tier {t}: {(novel.tier == t).sum()}")

def tier_counts_variant(tag):
    """Tier counts under convention `tag`, using that variant's own predicted intervals."""
    g = pd.DataFrame({"bg_pred": novel["bg_pred_" + tag],
                      "bg_ci_lo": novel["bg_ci_lo_" + tag],
                      "bg_ci_hi": novel["bg_ci_hi_" + tag]})
    g = add_tiers(g)
    return {t: int((g.tier == t).sum()) for t in ("A1", "A2", "B")}

for tag in ("zeros", "MP"):
    emit(f"  sensitivity, src_{tag}: {tier_counts_variant(tag)}")
emit("")
emit("Tier A1 (point estimate in PV window), ordered by distance to window centre —")
emit("a reading aid, NOT a confidence ranking:")
for _, r in novel[novel.tier == "A1"].iterrows():
    flag = "  [metal risk]" if r.metal_risk else ""
    emit(f"  {r.formula:14s} {r.bg_pred:5.3f} eV  CI [{r.bg_ci_lo_clip:5.3f}, {r.bg_ci_hi:5.3f}]{flag}")
emit("")
emit(f"Known-in-database candidates (validation set, n={len(known)}): "
     f"MAE {err.mean():.4f} eV, coverage {in_ci.mean():.3f}")
emit("These are NOT novel and must not be reported as candidates.")

with open("screening/v4/tier_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

open(FLAG, "w").close()
print("V4 Stage 3 complete.")
