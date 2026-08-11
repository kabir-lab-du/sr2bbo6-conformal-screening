"""
V3 pipeline Stage 6 — recompute ML predictions for the DFT-validated compounds.

Reads:  models/v2/{mapie,stacking}.pkl + data/v2 preprocessors
Writes: tables/Table3_dft_vs_ml_v3.csv, screening/v2/dft_comparison_summary.txt

WHY: the manuscript figure script (generate_main_figures.py, Fig08) hardcodes ML
predictions and 90% CIs for the DFT-validated compounds — and those numbers were produced
by the CORRUPTED XGBoost/stacking model. AUDIT_REPORT.md §E reports only 3/6 of the
HSE06-validated compounds falling inside the CI on that basis. Those predictions must be
regenerated from the repaired model before Fig08 or Table 3 can be trusted.

DFT values below are transcribed from AUDIT_REPORT.md §E (PBE / PBE+U / HSE06 columns),
which is the audited record of the VASP calculations. Nothing here recomputes DFT.
"""
import os, sys, pickle
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

os.makedirs("tables", exist_ok=True)
os.makedirs("screening/v2", exist_ok=True)

# formula, source-PBE, PBE+U, HSE06  (None = not available / pending)
DFT = [
    ("Sr2GaSbO6", 0.729, 1.074, 1.799),
    ("Sr2CrSbO6", 1.689, None,  0.580),
    ("Sr2MnNbO6", 1.202, 0.207, 0.086),
    ("Sr2TeZnO6", None,  0.879, None ),   # HSE06 pending
    ("Sr2MnWO6",  1.899, None,  None ),   # PBE+U metallic; HSE06 pending
    ("Sr2TaReO6", None,  None,  0.179),   # spin-down channel; spin-up 3.15 eV
    ("Sr2VHfO6",  None,  None,  2.360),
    ("Sr2TiMnO6", 0.995, None,  2.470),
]
SPIN_UP_TAREO6 = 3.15  # Sr2TaReO6 is near-half-metallic: reported per spin channel

with open("data/v2/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v2/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v2/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v2/mapie.pkl",       "rb") as f: mapie      = pickle.load(f)

from preproc_utils import featurize_df, FEAT_LABELS

df = pd.DataFrame({"formula": [d[0] for d in DFT]})
raw = np.array(featurize_df(df, formula_col="formula", n_jobs=8))

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]
magpie_idx   = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]

# screening candidates carry no database provenance -> src block is all zeros,
# identical to the convention used in v2_04_screening.py
X = scaler.transform(np.hstack([imputer.transform(raw[:, magpie_idx]),
                                np.zeros((len(df), len(src_names)))]))

y_pred = mapie.predict(X)
_, pis = mapie.predict_interval(X)
lo, hi = pis[:, 0, 0], pis[:, 1, 0]

rows = []
for i, (formula, pbe, pbeu, hse) in enumerate(DFT):
    covered = (hse is not None) and (lo[i] <= hse <= hi[i])
    rows.append({
        "formula": formula,
        "PBE_source": pbe, "PBE_U": pbeu, "HSE06": hse,
        "ML_pred_v3": round(float(y_pred[i]), 4),
        "CI_lo_v3":   round(float(lo[i]), 4),
        "CI_hi_v3":   round(float(hi[i]), 4),
        "CI_width_v3": round(float(hi[i] - lo[i]), 4),
        "abs_err_vs_HSE06": None if hse is None else round(abs(float(y_pred[i]) - hse), 4),
        "HSE06_in_CI": None if hse is None else bool(covered),
    })
out = pd.DataFrame(rows)
out.to_csv("tables/Table3_dft_vs_ml_v3.csv", index=False)

lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("DFT-validated compounds vs REPAIRED ML model (90% conformal intervals)")
emit("=" * 78)
emit(f"{'compound':12s} {'HSE06':>7s} {'ML v3':>7s} {'CI lo':>7s} {'CI hi':>7s} {'|err|':>7s}  in CI?")
for r in rows:
    hse = "  —  " if r["HSE06"] is None else f"{r['HSE06']:5.3f}"
    err = "  —  " if r["abs_err_vs_HSE06"] is None else f"{r['abs_err_vs_HSE06']:5.3f}"
    inci = "—" if r["HSE06_in_CI"] is None else ("YES" if r["HSE06_in_CI"] else "no")
    emit(f"{r['formula']:12s} {hse:>7s} {r['ML_pred_v3']:7.3f} {r['CI_lo_v3']:7.3f} "
         f"{r['CI_hi_v3']:7.3f} {err:>7s}  {inci}")

have = [r for r in rows if r["HSE06"] is not None]
cov  = sum(1 for r in have if r["HSE06_in_CI"])
emit("")
emit(f"Coverage on HSE06-validated compounds: {cov}/{len(have)} = {cov/len(have):.1%}")
emit(f"  (AUDIT_REPORT.md §E reported 3/6 = 50.0% using the CORRUPTED model)")
emit(f"Mean |error| vs HSE06: {np.mean([r['abs_err_vs_HSE06'] for r in have]):.3f} eV")

# split by electronic character, per the audit's d0/d10 vs open-shell distinction
CLOSED = {"Sr2GaSbO6", "Sr2TeZnO6"}
closed = [r for r in have if r["formula"] in CLOSED]
openm  = [r for r in have if r["formula"] not in CLOSED]
if closed:
    emit(f"  d0/d10 (closed-shell):      {sum(1 for r in closed if r['HSE06_in_CI'])}/{len(closed)} in CI, "
         f"mean |err| {np.mean([r['abs_err_vs_HSE06'] for r in closed]):.3f} eV")
if openm:
    emit(f"  open-shell (d1-d5) systems: {sum(1 for r in openm if r['HSE06_in_CI'])}/{len(openm)} in CI, "
         f"mean |err| {np.mean([r['abs_err_vs_HSE06'] for r in openm]):.3f} eV")
emit("")
emit(f"NOTE Sr2TaReO6 is near-half-metallic: HSE06 spin-down = 0.179 eV (used above), "
     f"spin-up = {SPIN_UP_TAREO6} eV.")
emit("     A composition-only model predicts a single gap and cannot represent this.")
emit("NOTE Sr2TeZnO6 and Sr2MnWO6 have no completed HSE06 and are excluded from coverage.")

with open("screening/v2/dft_comparison_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nWrote tables/Table3_dft_vs_ml_v3.csv and screening/v2/dft_comparison_summary.txt")
