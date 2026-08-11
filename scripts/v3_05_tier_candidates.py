"""
V3 pipeline Stage 5 — post-process the screening output into an honest tiered report.

Reads:  screening/v2/new_ranked_candidates.csv  (output of v2_04_screening.py)
        data/v2/full_A2BBO6.parquet             (to re-check novelty)
Writes: screening/v2/candidates_tiered.csv
        screening/v2/tier_summary.txt

WHY THIS STAGE EXISTS
---------------------
Two defects were found in the Stage 4 output on 2026-07-30. Both are inherited from the
original pipeline, not introduced by this rebuild.

(1) The composite `discovery_score` does not discriminate among top candidates.
    The 90% conformal intervals are ~1.70 eV wide while the PV window (1.2-1.8 eV) is
    only 0.6 eV wide. For every candidate whose CI fully spans the window, the overlap
    term is pinned at exactly 0.6 eV, so pv_frac = 0.6 / CI_width -- and CI width varies
    by only sigma = 0.058 eV across the whole candidate set. With Stage 3 (multitask NN)
    deliberately skipped, `p_stable` and `ehull` are literal constants, so they
    contribute nothing. Net effect: the top ~112 candidates' scores span < 2%, i.e. the
    rank order within that group is numerical noise, not signal.

    NOTE this is not merely a consequence of skipping the NN: in the ORIGINAL run,
    `P(stable)` across the reported top-15 spanned only 0.631-0.645 (a 2% range), so the
    score barely discriminated even with the NN present.

    Decision (user, 2026-07-30): stop presenting a rank-ordered leaderboard. Report the
    candidates that cannot be excluded from the PV window as a single tier, and state
    explicitly that the score does not resolve within it.

(2) 17 of the 313 "novel" candidates have B == B', so they are not double perovskites at
    all -- Sr2TiTiO6 is SrTiO3, Sr2TaTaO6 is SrTaO3, etc. All of these were verified to
    be ALREADY PRESENT in the training data: the novelty filter compared Sr2BB'O6-form
    strings without reducing them, so it never noticed. Two of them (Sr2TiTiO6,
    Sr2TaTaO6) had ranked in the top 20. They are removed here.

TIER DEFINITIONS (90% conformal intervals; PV window 1.2-1.8 eV)
---------------------------------------------------------------
  A  "PV window not excluded" -- CI overlaps [1.2, 1.8].
     Because the CIs are ~3x wider than the window, this is a NON-EXCLUSION statement,
     not a confirmation. Within tier A the score does not rank meaningfully.
       A1: point estimate also falls inside the window
       A2: point estimate falls outside, but the window is not excluded
  B  "PV window excluded" -- CI does not overlap the window at 90% confidence.

  Separately flagged: `metal_risk` where the point estimate <= 0.3 eV.
"""
import os, sys, re
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

PV_LO, PV_HI = 1.2, 1.8

os.makedirs("screening/v2", exist_ok=True)
df = pd.read_csv("screening/v2/new_ranked_candidates.csv")
n_start = len(df)

# ---- defect (2): drop B == B' pseudo-double-perovskites -----------------------------
def split_bb(f):
    m = re.match(r"Sr2([A-Z][a-z]?)([A-Z][a-z]?)O6", str(f))
    return m.groups() if m else (None, None)

df[["B", "Bp"]] = df["formula"].apply(lambda f: pd.Series(split_bb(f)))
same = df[df["B"] == df["Bp"]].copy()
df = df[df["B"] != df["Bp"]].copy()
print(f"Removed {len(same)} B==B' entries (not double perovskites; all already in the "
      f"training data): {sorted(same['formula'].tolist())}")
print(f"Candidates: {n_start} -> {len(df)}")

# ---- tiering ------------------------------------------------------------------------
df["ci_width"]   = df["bg_ci_hi"] - df["bg_ci_lo"]
df["ci_overlaps_pv"] = (df["bg_ci_lo"] <= PV_HI) & (df["bg_ci_hi"] >= PV_LO)
df["pred_in_pv"]     = df["bg_pred"].between(PV_LO, PV_HI)
df["ci_spans_pv"]    = (df["bg_ci_lo"] <= PV_LO) & (df["bg_ci_hi"] >= PV_HI)
df["metal_risk"]     = df["bg_pred"] <= 0.3

df["tier"] = np.where(~df["ci_overlaps_pv"], "B",
              np.where(df["pred_in_pv"], "A1", "A2"))

# Within-tier ordering is by |point estimate - window centre|. This is presented as a
# READING AID ONLY -- it is explicitly NOT a confidence-weighted ranking, and must not
# be reported as "top-N candidates by discovery score".
df["dist_to_pv_centre"] = (df["bg_pred"] - (PV_LO + PV_HI) / 2).abs()
df = df.sort_values(["tier", "dist_to_pv_centre"]).reset_index(drop=True)

cols = ["formula", "tier", "bg_pred", "bg_ci_lo", "bg_ci_hi", "ci_width",
        "dist_to_pv_centre", "pred_in_pv", "ci_spans_pv", "metal_risk", "score"]
cols = [c for c in cols if c in df.columns]
df[cols].to_csv("screening/v2/candidates_tiered.csv", index=False)

# ---- summary ------------------------------------------------------------------------
score_col = df["score"]
tierA = df[df["tier"].str.startswith("A")]
lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("Sr2BB'O6 screening — tiered report (90% conformal intervals, PV window 1.2–1.8 eV)")
emit("=" * 78)
emit(f"Candidates after removing {len(same)} B==B' non-double-perovskites: {len(df)}")
emit("")
emit(f"  Tier A  — PV window NOT excluded : {len(tierA):3d}  ({len(tierA)/len(df):.1%})")
emit(f"      A1  — point estimate in window: {(df.tier=='A1').sum():3d}")
emit(f"      A2  — point estimate outside  : {(df.tier=='A2').sum():3d}")
emit(f"  Tier B  — PV window excluded     : {(df.tier=='B').sum():3d}")
emit(f"  flagged metal_risk (pred <= 0.3 eV): {int(df.metal_risk.sum())}")
emit("")
emit(f"CI width: mean {df.ci_width.mean():.3f} eV, std {df.ci_width.std():.3f} eV "
     f"(PV window is only {PV_HI-PV_LO:.1f} eV wide)")
emit(f"{int(df.ci_spans_pv.sum())} candidates have a CI that FULLY spans the PV window "
     f"— for these the overlap term is pinned at {PV_HI-PV_LO:.1f} eV by construction.")
emit("")
emit("Why no leaderboard is reported:")
spanning = df[df["ci_spans_pv"]]
sp_lo, sp_hi = spanning.score.min(), spanning.score.max()
emit(f"  Among the {len(spanning)} candidates whose CI FULLY spans the PV window, the")
emit(f"  overlap term is pinned at {PV_HI-PV_LO:.1f} eV by construction, so the score reduces to")
emit(f"  0.25 x {PV_HI-PV_LO:.1f}/CI_width. Their scores span {sp_lo:.6f}–{sp_hi:.6f}, a relative")
emit(f"  spread of only {100*(sp_hi-sp_lo)/sp_hi:.2f}% — driven purely by CI-width jitter")
emit(f"  (sigma = {df.ci_width.std():.3f} eV). These {len(spanning)} are mutually indistinguishable")
emit( "  and MUST NOT be presented as a ranked top-N.")
emit("")
emit(f"  Across the wider tier A ({len(tierA)} candidates) the score does span "
     f"{tierA.score.min():.6f}–{tierA.score.max():.6f},")
emit( "  but that variation reflects only how much of the window each CI happens to cover.")
emit( "  With Stage 3 skipped, p_stable and ehull are constants, so the score carries NO")
emit( "  stability information whatsoever — it is a repackaging of the point estimate and")
emit( "  its interval, nothing more.")
emit("")
emit("Tier A1 members (point estimate inside the PV window), ordered by distance to")
emit("window centre — a reading aid, NOT a confidence ranking:")
for _, r in df[df.tier == "A1"].iterrows():
    flag = "  [metal risk]" if r.metal_risk else ""
    emit(f"    {r.formula:14s} {r.bg_pred:5.3f} eV  CI [{r.bg_ci_lo:5.3f}, {r.bg_ci_hi:5.3f}]{flag}")

with open("screening/v2/tier_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print()
print("Wrote screening/v2/candidates_tiered.csv and screening/v2/tier_summary.txt")
