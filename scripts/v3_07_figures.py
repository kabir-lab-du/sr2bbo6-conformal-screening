"""
V3 pipeline Stage 7 — regenerate manuscript figures from the REPAIRED models.

Writes: figures/v3/Fig*.png and .pdf  (300 dpi, print-oriented, light surface)

WHAT CHANGED vs. reference/manuscript_figures_scripts/generate_main_figures.py
-----------------------------------------------------------------------------
* Fig03 (stability classification) is NOT produced. It is a ROC + confusion matrix
  from the multitask NN, and Stage 3 was deliberately skipped (user decision,
  2026-07-30), so the inputs do not exist. Figure numbering is otherwise preserved so
  the manuscript's Fig01/02/04..10 references stay valid.
* Fig07 / Fig10 no longer use `nlargest(n, "score")`. The discovery score does not
  discriminate among the candidates whose CI spans the PV window (see MANIFEST.md,
  "Stage 4"), so a leaderboard would be a fabricated ordering. They now show the tier
  structure from scripts/v3_05_tier_candidates.py.
* Fig08 reads tables/Table3_dft_vs_ml_v3.csv instead of the hardcoded ML predictions in
  the original script — those were produced by the corrupted model.
* FigE1 (extra, not numbered into the manuscript) plots ML error against electronic
  character, which is the quantitative form of the paper's central limitation claim.

COLOR
-----
Uses the dataviz skill's documented default categorical palette, unmodified:
slot1 #2a78d6, slot2 #eb6834, slot3 #1baf7a, slot4 #eda100. Bar forms use slots 1-4
(adjacent pairlist); scatter/category forms use only slots 1-3, which is the documented
all-pairs-safe subset. No values were substituted, so the palette's published validation
applies. Sequential ramps use a single hue; no rainbow, no dual axes anywhere.
"""
import os, sys, pickle, warnings
sys.path.insert(0, "scripts")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "figures/v3"
os.makedirs(OUT, exist_ok=True)

# ---- design tokens ------------------------------------------------------------------
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK      = "#0b0b0b"   # text-primary
INK_2    = "#52514e"   # text-secondary
INK_MUTE = "#8a8984"   # muted / grid
SURFACE  = "#ffffff"
GOOD, BAD = "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK_MUTE, "axes.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

def finish(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")

def grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

# ---- load ---------------------------------------------------------------------------
print("Loading data and models...")
full   = pd.read_parquet("data/v2/full_A2BBO6.parquet")
test   = pd.read_parquet("data/v2/test_featurized.parquet")
sr2    = pd.read_parquet("data/v2/sr2_holdout_featurized.parquet")
train  = pd.read_parquet("data/v2/train_featurized.parquet")
metrics = pd.read_csv("models/v2/metrics_comparison.csv")
tiers   = pd.read_csv("screening/v2/candidates_tiered.csv")
dft     = pd.read_csv("tables/Table3_dft_vs_ml_v3.csv")

with open("data/v2/imputer.pkl","rb") as f: imputer = pickle.load(f)
with open("data/v2/scaler.pkl","rb") as f: scaler = pickle.load(f)
with open("data/v2/feature_names.pkl","rb") as f: feat_names = pickle.load(f)

magpie = [f for f in feat_names if not f.startswith("src_")]
srcs   = [f for f in feat_names if f.startswith("src_")]
def prep(df):
    return scaler.transform(np.hstack([imputer.transform(df[magpie].values),
                                       df[srcs].values.astype(float)]))
X_test, y_test = prep(test), test["band_gap"].values
X_sr2,  y_sr2  = prep(sr2),  sr2["band_gap"].values

with open("models/v2/stacking.pkl","rb") as f: stacking = pickle.load(f)
with open("models/v2/catboost.pkl","rb") as f: catboost = pickle.load(f)
with open("models/v2/mapie.pkl","rb") as f: mapie = pickle.load(f)

yp_test, yp_sr2 = stacking.predict(X_test), stacking.predict(X_sr2)

# =====================================================================================
# Fig01 — dataset composition
# =====================================================================================
print("Fig01 dataset composition")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
counts = full["source"].value_counts()
ax1.bar(range(len(counts)), counts.values, color=S1, width=0.62, zorder=3)
ax1.set_xticks(range(len(counts))); ax1.set_xticklabels(counts.index, rotation=0)
ax1.set_ylabel("compounds"); ax1.set_title("a  Source database", loc="left")
for i, v in enumerate(counts.values):
    ax1.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7, color=INK_2)
ax1.set_ylim(0, counts.values.max()*1.16); grid(ax1)

ax2.hist(full["band_gap"], bins=45, color=S1, alpha=0.85, zorder=3)
ax2.axvspan(1.2, 1.8, color=S2, alpha=0.16, zorder=2)
ax2.text(1.5, ax2.get_ylim()[1]*0.94, "PV window\n1.2–1.8 eV", ha="center", va="top",
         fontsize=7, color=INK_2)
ax2.set_xlabel("band gap (eV)"); ax2.set_ylabel("compounds")
ax2.set_title(f"b  Band-gap distribution (N = {len(full):,})", loc="left"); grid(ax2)
fig.suptitle("Figure 1. Dataset composition after CMR exclusion", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); finish(fig, "Fig01_dataset_composition")

# =====================================================================================
# Fig02 — parity, stacking ensemble (single series per panel -> no legend)
# =====================================================================================
print("Fig02 parity")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, (yt, yp, lab, sub) in zip(axes, [
        (y_test, yp_test, "a  Held-out test set", "test"),
        (y_sr2,  yp_sr2,  "b  Sr₂ holdout (out-of-domain)", "sr2")]):
    ax.scatter(yt, yp, s=7, alpha=0.42, color=S1, edgecolors="none", zorder=3)
    lim = [0, max(yt.max(), yp.max())*1.05]
    ax.plot(lim, lim, color=INK_MUTE, lw=1.0, ls="--", zorder=4)
    r = metrics.loc[metrics.model == "Stacking"]
    r2  = r["global_r2"].iloc[0] if sub == "test" else r["sr2_r2"].iloc[0]
    mae = r["global_mae"].iloc[0] if sub == "test" else r["sr2_mae"].iloc[0]
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f} eV\nN = {len(yt):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("DFT band gap (eV)"); ax.set_ylabel("predicted band gap (eV)")
    ax.set_title(lab, loc="left"); grid(ax, axis="both")
fig.suptitle("Figure 2. Stacking-ensemble parity", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); finish(fig, "Fig02_parity_stacking")

# =====================================================================================
# Fig04 — model comparison (grouped bars, 4 categorical slots, single axis per panel)
# =====================================================================================
print("Fig04 model comparison")
order = ["CatBoost", "XGBoost", "LightGBM", "Stacking"]
m = metrics.set_index("model").loc[order]
cols = [S1, S2, S3, S4]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, col, ttl, ylab in [(axa, "global_mae", "a  Test MAE", "MAE (eV)"),
                           (axb, "sr2_mae",   "b  Sr₂ holdout MAE", "MAE (eV)")]:
    ax.bar(range(4), m[col].values, color=cols, width=0.62, zorder=3)
    for i, v in enumerate(m[col].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(4)); ax.set_xticklabels(order, rotation=12, ha="right")
    ax.set_ylabel(ylab); ax.set_title(ttl, loc="left")
    ax.set_ylim(0, m[col].max()*1.18); grid(ax)
fig.suptitle("Figure 4. Model comparison (lower is better)", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92]); finish(fig, "Fig04_model_comparison")

# =====================================================================================
# Fig05 — conformal prediction
# =====================================================================================
print("Fig05 conformal (computing intervals)")
_, pis_t = mapie.predict_interval(X_test)
_, pis_s = mapie.predict_interval(X_sr2)
lo_t, hi_t = pis_t[:, 0, 0], pis_t[:, 1, 0]
lo_s, hi_s = pis_s[:, 0, 0], pis_s[:, 1, 0]
cov_t = np.mean((y_test >= lo_t) & (y_test <= hi_t))
cov_s = np.mean((y_sr2  >= lo_s) & (y_sr2  <= hi_s))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))
axa.bar([0, 1], [cov_t, cov_s], color=[S1, S2], width=0.5, zorder=3)
axa.axhline(0.90, color=INK_MUTE, ls="--", lw=1.0, zorder=4)
axa.text(1.46, 0.905, "nominal 0.90", fontsize=7, color=INK_2, va="bottom", ha="right")
for i, v in enumerate([cov_t, cov_s]):
    axa.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
axa.set_xticks([0, 1]); axa.set_xticklabels(["test", "Sr$_2$ holdout"])
axa.set_ylim(0, 1.06); axa.set_ylabel("empirical coverage")
axa.set_title("a  Coverage at 90% nominal", loc="left"); grid(axa)

axb.hist(hi_t - lo_t, bins=40, color=S1, alpha=0.85, label="test", zorder=3)
axb.hist(hi_s - lo_s, bins=40, color=S2, alpha=0.6, label="Sr$_2$ holdout", zorder=3)
axb.axvline(0.6, color=INK_MUTE, ls="--", lw=1.0, zorder=4)
axb.text(0.63, axb.get_ylim()[1]*0.9, "PV window\nwidth 0.6 eV", fontsize=7, color=INK_2)
axb.set_xlabel("90% interval width (eV)"); axb.set_ylabel("compounds")
axb.set_title("b  Interval width", loc="left"); axb.legend(loc="upper right"); grid(axb)
fig.suptitle("Figure 5. Conformal prediction intervals", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92]); finish(fig, "Fig05_conformal_prediction")

# =====================================================================================
# Fig06 — SHAP beeswarm (CatBoost; audit's reliable reference learner)
# =====================================================================================
print("Fig06 SHAP (this takes a minute)")
try:
    import shap
    idx = np.random.RandomState(0).choice(len(X_test), min(600, len(X_test)), replace=False)
    sv = shap.TreeExplainer(catboost).shap_values(X_test[idx])
    fig = plt.figure(figsize=(6.4, 4.4))
    shap.summary_plot(sv, X_test[idx], feature_names=feat_names, show=False,
                      max_display=15, plot_size=None)
    plt.title("Figure 6. SHAP feature attribution (CatBoost)", loc="left",
              fontsize=9.5, weight="bold", pad=10)
    plt.xlabel("SHAP value (eV)")
    finish(plt.gcf(), "Fig06_shap_beeswarm")
except Exception as e:
    print(f"  SKIPPED Fig06: {type(e).__name__}: {e}")

# =====================================================================================
# Fig07 — screening landscape, TIERED (3 categories -> slots 1-3, all-pairs safe)
# =====================================================================================
print("Fig07 screening landscape (tiered)")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
tier_style = [("A1", S1, "A1  window not excluded, point estimate inside"),
              ("A2", S3, "A2  window not excluded, point estimate outside"),
              ("B",  INK_MUTE, "B   PV window excluded at 90%")]
for t, c, lab in tier_style:
    d = tiers[tiers.tier == t]
    ax.scatter(d.bg_pred, d.ci_width, s=17, alpha=0.75, color=c,
               edgecolors="none", label=f"{lab}  (n={len(d)})", zorder=3)
ax.axvspan(1.2, 1.8, color=S2, alpha=0.14, zorder=2)
ax.text(1.5, ax.get_ylim()[1], "PV window", ha="center", va="top", fontsize=7, color=INK_2)
ax.set_xlabel("predicted band gap (eV)")
ax.set_ylabel("90% interval width (eV)")
ax.legend(loc="upper right", handletextpad=0.4)
grid(ax, axis="both")
fig.suptitle(f"Figure 7. Screening landscape, {len(tiers)} candidates "
             f"(tiered — the composite score does not rank within tier A)",
             x=0.02, ha="left", fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92]); finish(fig, "Fig07_screening_landscape")

# =====================================================================================
# Fig08 — DFT validation, from the repaired model
# =====================================================================================
print("Fig08 DFT validation")
d = dft[dft["HSE06"].notna()].copy().sort_values("HSE06")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ypos = np.arange(len(d))
# CI bars, clipped at 0 — a band gap cannot be negative
ci_lo = d["CI_lo_v3"].clip(lower=0).values
ci_hi = d["CI_hi_v3"].values
ax.hlines(ypos, ci_lo, ci_hi, color=INK_MUTE, lw=2.4, alpha=0.5, zorder=3,
          label="90% conformal interval (clipped at 0)")
ax.scatter(d["ML_pred_v3"], ypos, s=42, color=S1, zorder=5, label="ML prediction")
inci = d["HSE06_in_CI"].astype(bool).values
ax.scatter(d["HSE06"][inci], ypos[inci], s=64, marker="D", color=GOOD, zorder=6,
           label="HSE06 (inside interval)")
ax.scatter(d["HSE06"][~inci], ypos[~inci], s=64, marker="X", color=BAD, zorder=6,
           label="HSE06 (outside interval)")
ax.set_yticks(ypos)
ax.set_yticklabels([f"Sr$_2${f[3:-2]}O$_6$" for f in d["formula"]])
ax.axvspan(1.2, 1.8, color=S2, alpha=0.12, zorder=1)
ax.set_xlabel("band gap (eV)"); ax.set_xlim(left=0)
ax.legend(loc="lower right", handletextpad=0.5)
grid(ax, axis="x")
n_cov = int(inci.sum())
fig.suptitle(f"Figure 8. DFT validation with the repaired model "
             f"({n_cov}/{len(d)} inside the 90% interval)",
             x=0.02, ha="left", fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); finish(fig, "Fig08_dft_validation")

# =====================================================================================
# Fig09 — B/B' map of PREDICTED BAND GAP (diverging about the PV window centre)
#
# The original version of this figure counted candidates per B/B' pair. That is
# uninformative here: each composition appears at most ONCE, so every cell is 0 or 1 and
# a sequential ramp implies a magnitude that does not exist (it renders as a binary
# blob). Replaced with the predicted gap itself, which is a real continuous quantity,
# encoded as DIVERGING about the PV window centre (1.5 eV) so the polarity that matters
# — too small / in window / too large — is what the colour shows. Neutral midpoint is
# gray, poles are the palette's cool and warm hues; no hue sits at the midpoint.
# =====================================================================================
print("Fig09 B/B' band-gap map")
import re
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
def bb(f):
    m = re.match(r"Sr2([A-Z][a-z]?)([A-Z][a-z]?)O6", str(f))
    return m.groups() if m else (None, None)
tt = tiers.copy()
tt[["B", "Bp"]] = tt["formula"].apply(lambda f: pd.Series(bb(f)))
tt = tt.dropna(subset=["B", "Bp"])
els = sorted(set(tt.B) | set(tt.Bp))
ix = {e: i for i, e in enumerate(els)}
G = np.full((len(els), len(els)), np.nan)
for _, r in tt.iterrows():
    G[ix[r.B], ix[r.Bp]] = r.bg_pred; G[ix[r.Bp], ix[r.B]] = r.bg_pred

cmap = LinearSegmentedColormap.from_list("pv_div", [S1, "#e9e8e4", S2])
norm = TwoSlopeNorm(vmin=np.nanmin(G), vcenter=1.5, vmax=np.nanmax(G))
fig, ax = plt.subplots(figsize=(6.8, 5.8))
im = ax.imshow(np.ma.masked_invalid(G), cmap=cmap, norm=norm)
ax.set_xticks(range(len(els))); ax.set_xticklabels(els, rotation=90, fontsize=6)
ax.set_yticks(range(len(els))); ax.set_yticklabels(els, fontsize=6)
ax.set_xlabel("B′ site"); ax.set_ylabel("B site")
cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
cb.set_label("predicted band gap (eV)\nmidpoint = PV window centre, 1.5 eV", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax.set_title(f"White = pair not among the {len(tt)} screened candidates", loc="left",
             fontsize=7.5, color=INK_2)
fig.suptitle("Figure 9. Predicted band gap by B/B′ pair", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); finish(fig, "Fig09_bxb_heatmap")

# =====================================================================================
# Fig10 — tier composition (replaces the old top-N leaderboard)
# =====================================================================================
print("Fig10 tier composition")
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                               gridspec_kw={"width_ratios": [1, 1.45]})
tc = [("A1", (tiers.tier == "A1").sum(), S1),
      ("A2", (tiers.tier == "A2").sum(), S3),
      ("B",  (tiers.tier == "B").sum(),  INK_MUTE)]
axa.bar([t[0] for t in tc], [t[1] for t in tc], color=[t[2] for t in tc],
        width=0.6, zorder=3)
for i, (_, n, _) in enumerate(tc):
    axa.text(i, n, f"{n}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
axa.set_ylabel("candidates"); axa.set_title("a  Tier sizes", loc="left")
axa.set_ylim(0, max(t[1] for t in tc)*1.16); grid(axa)

N_SHOWN = 20
a1_all = tiers[tiers.tier == "A1"].sort_values("dist_to_pv_centre")
a1 = a1_all.head(N_SHOWN)
yp2 = np.arange(len(a1))
axb.hlines(yp2, a1.bg_ci_lo.clip(lower=0), a1.bg_ci_hi, color=INK_MUTE, lw=2.2,
           alpha=0.45, zorder=3)
axb.scatter(a1.bg_pred, yp2, s=26, color=S1, zorder=5)
axb.axvspan(1.2, 1.8, color=S2, alpha=0.14, zorder=1)
axb.set_yticks(yp2)
axb.set_yticklabels([f"Sr$_2${f[3:-2]}O$_6$" for f in a1.formula], fontsize=6.5)
axb.invert_yaxis()
axb.set_xlabel("band gap (eV)"); axb.set_xlim(left=0)
axb.set_title(f"b  Tier A1 — first {len(a1)} of {len(a1_all)} shown, ordered by distance\n"
              f"     to window centre. NOT a confidence ranking.", loc="left", fontsize=8)
grid(axb, axis="x")
fig.suptitle("Figure 10. Candidate tiers (no leaderboard — see Methods)",
             x=0.02, ha="left", fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); finish(fig, "Fig10_candidate_tiers")

# =====================================================================================
# FigE1 — EXTRA: ML error vs electronic character (candidate replacement for Fig03)
# =====================================================================================
print("FigE1 error by electronic character")
CLOSED = {"Sr2GaSbO6", "Sr2TeZnO6"}
dd = dft[dft["HSE06"].notna()].copy()
dd["klass"] = np.where(dd.formula.isin(CLOSED), "d$^0$/d$^{10}$\nclosed shell",
                       "open shell\n(d$^1$–d$^5$)")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
xlabels = []
for i, (k, g) in enumerate(dd.groupby("klass")):
    ax.scatter(np.full(len(g), i) + np.linspace(-0.07, 0.07, len(g)),
               g.abs_err_vs_HSE06, s=52, color=[S1, S2][i], zorder=4)
    ax.hlines(g.abs_err_vs_HSE06.mean(), i-0.22, i+0.22, color=INK, lw=1.6, zorder=5)
    ax.text(i+0.26, g.abs_err_vs_HSE06.mean(),
            f"mean {g.abs_err_vs_HSE06.mean():.3f} eV", fontsize=7,
            va="center", color=INK_2)
    # n is carried on the axis label — with n=1 in one class this caveat must be visible
    xlabels.append(f"{k}\nn = {len(g)}")
ax.set_xticks(range(dd.klass.nunique()))
ax.set_xticklabels(xlabels)
ax.set_ylabel("|ML − HSE06|  (eV)"); ax.set_ylim(bottom=0)
ax.set_xlim(-0.5, dd.klass.nunique()-0.1)
grid(ax)
fig.suptitle("Figure E1. ML error by electronic character", x=0.02, ha="left",
             fontsize=9.5, weight="bold")
ax.set_title("EXTRA — not numbered into the manuscript; candidate replacement for Fig03",
             loc="left", fontsize=7, color=INK_2)
fig.tight_layout(rect=[0, 0, 1, 0.92]); finish(fig, "FigE1_error_by_chemistry")

print(f"\nDone. Figures in {OUT}/")
print("NOTE: Fig03 (stability classification) intentionally absent — requires the")
print("      multitask NN, which was not retrained.")
