"""
Publication figures for the Sr2BB'O6 v4 manuscript.
Sizing: single column 3.4 in, double column 7.0 in (typical two-column journal).
All figures use constrained_layout; every text element is placed to avoid overlap
and each output is visually inspected afterwards.
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = "../v4_results"
OUT = "figures"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "font.family": "DejaVu Sans", "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "savefig.dpi": 300, "figure.constrained_layout.use": True,
})

# Okabe–Ito colorblind-safe palette
BLUE, ORANGE, GREEN, VERM, SKY, YELLOW, PURPLE, GRAY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#F0E442", "#CC79A7", "#666666")

PV_LO, PV_HI = 1.2, 1.8

def sub(f):
    """Sr2MoGaO6 -> Sr$_2$MoGaO$_6$"""
    return re.sub(r"(\d+)", r"$_{\1}$", f)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("wrote", name)

# ---------------------------------------------------------------- fig 1: workflow
fig, ax = plt.subplots(figsize=(7.0, 1.9))
ax.set_xlim(-0.5, 100.5); ax.set_ylim(0, 26); ax.axis("off")

boxes = [
    (0.0,  "DFT source\npool\nMP, OQMD, JARVIS,\nAFLOW, OPTIMADE\n(CMR excluded)"),
    (17.0, "Merge +\nground-state\ndedup ($E_{hull}$),\n6,295 labels"),
    (34.0, "Composition\nfeatures\nMagpie + src,\nVIF-pruned (55)"),
    (51.0, "Stacked GBDT\nCatBoost /\nXGBoost /\nLightGBM $\\to$ Ridge"),
    (68.0, "CV+ conformal\n90% CIs,\nMondrian\nclasses"),
    (85.0, "Screen\nSr$_2$BB'O$_6$ space\n$\\to$ 104 novel\ncandidates"),
]
W, H, Y = 15.0, 22, 2
for x, txt in boxes:
    ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.4",
                                fc="#EAF2F8", ec=BLUE, lw=0.9))
    ax.text(x + W / 2, Y + H / 2, txt, ha="center", va="center", fontsize=6.2)
for i in range(len(boxes) - 1):
    x0 = boxes[i][0] + W + 0.6
    x1 = boxes[i + 1][0] - 0.6
    ax.add_patch(FancyArrowPatch((x0, Y + H / 2), (x1, Y + H / 2),
                                 arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=1.0))
save(fig, "fig1_workflow")

# ---------------------------------------------------------------- fig 2: parity
test = pd.read_csv(f"{DATA}/test_predictions.csv")
sr2 = pd.read_csv(f"{DATA}/sr2_predictions.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
for ax, df, title, mae, r2 in [
    (axes[0], test, "Held-out test set (n = 857)", 0.303, 0.789),
    (axes[1], sr2, "Sr$_2$BB'O$_6$ holdout (n = 582)", 0.390, 0.783),
]:
    cov = (df.y_true >= df.ci_lo) & (df.y_true <= df.ci_hi)
    ax.plot([0, 8.5], [0, 8.5], color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.scatter(df.y_true[cov], df.y_pred[cov], s=9, c=BLUE, alpha=0.45, lw=0,
               zorder=2, label=f"inside 90% CI ({cov.mean():.1%})")
    ax.scatter(df.y_true[~cov], df.y_pred[~cov], s=13, c=VERM, marker="x", lw=0.8,
               zorder=3, label=f"outside 90% CI ({(~cov).mean():.1%})")
    ax.set_title(title)
    ax.set_xlabel("DFT band gap (eV)")
    ax.text(0.03, 0.97, f"MAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(fc="white", ec=GRAY, lw=0.5, alpha=0.85, pad=2.5))
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor=GRAY,
              handletextpad=0.3, borderpad=0.4)
    ax.set_xlim(-0.15, 8.5); ax.set_ylim(-0.15, 8.5)
    ax.set_aspect("equal")
axes[0].set_ylabel("Predicted band gap (eV)")
save(fig, "fig2_parity")

# ---------------------------------------------------------------- fig 3: LOCO
loco = pd.read_csv(f"{DATA}/loco_results.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.bar(loco["fold"], loco["mae"], color=SKY, ec=BLUE, lw=0.6, width=0.7, zorder=2)
mean_mae = loco["mae"].mean()
ax.axhline(mean_mae, color=VERM, lw=1.2, zorder=3, label=f"LOCO mean = {mean_mae:.3f} eV")
ax.axhline(0.3030, color="#333333", lw=1.2, ls="--", zorder=3,
           label="random split = 0.303 eV")
ax.set_xlabel("Leave-one-cluster-out fold (grouped by B-site pair)")
ax.set_ylabel("MAE (eV)")
ax.set_xticks(loco["fold"])
ax.set_ylim(0, 0.47)
ax.legend(loc="upper center", ncols=2, frameon=True, framealpha=0.95,
          edgecolor=GRAY, columnspacing=1.0, handlelength=1.6)
save(fig, "fig3_loco")

# ---------------------------------------------------------------- fig 4: Mondrian widths vs PV window
fig, ax = plt.subplots(figsize=(3.4, 2.7))
cats = ["Closed shell\n($d^0/d^{10}$)", "Open shell", "Global\nquantile"]
widths = [2.227, 1.479, 1.707]
covs = ["cov. 0.926", "cov. 0.916", "cov. 0.912"]
colors = [VERM, BLUE, GRAY]
bars = ax.bar(cats, widths, color=colors, width=0.55, alpha=0.85, zorder=2)
for b, c in zip(bars, covs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, c,
            ha="center", va="bottom", fontsize=7)
ax.axhline(0.6, color=GREEN, lw=1.2, ls="-", zorder=3,
           label="PV window width (0.6 eV)")
ax.set_ylabel("90% interval width (eV)")
ax.set_ylim(0, 2.75)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor=GRAY,
          handlelength=1.6)
save(fig, "fig4_mondrian")

# ---------------------------------------------------------------- fig 5: A1 forest (headline)
nm = pd.read_csv(f"{DATA}/novel_mondrian.csv")
a1 = nm[nm.tier == "A1"].sort_values("bg_pred", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.4, 3.9))
ax.axvspan(PV_LO, PV_HI, color=GREEN, alpha=0.15, zorder=1)
ax.text(PV_LO - 0.08, len(a1) + 0.75, "PV window", ha="right", va="center",
        fontsize=7, color="#00755E")
for i, r in a1.iterrows():
    closed = bool(getattr(r, "reclassified", False))
    col = VERM if closed else BLUE
    ax.plot([max(r.mondrian_lo, 0), r.mondrian_hi], [i, i], color=col, lw=1.5,
            solid_capstyle="butt", zorder=2)
    ax.plot(r.bg_pred, i, "o", color=col, ms=4, zorder=3)
ax.set_yticks(range(len(a1)))
ax.set_yticklabels([sub(f) for f in a1.formula], fontsize=7)
ax.set_xlabel("Band gap (eV) — 90% Mondrian interval")
ax.set_xlim(-0.1, 3.6)
ax.set_ylim(-0.6, len(a1) + 1.4)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=VERM, lw=1.5, marker="o", ms=4,
           label="audit-reclassified (was closed)"),
    Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=4, label="open shell"),
], loc="upper right", frameon=True, framealpha=0.95, edgecolor=GRAY,
   handlelength=1.4, borderpad=0.35, handletextpad=0.4)
save(fig, "fig5_forest")

# ---------------------------------------------------------------- fig 6: src sensitivity
ss = pd.read_csv(f"{DATA}/src_sensitivity.csv")
fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6))
ax = axes[0]
convs = ["all src = 0", "src_OQMD = 1\n(reported)", "src_MP = 1"]
counts = [11, 12, 26]
bars = ax.bar(convs, counts, color=[GRAY, BLUE, ORANGE], width=0.5, alpha=0.85)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(c),
            ha="center", va="bottom", fontsize=7.5)
ax.set_ylabel("Tier-A1 count")
ax.set_ylim(0, 30)
ax = axes[1]
bins = np.linspace(-0.05, 0.75, 33)
ax.hist(ss.d_zeros.abs(), bins=bins, color=GRAY, alpha=0.55,
        label="|$\\Delta$| all-zeros vs OQMD")
ax.hist(ss.d_MP.abs(), bins=bins, histtype="step", color=ORANGE, lw=1.3,
        label="|$\\Delta$| MP vs OQMD")
ax.set_xlabel("Prediction shift under src convention (eV)")
ax.set_ylabel("Candidates")
ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY)
save(fig, "fig6_src")

# ---------------------------------------------------------------- SI: feature importance
fi = pd.read_csv(f"{DATA}/feature_importance.csv").head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(3.4, 3.4))
labels = [f.replace("MagpieData ", "") for f in fi.feature]
ax.barh(labels, fi.mean_norm * 100, color=SKY, ec=BLUE, lw=0.5, height=0.65)
ax.set_xlabel("Mean normalised importance (%)")
ax.set_xlim(0, 12)
ax.tick_params(axis="y", labelsize=6.8)
save(fig, "si_fig1_importance")

# ---------------------------------------------------------------- SI: train label distribution
tl = pd.read_csv(f"{DATA}/train_label_dist.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
srcs = tl.src.value_counts().index.tolist()
stacks = [tl[tl.src == s].band_gap.values for s in srcs]
ax.hist(stacks, bins=np.linspace(0, 9, 46), stacked=True,
        color=[BLUE, ORANGE, GREEN, PURPLE, SKY, GRAY][:len(srcs)],
        label=[f"{s} ({len(v):,})" for s, v in zip(srcs, stacks)])
ax.set_xlabel("Band-gap label (eV)")
ax.set_ylabel("Training entries")
ax.legend(frameon=True, framealpha=0.9, edgecolor=GRAY, fontsize=6.5)
save(fig, "si_fig2_labeldist")

# ---------------------------------------------------------------- SI: known-compound validation
kv = pd.read_csv(f"{DATA}/known_validation.csv")
fig, ax = plt.subplots(figsize=(3.4, 3.3))
inside = kv.db_in_ci.astype(bool)
ax.plot([0, 6], [0, 6], color=GRAY, lw=0.8, ls="--", zorder=1)
ax.scatter(kv.db_gap_v4[inside], kv.bg_pred[inside], s=12, c=BLUE, alpha=0.55, lw=0,
           label=f"inside 90% CI ({inside.mean():.1%})", zorder=2)
ax.scatter(kv.db_gap_v4[~inside], kv.bg_pred[~inside], s=16, c=VERM, marker="x", lw=0.9,
           label=f"outside 90% CI ({(~inside).mean():.1%})", zorder=3)
ax.text(0.03, 0.97, "MAE = 0.355 eV\nn = 192", transform=ax.transAxes, va="top",
        fontsize=7.5, bbox=dict(fc="white", ec=GRAY, lw=0.5, alpha=0.85, pad=2.5))
ax.set_xlabel("Database band gap (eV)")
ax.set_ylabel("Predicted band gap (eV)")
ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor=GRAY)
ax.set_xlim(-0.1, 6); ax.set_ylim(-0.1, 6)
ax.set_aspect("equal")
save(fig, "si_fig3_known_validation")

print("ALL FIGURES DONE")
