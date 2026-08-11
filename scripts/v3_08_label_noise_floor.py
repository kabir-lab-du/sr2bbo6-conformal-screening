"""
V3 pipeline Stage 8 — empirical label-noise floor from cross-database disagreement.

Reads:  data/v2_sources/*.parquet  (the raw per-database pulls, pre-deduplication)
Writes: tables/label_noise_floor.csv, tables/label_noise_pairs.csv,
        figures/v3/FigE2_label_noise.png/.pdf, screening/v2/label_noise_summary.txt

QUESTION
--------
Stage 1 deduplicates by `formula_norm`, keeping one row per composition by source
priority. Before that step, the SAME composition is often reported by several databases
with DIFFERENT band gaps. That disagreement is a direct, model-free estimate of the
irreducible error in the labels: no regressor can do better on this dataset than the data
disagrees with itself.

If the cross-database spread is comparable to the stacking ensemble's test MAE
(0.293 eV), then the model is already at the noise floor and no change of architecture —
ALIGNN, E3NN, anything — can improve it on these labels. If the spread is much smaller,
there is genuine headroom and a better model is worth building.

METHOD
------
Reproduce Stage 1's pipeline exactly up to (but NOT including) the dedup step, so the
comparison set is the same rows the model was trained from. Then, for every composition
reported by 2+ distinct sources, measure the disagreement.

Two estimators are reported:
  * pairwise |Δgap| between database pairs — the direct quantity, comparable to MAE
  * per-composition spread (max-min, std) — how badly a single formula is pinned down

CAVEAT recorded in the output: databases differ in functional AND in relaxation settings,
so this measures total inter-database inconsistency, not pure random noise. It is an
upper bound on what a model could be expected to reconcile, and a lower bound on
achievable MAE only insofar as the model must reproduce ONE of the disagreeing values.
"""
import os, sys, itertools
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np
from pymatgen.core import Composition
from utils import compute_bartel_tau, norm_formula

os.makedirs("tables", exist_ok=True)
os.makedirs("figures/v3", exist_ok=True)
os.makedirs("screening/v2", exist_ok=True)

SOURCES = {
    "mp_all":    "data/v2_sources/mp_all.parquet",
    "oqmd_all":  "data/v2_sources/oqmd_all.parquet",
    "jarvis_all":"data/v2_sources/jarvis_all.parquet",
    "cmr_all":   "data/v2_sources/cmr_all.parquet",
    "aflow_all": "data/v2_sources/aflow_all.parquet",
    "optimade":  "data/v2_sources/optimade.parquet",
}

dfs = []
for name, path in SOURCES.items():
    if not os.path.exists(path):
        continue
    d = pd.read_parquet(path)
    if len(d) == 0:
        continue
    d["source"] = d["source"].str.replace(r"^CMR-.*", "CMR", regex=True)
    dfs.append(d)
raw = pd.concat(dfs, ignore_index=True)
print(f"Raw combined: {len(raw):,}")

# --- reproduce Stage 1 filters up to, but not including, dedup ------------------------
def is_A2BBO6(formula):
    try:
        comp = Composition(formula).reduced_composition
        if not (3 <= len(list(comp.elements)) <= 5):
            return False
        o = comp["O"]
        cat = sum(v for k, v in comp.items() if str(k) != "O")
        return abs(cat / o - (4/6)) < 0.05
    except Exception:
        return False

raw = raw[raw["formula_pretty"].apply(is_A2BBO6)]
raw = raw[raw["is_metal"].fillna(False) == False]
raw = raw[raw["band_gap"].fillna(0) >= 0.05]
raw = raw[raw["formula_pretty"].apply(compute_bartel_tau) <= 6.0]
raw = raw[raw["band_gap"].notna()].copy()
raw["formula_norm"] = raw["formula_pretty"].apply(norm_formula)
print(f"After Stage-1 filters (pre-dedup): {len(raw):,}")

# one value per (composition, source): if a source lists a formula more than once,
# collapse to its median so intra-source duplication cannot masquerade as disagreement
per_src = (raw.groupby(["formula_norm", "source"])["band_gap"]
              .median().reset_index())

grp = per_src.groupby("formula_norm")
multi = grp.filter(lambda g: g["source"].nunique() >= 2)
n_multi = multi["formula_norm"].nunique()
print(f"Compositions reported by 2+ databases: {n_multi:,}")

if n_multi == 0:
    print("No cross-database duplicates found — cannot estimate a noise floor this way.")
    sys.exit(0)

# --- pairwise disagreements -----------------------------------------------------------
pairs = []
for f, g in multi.groupby("formula_norm"):
    recs = list(zip(g["source"], g["band_gap"]))
    for (s1, v1), (s2, v2) in itertools.combinations(recs, 2):
        a, b = sorted([s1, s2])
        va, vb = (v1, v2) if a == s1 else (v2, v1)
        pairs.append({"formula_norm": f, "src_a": a, "src_b": b,
                      "gap_a": va, "gap_b": vb, "abs_diff": abs(v1 - v2)})
P = pd.DataFrame(pairs)
P.to_csv("tables/label_noise_pairs.csv", index=False)

# --- per-composition spread -----------------------------------------------------------
spread = (multi.groupby("formula_norm")["band_gap"]
               .agg(n_sources="count", gap_min="min", gap_max="max",
                    gap_mean="mean", gap_std="std")
               .reset_index())
spread["range"] = spread["gap_max"] - spread["gap_min"]
spread.to_csv("tables/label_noise_floor.csv", index=False)

STACK_MAE = 0.2930   # models/v2/metrics_comparison.csv, Stacking global_mae

lines = []
def emit(s=""):
    lines.append(s); print(s)

emit("Empirical label-noise floor from cross-database disagreement")
emit("=" * 74)
emit(f"Compositions reported by 2+ databases : {n_multi:,}")
emit(f"Database-pair comparisons             : {len(P):,}")
emit("")
emit("PAIRWISE |Δ band gap| between databases  (directly comparable to model MAE)")
emit(f"  mean   {P.abs_diff.mean():.4f} eV      <-- the quantity to compare against MAE")
emit(f"  median {P.abs_diff.median():.4f} eV")
emit(f"  std    {P.abs_diff.std():.4f} eV")
for q in (0.25, 0.75, 0.90):
    emit(f"  p{int(q*100):<3d}   {P.abs_diff.quantile(q):.4f} eV")
emit(f"  max    {P.abs_diff.max():.4f} eV")
emit(f"  fraction of pairs disagreeing by >0.5 eV: {(P.abs_diff > 0.5).mean():.1%}")
emit(f"  fraction of pairs disagreeing by >1.0 eV: {(P.abs_diff > 1.0).mean():.1%}")
emit("")
emit("PER-COMPOSITION spread")
emit(f"  mean range (max-min) {spread['range'].mean():.4f} eV")
emit(f"  median range         {spread['range'].median():.4f} eV")
emit("")
emit("DISAGREEMENT BY DATABASE PAIR (mean |Δ|, n comparisons)")
bysrc = (P.groupby(["src_a", "src_b"])["abs_diff"]
           .agg(["mean", "median", "count"])
           .sort_values("count", ascending=False))
for (a, b), r in bysrc.iterrows():
    emit(f"  {a:9s} vs {b:9s}  mean {r['mean']:.4f}  median {r['median']:.4f}  (n={int(r['count'])})")
emit("")
emit("VERDICT")
emit("-------")
# The all-pairs mean is dominated by CMR (GLLB-SC), which is EXCLUDED from the training
# data this model was built on. The number relevant to the deployed model is therefore
# the CMR-free subset; both are reported so the CMR effect is visible rather than buried.
Pn = P[(P.src_a != "CMR") & (P.src_b != "CMR")]
Pc = P[(P.src_a == "CMR") | (P.src_b == "CMR")]
emit(f"  Stacking ensemble test MAE                 : {STACK_MAE:.4f} eV")
emit(f"  Mean disagreement, ALL pairs               : {P.abs_diff.mean():.4f} eV  (n={len(P):,})")
emit(f"  Mean disagreement, CMR pairs only          : {Pc.abs_diff.mean():.4f} eV  (n={len(Pc):,})")
emit(f"  Mean disagreement, EXCLUDING CMR           : {Pn.abs_diff.mean():.4f} eV  (n={len(Pn):,})  <-- relevant to this model")
emit(f"  Median disagreement, EXCLUDING CMR         : {Pn.abs_diff.median():.4f} eV")
emit("")
emit(f"  The CMR (GLLB-SC) rows disagree with PBE-level databases by "
     f"{Pc.abs_diff.mean():.2f} eV on average —")
emit(f"  {Pc.abs_diff.mean()/max(Pn.abs_diff.mean(),1e-9):.1f}x the disagreement among the PBE-level sources. This is direct,")
emit("  model-free evidence that excluding CMR was correct, independent of the SHAP result.")
emit("")
ratio = STACK_MAE / Pn.abs_diff.mean()
emit(f"  model MAE / label disagreement (excl. CMR) : {ratio:.2f}")
if ratio <= 1.15:
    emit("  => The model's error is AT OR BELOW the level at which the databases")
    emit("     disagree with each other. The labels, not the architecture, are the")
    emit("     binding constraint. A better model (ALIGNN/E3NN/anything) cannot be")
    emit("     expected to improve MAE materially on THESE labels.")
elif ratio <= 1.6:
    emit("  => The model's error is close to the label disagreement. There is limited")
    emit("     headroom; expect modest gains at best from a better architecture.")
else:
    emit("  => The model's error is well ABOVE the label disagreement. There is genuine")
    emit("     headroom — a better model is worth building on these labels.")
emit("")
emit("CAVEATS")
emit("-------")
emit("1. Databases differ in exchange-correlation functional AND relaxation settings, so")
emit("   this measures total inter-database inconsistency, not pure random noise.")
emit("2. The model is trained and tested on the DEDUPLICATED set, which keeps one value")
emit("   per composition by source priority. Its MAE is measured against that retained")
emit("   convention, not against 'the' true gap. The correct reading: the retained label")
emit("   is itself uncertain by roughly the disagreement above, so pushing model error")
emit("   below that level is fitting a convention, not the physics.")
emit("3. Compositions reported by 2+ databases may not represent the whole set — widely")
emit("   computed materials tend to be better characterised, which if anything makes")
emit("   this an UNDER-estimate of disagreement across the full composition space.")

with open("screening/v2/label_noise_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# --- figure ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S1, S2, INK, INK_2, INK_MUTE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK_MUTE, "axes.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK,
    "text.color": INK, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax1.hist(P.abs_diff, bins=50, color=S1, zorder=3)
ax1.axvline(P.abs_diff.mean(), color=INK, lw=1.4, zorder=5)
ax1.axvline(STACK_MAE, color=S2, lw=1.4, ls="--", zorder=5)
ax1.text(P.abs_diff.mean(), ax1.get_ylim()[1]*0.94,
         f"  mean database\n  disagreement\n  {P.abs_diff.mean():.3f} eV",
         fontsize=7, va="top", color=INK)
ax1.text(STACK_MAE, ax1.get_ylim()[1]*0.45, f"  model MAE\n  {STACK_MAE:.3f} eV",
         fontsize=7, va="top", color=S2)
ax1.set_xlabel("|Δ band gap| between databases (eV)")
ax1.set_ylabel("composition pairs")
ax1.set_title(f"a  Cross-database disagreement (n = {len(P):,})", loc="left")
ax1.grid(True, axis="y", alpha=0.55); ax1.set_axisbelow(True)

top = bysrc.sort_values("count", ascending=False).head(8).iloc[::-1]
labels = [f"{a}–{b}" for a, b in top.index]
ax2.barh(range(len(top)), top["mean"].values, color=S1, height=0.62, zorder=3)
ax2.axvline(STACK_MAE, color=S2, lw=1.4, ls="--", zorder=5)
for i, (v, n) in enumerate(zip(top["mean"].values, top["count"].values)):
    ax2.text(v, i, f" {v:.2f}  (n={int(n)})", va="center", fontsize=6.8, color=INK_2)
ax2.set_yticks(range(len(top))); ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel("mean |Δ band gap| (eV)")
ax2.set_xlim(0, max(top["mean"].max()*1.45, STACK_MAE*1.3))
ax2.set_title("b  By database pair", loc="left")
ax2.grid(True, axis="x", alpha=0.55); ax2.set_axisbelow(True)
fig.suptitle("Figure E2. Label-noise floor: the databases disagree with each other",
             x=0.02, ha="left", fontsize=9.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("figures/v3/FigE2_label_noise.png", bbox_inches="tight")
fig.savefig("figures/v3/FigE2_label_noise.pdf", bbox_inches="tight")
plt.close(fig)
print("\nWrote tables/label_noise_floor.csv, tables/label_noise_pairs.csv,")
print("      figures/v3/FigE2_label_noise.{png,pdf}, screening/v2/label_noise_summary.txt")
