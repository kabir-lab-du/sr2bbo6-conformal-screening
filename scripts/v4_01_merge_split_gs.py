"""
V4 pipeline Stage 1 — merge, filter, GROUND-STATE dedup, split.

Change vs v2_01 (FABLE_REVIEW.md F2): Stage-1 dedup previously kept an effectively
arbitrary row per composition (unstable sort by source priority, keep="first"), so
labels were "the gap of whichever entry happened to sort first" — within-OQMD spread
for multi-entry compositions is 0.65 eV mean range. V4 keeps the LOWEST
energy_above_hull entry per composition (the ground-state polymorph). Fallback when no
entry in the group has e_hull (AFLOW rows): winning source by priority, label = that
source's MEDIAN gap for the composition (defensible central value, not an arbitrary row).

Everything else (CMR exclusion, filters, Sr2 segregation, stratified split, seed 42) is
IDENTICAL to v2_01. The composition set is unchanged; only labels (and hence the
stratification bins / train-test row membership) can shift. The Sr2 holdout membership
is identical to v3 by construction (mask depends only on formulas).
"""
import os, sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from joblib import Parallel, delayed
from utils import compute_bartel_tau, norm_formula
from sklearn.model_selection import train_test_split

FLAG = "V4_STAGE_1_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 1 cached — skipping"); sys.exit(0)

os.makedirs("data/v4", exist_ok=True)
N_CORES = 96

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
        print(f"MISSING: {path} — skipping"); continue
    df = pd.read_parquet(path)
    if len(df) == 0:
        print(f"{name}: empty — skipping"); continue
    df["source"] = df["source"].str.replace(r"^CMR-.*", "CMR", regex=True)
    for col in ["is_metal", "formation_energy_per_atom", "energy_above_hull", "functional"]:
        if col not in df.columns:
            df[col] = None
    dfs.append(df)
    print(f"Loaded {name}: {len(df):,} rows, band_gap={df['band_gap'].notna().sum():,}")

raw = pd.concat(dfs, ignore_index=True)
print(f"\nStep 0  — Raw combined: {len(raw):,}")

n_before_cmr = len(raw)
raw = raw[raw["source"] != "CMR"].copy()
print(f"Step 0b — Excluded CMR (GLLB-SC functional confound): "
      f"{n_before_cmr:,} -> {len(raw):,} ({n_before_cmr - len(raw):,} dropped)")

def is_A2BBO6(formula):
    try:
        comp = Composition(formula).reduced_composition
        elems = list(comp.elements)
        if len(elems) < 3 or len(elems) > 5:
            return False
        o_count = comp["O"]
        cation_count = sum(v for k, v in comp.items() if str(k) != "O")
        return abs(cation_count / o_count - (4/6)) < 0.05
    except Exception:
        return False

flags = Parallel(n_jobs=N_CORES)(delayed(is_A2BBO6)(f) for f in raw["formula_pretty"])
step1 = raw[np.array(flags)].copy()
print(f"Step 1  — A2BB'O6-stoichiometry oxides: {len(step1):,}")

step2 = step1[step1["is_metal"].fillna(False) == False].copy()
step2 = step2[step2["band_gap"].fillna(0) >= 0.05]
print(f"Step 2  — Non-metals (gap >= 0.05 eV): {len(step2):,}")

taus = Parallel(n_jobs=N_CORES)(delayed(compute_bartel_tau)(f) for f in step2["formula_pretty"])
step2["bartel_tau"] = taus
step3 = step2[step2["bartel_tau"] <= 6.0].copy()
print(f"Step 3  — Bartel tau <= 6.0: {len(step3):,}")

# ---- V4 ground-state dedup ----------------------------------------------------------
SOURCE_PRIORITY = {"CMR": 0, "MP": 1, "JARVIS": 2, "AFLOW": 3, "OQMD": 4, "OPTIMADE": 5}
step3["formula_norm"] = step3["formula_pretty"].apply(norm_formula)
step3["source_rank"]  = step3["source"].map(SOURCE_PRIORITY).fillna(99)
step3["ehull_num"]    = pd.to_numeric(step3["energy_above_hull"], errors="coerce")

n_groups_multi = (step3.groupby("formula_norm").size() > 1).sum()

def pick_ground_state(g):
    with_eh = g[g["ehull_num"].notna()]
    if len(with_eh):
        return (with_eh.sort_values(["ehull_num", "source_rank", "band_gap"],
                                    kind="mergesort").iloc[0])
    best_rank = g["source_rank"].min()
    gs = g[g["source_rank"] == best_rank]
    row = gs.sort_values("band_gap", kind="mergesort").iloc[0].copy()
    row["band_gap"] = gs["band_gap"].median()
    return row

step4 = (step3.groupby("formula_norm", group_keys=False, sort=False)
              .apply(pick_ground_state))
step4 = step4.reset_index(drop=True)
print(f"Step 4  — Ground-state dedup: {len(step4):,} "
      f"({n_groups_multi:,} compositions had >1 entry)")

step5 = step4[step4["band_gap"].notna()].copy()
print(f"Step 5  — Non-null band gap: {len(step5):,}")

# ---- label-shift accounting vs v3 (for MANIFEST) ------------------------------------
if os.path.exists("data/v2/full_A2BBO6.parquet"):
    v3 = pd.read_parquet("data/v2/full_A2BBO6.parquet")[["formula_norm", "band_gap"]]
    v3 = v3.rename(columns={"band_gap": "bg_v3"})
    cmp = step5[["formula_norm", "band_gap"]].merge(v3, on="formula_norm", how="inner")
    d = (cmp["band_gap"] - cmp["bg_v3"].astype(float)).abs()
    print(f"\nLabel shift vs v3 (same compositions, n={len(cmp):,}):")
    print(f"  labels changed by >0.01 eV : {(d > 0.01).sum():,} ({(d > 0.01).mean():.1%})")
    print(f"  mean |Δ| {d.mean():.4f}  median {d.median():.4f}  "
          f"p90 {d.quantile(.9):.4f}  max {d.max():.4f} eV")

print(f"\nSource breakdown:\n{step5['source'].value_counts().to_string()}")
print(f"Band gap stats:\n{step5['band_gap'].describe().to_string()}")

def _sanitize_df(df):
    out = df.copy()
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].where(out[col].isna(), out[col].astype(str))
    return out

_sanitize_df(step5).to_parquet("data/v4/full_A2BBO6.parquet")

def is_sr2_compound(formula_norm):
    try:
        comp = Composition(formula_norm).reduced_composition
        sr_count = comp.get("Sr", 0)
        o_count  = comp.get("O",  0)
        return (sr_count > 0 and o_count > 0 and
                abs(sr_count / o_count - 1/3) < 0.05)
    except Exception:
        return False

sr2_mask = step5["formula_norm"].apply(is_sr2_compound)
sr2_df   = step5[sr2_mask].copy()
rest_df  = step5[~sr2_mask].copy()
print(f"\nSr2 holdout: {len(sr2_df):,}, Non-Sr2 pool: {len(rest_df):,}")

rest_df["bg_bin"] = pd.cut(rest_df["band_gap"], bins=10, labels=False).fillna(0).astype(int)
train_df, test_df = train_test_split(
    rest_df, test_size=0.15, stratify=rest_df["bg_bin"], random_state=42
)

_sanitize_df(train_df).to_parquet("data/v4/train.parquet")
_sanitize_df(test_df).to_parquet("data/v4/test.parquet")
_sanitize_df(sr2_df).to_parquet("data/v4/sr2_holdout.parquet")
print(f"Train: {len(train_df):,} / Test: {len(test_df):,} / Sr2 holdout: {len(sr2_df):,}")

open(FLAG, "w").close()
print("V4 Stage 1 complete.")
