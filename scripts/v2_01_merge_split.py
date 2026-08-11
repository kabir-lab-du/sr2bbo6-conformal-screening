"""
V2 pipeline Stage 1 — merge all v2_sources, filter, split.
Reads: data/v2_sources/{mp_all,oqmd_all,jarvis_all,cmr_all,aflow_all,optimade}.parquet
Writes: data/v2/{full_A2BBO6,train,test,sr2_holdout}.parquet

CLEAN-REBUILD CHANGE vs. original v2 script: CMR (GLLB-SC functional) entries are now
dropped right after merge (Step 0b). Reasons, per reference/AUDIT_REPORT.md from the prior
audit of this pipeline:
  1. The manuscript's stated dataset sizes (4,737/833/536) already assume CMR excluded;
     the previously-deployed models were trained on the full CMR-included set (5,381/
     950/567), a text/model mismatch.
  2. `src_CMR` was the #1 SHAP feature for every GBDT learner (CatBoost, LightGBM, and
     the corrupted XGBoost alike) — a database-provenance confound (GLLB-SC vs PBE gaps),
     not real chemistry. Excluding CMR removes both the confounded rows and the
     (now-constant, uninformative) src_CMR one-hot column.
"""
import os, sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from utils import compute_bartel_tau, norm_formula
from sklearn.model_selection import train_test_split

FLAG = "V2_STAGE_1_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V2 Stage 1 cached — skipping"); sys.exit(0)

os.makedirs("data/v2", exist_ok=True)

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
    bg_n = df["band_gap"].notna().sum()
    print(f"Loaded {name}: {len(df):,} rows, band_gap={bg_n:,}")

raw = pd.concat(dfs, ignore_index=True)
print(f"\nStep 0 — Raw combined: {len(raw):,}")

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

raw["is_A2BBO6"] = raw["formula_pretty"].apply(is_A2BBO6)
step1 = raw[raw["is_A2BBO6"]].copy()
print(f"Step 1 — A2BB'O6 stoichiometry: {len(step1):,}")

step2 = step1[step1["is_metal"].fillna(False) == False].copy()
step2 = step2[step2["band_gap"].fillna(0) >= 0.05]
print(f"Step 2 — Non-metals (gap >= 0.05 eV): {len(step2):,}")

step2["bartel_tau"] = step2["formula_pretty"].apply(compute_bartel_tau)
step3 = step2[step2["bartel_tau"] <= 6.0].copy()
print(f"Step 3 — Bartel tau <= 6.0: {len(step3):,}")

SOURCE_PRIORITY = {"CMR": 0, "MP": 1, "JARVIS": 2, "AFLOW": 3, "OQMD": 4, "OPTIMADE": 5}
step3["formula_norm"] = step3["formula_pretty"].apply(norm_formula)
step3["source_rank"]  = step3["source"].map(SOURCE_PRIORITY).fillna(99)
step4 = (step3.sort_values("source_rank")
              .drop_duplicates(subset="formula_norm", keep="first")
              .copy())
print(f"Step 4 — Deduplicated: {len(step4):,}")

step5 = step4[step4["band_gap"].notna()].copy()
print(f"Step 5 — Non-null band gap: {len(step5):,}")

print(f"\nSource breakdown:\n{step5['source'].value_counts().to_string()}")
print(f"Band gap stats:\n{step5['band_gap'].describe().to_string()}")

def _sanitize_df(df):
    """Cast mixed-type object columns to str so pyarrow can write parquet."""
    out = df.copy()
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].where(out[col].isna(), out[col].astype(str))
    return out

_sanitize_df(step5).to_parquet("data/v2/full_A2BBO6.parquet")

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

_sanitize_df(train_df).to_parquet("data/v2/train.parquet")
_sanitize_df(test_df).to_parquet("data/v2/test.parquet")
_sanitize_df(sr2_df).to_parquet("data/v2/sr2_holdout.parquet")
print(f"Train: {len(train_df):,} / Test: {len(test_df):,} / Sr2 holdout: {len(sr2_df):,}")

open(FLAG, "w").close()
print("V2 Stage 1 complete.")
