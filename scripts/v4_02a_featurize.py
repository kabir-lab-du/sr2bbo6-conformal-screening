"""
V4 pipeline Stage 2A — Magpie featurization + VIF selection on the v4 (ground-state)
splits. Protocol identical to v2_02a (imputer/scaler/VIF fit on TRAIN ONLY); only the
input paths and the core count differ.
"""
import sys, pickle
sys.path.insert(0, "scripts")
import os
import pandas as pd
import numpy as np

FLAG = "V4_STAGE_2A_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V4 Stage 2A cached — skipping"); sys.exit(0)

from preproc_utils import featurize_df, FEAT_LABELS
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

train_df = pd.read_parquet("data/v4/train.parquet")
test_df  = pd.read_parquet("data/v4/test.parquet")
sr2_df   = pd.read_parquet("data/v4/sr2_holdout.parquet")

ALL_SOURCES = ["MP", "AFLOW", "OQMD", "JARVIS", "OPTIMADE"]
SOURCE_FEAT_COLS = [f"src_{s}" for s in ALL_SOURCES]

def add_source_features(df):
    for src in ALL_SOURCES:
        df[f"src_{src}"] = (df["source"] == src).astype(float)
    return df

train_df = add_source_features(train_df.copy())
test_df  = add_source_features(test_df.copy())
sr2_df   = add_source_features(sr2_df.copy())

N_CORES = 96
print(f"Featurizing train ({len(train_df):,} rows)...")
train_feats_raw = featurize_df(train_df, n_jobs=N_CORES)
print(f"Featurizing test ({len(test_df):,} rows)...")
test_feats_raw  = featurize_df(test_df,  n_jobs=N_CORES)
print(f"Featurizing Sr2 holdout ({len(sr2_df):,} rows)...")
sr2_feats_raw   = featurize_df(sr2_df,   n_jobs=N_CORES)

X_train_raw = np.array(train_feats_raw)
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train_raw)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

print("Computing VIF on full training set...")
vif_vals = []
for i in range(X_train_scaled.shape[1]):
    try:
        v = variance_inflation_factor(X_train_scaled, i)
    except Exception:
        v = 999.0
    vif_vals.append(v)

vif_df = pd.DataFrame({"feature": FEAT_LABELS, "VIF": vif_vals})
keep_features = vif_df[vif_df["VIF"] <= 50]["feature"].tolist()
if len(keep_features) < 10:
    keep_features = vif_df.sort_values("VIF").head(50)["feature"].tolist()
    print("VIF threshold too strict — keeping top-50 lowest VIF features")
print(f"Features after VIF: {len(keep_features)} / {len(FEAT_LABELS)}")
vif_df.to_csv("data/v4/vif_results.csv", index=False)

keep_idx = [FEAT_LABELS.index(f) for f in keep_features]
keep_features_full = keep_features + SOURCE_FEAT_COLS

X_train_sel_raw = X_train_raw[:, keep_idx]
imputer2 = KNNImputer(n_neighbors=5)
imputer2.fit(X_train_sel_raw)
X_train_sel_imp = imputer2.transform(X_train_sel_raw)

src_train = train_df[SOURCE_FEAT_COLS].reset_index(drop=True).values.astype(float)
X_train_full = np.hstack([X_train_sel_imp, src_train])

scaler2 = RobustScaler()
scaler2.fit(X_train_full)

with open("data/v4/imputer.pkl",       "wb") as f: pickle.dump(imputer2, f)
with open("data/v4/scaler.pkl",        "wb") as f: pickle.dump(scaler2, f)
with open("data/v4/feature_names.pkl", "wb") as f: pickle.dump(keep_features_full, f)

def build_feat_df(df, raw_feats, split_name):
    src_cols  = df[SOURCE_FEAT_COLS].reset_index(drop=True)
    raw_arr   = np.array(raw_feats)
    feat_part = pd.DataFrame(raw_arr[:, keep_idx], columns=keep_features)
    meta_cols = ["formula_norm", "band_gap", "formation_energy_per_atom",
                 "energy_above_hull", "source"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    out  = pd.concat([meta, feat_part.reset_index(drop=True),
                      src_cols.reset_index(drop=True)], axis=1)
    out.to_parquet(f"data/v4/{split_name}_featurized.parquet")
    print(f"  {split_name}: {len(out):,} rows × {len(keep_features_full)} features")
    return out

build_feat_df(train_df, train_feats_raw, "train")
build_feat_df(test_df,  test_feats_raw,  "test")
build_feat_df(sr2_df,   sr2_feats_raw,   "sr2_holdout")

open(FLAG, "w").close()
print("V4 Stage 2A complete.")
