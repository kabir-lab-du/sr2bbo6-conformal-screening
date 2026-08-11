"""
V4 Stage 7 (post-hoc, read-only w.r.t. models) — export per-sample predictions,
conformal intervals, and feature importances for the manuscript figures.
Prediction-only: no fitting. GBDT threads stay at 8 via the pickled models.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import pickle
import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/sr2bbo6_stacking_v3"))
OUT = "screening/v4"

with open("data/v4/imputer.pkl", "rb") as f: imputer = pickle.load(f)
with open("data/v4/scaler.pkl", "rb") as f: scaler = pickle.load(f)
with open("data/v4/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v4/stacking.pkl", "rb") as f: stacking = pickle.load(f)

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names = [f for f in feat_names if f.startswith("src_")]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

def formula_col(feat_df, raw_path):
    if "formula" in feat_df.columns:
        return feat_df["formula"].values
    raw = pd.read_parquet(raw_path)
    if "formula" in raw.columns and len(raw) == len(feat_df):
        return raw["formula"].values
    return np.array([""] * len(feat_df))

def src_label(df):
    cols = [c for c in src_names if c in df.columns]
    if not cols:
        return np.array(["?"] * len(df))
    arr = df[cols].values.astype(float)
    lab = np.array([cols[i].replace("src_", "") if arr[r].sum() > 0 else "other"
                    for r, i in zip(range(len(df)), arr.argmax(axis=1))])
    return lab

test_df = pd.read_parquet("data/v4/test_featurized.parquet")
sr2_df = pd.read_parquet("data/v4/sr2_holdout_featurized.parquet")
train_df = pd.read_parquet("data/v4/train_featurized.parquet")

X_test, y_test = prep(test_df), test_df["band_gap"].values
X_sr2, y_sr2 = prep(sr2_df), sr2_df["band_gap"].values

print("predicting (stacking)...")
yp_test = stacking.predict(X_test)
yp_sr2 = stacking.predict(X_sr2)

print("loading MAPIE (large pickle)...")
with open("models/v4/mapie.pkl", "rb") as f: mapie = pickle.load(f)
_, pis_test = mapie.predict_interval(X_test)
_, pis_sr2 = mapie.predict_interval(X_sr2)

pd.DataFrame({
    "formula": formula_col(test_df, "data/v4/test.parquet"),
    "src": src_label(test_df),
    "y_true": y_test, "y_pred": yp_test,
    "ci_lo": pis_test[:, 0, 0], "ci_hi": pis_test[:, 1, 0],
}).to_csv(f"{OUT}/test_predictions.csv", index=False)

pd.DataFrame({
    "formula": formula_col(sr2_df, "data/v4/sr2_holdout.parquet"),
    "src": src_label(sr2_df),
    "y_true": y_sr2, "y_pred": yp_sr2,
    "ci_lo": pis_sr2[:, 0, 0], "ci_hi": pis_sr2[:, 1, 0],
}).to_csv(f"{OUT}/sr2_predictions.csv", index=False)

print("feature importances...")
imps = {}
with open("models/v4/xgboost.pkl", "rb") as f:
    m = pickle.load(f)
    imps["xgboost"] = np.asarray(m.feature_importances_, dtype=float)
with open("models/v4/lgbm.pkl", "rb") as f:
    m = pickle.load(f)
    imps["lightgbm"] = np.asarray(m.feature_importances_, dtype=float)
with open("models/v4/catboost.pkl", "rb") as f:
    m = pickle.load(f)
    imps["catboost"] = np.asarray(m.get_feature_importance(), dtype=float)

fi = pd.DataFrame({"feature": feat_names})
for k, v in imps.items():
    fi[k] = v / v.sum()
fi["mean_norm"] = fi[list(imps)].mean(axis=1)
fi.sort_values("mean_norm", ascending=False).to_csv(f"{OUT}/feature_importance.csv", index=False)

pd.DataFrame({
    "band_gap": train_df["band_gap"].values,
    "src": src_label(train_df),
}).to_csv(f"{OUT}/train_label_dist.csv", index=False)

print("coverage check: test %.4f | sr2 %.4f" % (
    np.mean((y_test >= pis_test[:, 0, 0]) & (y_test <= pis_test[:, 1, 0])),
    np.mean((y_sr2 >= pis_sr2[:, 0, 0]) & (y_sr2 <= pis_sr2[:, 1, 0]))))
print("EXPORT DONE")
