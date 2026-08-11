"""
Shared feature pipeline utilities.
All preprocessing objects (imputer, scaler, feature names) are saved to
data/processed/ and loaded from there. Scripts NEVER refit on non-training data.
FIX 4: Enforces single-fit-transform-on-train, transform-only-elsewhere.
"""
import pickle, os
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from joblib import Parallel, delayed

EPF = ElementProperty.from_preset("magpie")
FEAT_LABELS = EPF.feature_labels()

IMPUTER_PATH   = "data/processed/imputer.pkl"
SCALER_PATH    = "data/processed/scaler.pkl"
FEATNAMES_PATH = "data/processed/feature_names.pkl"


def featurize_formula(formula):
    try:
        return EPF.featurize(Composition(formula))
    except Exception:
        return [np.nan] * len(FEAT_LABELS)


def featurize_df(df, formula_col="formula_pretty", n_jobs=36):
    feats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(featurize_formula)(f) for f in df[formula_col]
    )
    return feats


def load_preprocessors():
    if not all(os.path.exists(p) for p in [IMPUTER_PATH, SCALER_PATH, FEATNAMES_PATH]):
        raise FileNotFoundError(
            "Preprocessors not found. Run 02a_featurize.py first.")
    with open(IMPUTER_PATH,   "rb") as f: imputer    = pickle.load(f)
    with open(SCALER_PATH,    "rb") as f: scaler     = pickle.load(f)
    with open(FEATNAMES_PATH, "rb") as f: feat_names = pickle.load(f)
    return imputer, scaler, feat_names


def transform(df_or_array, imputer, scaler, feat_names,
              formula_col="formula_pretty", precomputed_feats=None):
    """Featurize (if needed), impute, and scale. Never refits."""
    import pandas as pd
    if precomputed_feats is not None:
        raw = np.array(precomputed_feats)
    elif isinstance(df_or_array, pd.DataFrame):
        raw = np.array(featurize_df(df_or_array, formula_col=formula_col))
    else:
        raw = np.array(df_or_array)

    feat_idx = [FEAT_LABELS.index(f) for f in feat_names if f in FEAT_LABELS]
    raw_sel = raw[:, feat_idx] if raw.ndim == 2 else raw

    X = imputer.transform(raw_sel)
    X = scaler.transform(X)
    return X
