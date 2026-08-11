"""
V2 pipeline Stage 2B — Optuna GBDT stacking + MAPIE conformal prediction.
Reads: data/v2/{train,test,sr2_holdout}_featurized.parquet + data/v2/preprocessors
Writes: models/v2/{stacking,mapie,gp,top15_idx}.pkl + metrics_comparison.csv

CLEAN-REBUILD CHANGES vs. original v2 script:
  - thread_count/n_jobs capped at 36 (shared server core budget), not 40.
  - All three Optuna studies use persistent sqlite storage (optuna_studies/gbdt_v3.db)
    with load_if_exists=True, so a killed/crashed run resumes from its last completed
    trial instead of restarting. Safe to just re-run this script after any interruption.
  - CatBoost and LightGBM studies are seeded (enqueue_trial) with the hyperparameters
    extracted from the last known-good models/v2/{catboost,lgbm}.pkl (see
    reference/old_hyperparams/*_optuna_seed.json) — those two were verified correct in
    the prior audit, so seeding just gives Optuna a head start near a good region on the
    new (CMR-excluded) data; it still runs a full search, doesn't skip it.
  - XGBoost is deliberately NOT seeded — it's the model that was found corrupted
    (src_CMR at 58% importance), and we don't want to bias a fresh search toward
    whatever produced that. It gets a full, unseeded 80-trial search.
  - After computing top15_idx from the new XGBoost, this script now prints the top-15
    feature names + importances and flags if any single feature exceeds 40% of total
    importance — an automatic version of the check that caught the original corruption.
  - TRIAL BUDGET REDUCED from the original 100/80/80 to 30/40/40 (CatBoost/XGBoost/
    LightGBM), decided 2026-07-29 after measuring actual cost on this server: CatBoost
    trials ran ~17 min each (depth 9, ~2,700 iterations, 5-fold CV, on 36 contended
    cores), projecting ~28 h for CatBoost alone and ~33 h for all of Stage 2B.
    Justification for the cut, in order of weight:
      1. This project's own prior evidence: ~/ML_Project/PROJECT_LOG.md (the sibling
         PINN pipeline, same team/data/problem) records that going from 25/20/20 to
         200/200/200 Optuna trials produced "essentially no improvement in bandgap
         prediction" — an 8-10x budget increase for minimal gain.
      2. CatBoost and LightGBM are SEEDED here with known-good hyperparameters from the
         previously-verified models, so the search starts in a good region rather than
         cold. Trial 0 (the seed) already scored 0.3538 eV CV MAE.
      3. TPE converges quickly in these 5-7 dimensional spaces.
    This is a deliberate, documented deviation from the original methodology — it MUST be
    stated in the manuscript's methods section rather than reporting the old 100/80/80.
  - Resume logic now counts only COMPLETE trials toward the budget. Previously it used
    len(study.trials), which also counts FAIL and stale RUNNING trials — so a crash
    mid-trial would permanently consume a slot from the budget on every resume.
"""
import sys, pickle, os, json
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V2_STAGE_2B_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V2 Stage 2B cached — skipping"); sys.exit(0)

os.makedirs("models/v2", exist_ok=True)
os.makedirs("optuna_studies", exist_ok=True)

N_CORES = 36  # shared-server budget — do not raise without checking `uptime` / other users first

# Optuna trial budgets — reduced from the original 100/80/80; see module docstring for
# the measured-cost + prior-evidence justification. Raising these back up is safe but
# costs roughly 17 min per additional CatBoost trial on this hardware.
N_TRIALS_CB   = 30
N_TRIALS_XGB  = 40
N_TRIALS_LGBM = 40

def n_complete(study):
    """Trials that actually finished. Excludes FAIL/PRUNED and stale RUNNING trials left
    behind by a killed process — those must not consume budget slots on resume."""
    return len([t for t in study.trials if t.state.name == "COMPLETE"])

def load_seed(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

CB_SEED   = load_seed("reference/old_hyperparams/catboost_optuna_seed.json")
LGBM_SEED = load_seed("reference/old_hyperparams/lgbm_optuna_seed.json")

with open("data/v2/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v2/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v2/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from mapie.regression import CrossConformalRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

train_df = pd.read_parquet("data/v2/train_featurized.parquet")
test_df  = pd.read_parquet("data/v2/test_featurized.parquet")
sr2_df   = pd.read_parquet("data/v2/sr2_holdout_featurized.parquet")

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

def prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

X_train = prep(train_df); y_train = train_df["band_gap"].values
X_test  = prep(test_df);  y_test  = test_df["band_gap"].values
X_sr2   = prep(sr2_df);   y_sr2   = sr2_df["band_gap"].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}, Sr2: {X_sr2.shape}")

# CatBoost Optuna
def cb_objective(trial):
    p = {
        "iterations":      trial.suggest_int("iterations", 500, 3000),
        "depth":           trial.suggest_int("depth", 4, 10),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "task_type": "CPU", "thread_count": N_CORES, "verbose": 0, "early_stopping_rounds": 50,
    }
    scores = cross_val_score(CatBoostRegressor(**p), X_train, y_train,
                             cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
    return -scores.mean()

print(f"Optimizing CatBoost ({N_TRIALS_CB} trials)...")
cb_study = optuna.create_study(direction="minimize", study_name="catboost_v3",
                                storage="sqlite:///optuna_studies/gbdt_v3.db", load_if_exists=True)
if len(cb_study.trials) == 0 and CB_SEED is not None:
    cb_study.enqueue_trial(CB_SEED)
    print(f"Seeded CatBoost study with prior known-good params: {CB_SEED}")
n_remaining = max(0, N_TRIALS_CB - n_complete(cb_study))
print(f"CatBoost: {n_complete(cb_study)} trials complete, running {n_remaining} more")
cb_study.optimize(cb_objective, n_trials=n_remaining, n_jobs=1)
best_cb = CatBoostRegressor(**cb_study.best_params,
                             task_type="CPU", thread_count=N_CORES, verbose=0)
best_cb.fit(X_train, y_train)

# XGBoost Optuna
def xgb_objective(trial):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "tree_method": "hist", "n_jobs": 1, "device": "cpu",
    }
    scores = cross_val_score(XGBRegressor(**p), X_train, y_train,
                             cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
    return -scores.mean()

print(f"Optimizing XGBoost ({N_TRIALS_XGB} trials, unseeded — see module docstring)...")
xgb_study = optuna.create_study(direction="minimize", study_name="xgboost_v3",
                                 storage="sqlite:///optuna_studies/gbdt_v3.db", load_if_exists=True)
n_remaining = max(0, N_TRIALS_XGB - n_complete(xgb_study))
print(f"XGBoost: {n_complete(xgb_study)} trials complete, running {n_remaining} more")
xgb_study.optimize(xgb_objective, n_trials=n_remaining, n_jobs=5)
best_xgb = XGBRegressor(**xgb_study.best_params,
                          tree_method="hist", n_jobs=N_CORES, device="cpu")
best_xgb.fit(X_train, y_train)

# LightGBM Optuna
def lgbm_objective(trial):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "device": "cpu", "n_jobs": 1, "verbosity": -1,
    }
    scores = cross_val_score(LGBMRegressor(**p), X_train, y_train,
                             cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
    return -scores.mean()

print(f"Optimizing LightGBM ({N_TRIALS_LGBM} trials)...")
lgbm_study = optuna.create_study(direction="minimize", study_name="lightgbm_v3",
                                  storage="sqlite:///optuna_studies/gbdt_v3.db", load_if_exists=True)
if len(lgbm_study.trials) == 0 and LGBM_SEED is not None:
    lgbm_study.enqueue_trial(LGBM_SEED)
    print(f"Seeded LightGBM study with prior known-good params: {LGBM_SEED}")
n_remaining = max(0, N_TRIALS_LGBM - n_complete(lgbm_study))
print(f"LightGBM: {n_complete(lgbm_study)} trials complete, running {n_remaining} more")
lgbm_study.optimize(lgbm_objective, n_trials=n_remaining, n_jobs=5)
best_lgbm = LGBMRegressor(**lgbm_study.best_params, device="cpu", n_jobs=N_CORES, verbosity=-1)
best_lgbm.fit(X_train, y_train)

# Stacking ensemble
stacking = StackingRegressor(
    estimators=[("catboost", best_cb), ("xgboost", best_xgb), ("lgbm", best_lgbm)],
    final_estimator=Ridge(alpha=1.0),
    passthrough=False, n_jobs=1,
)
print("Fitting stacking ensemble...")
stacking.fit(X_train, y_train)

# Evaluate all models
metrics = []
for name, model in [("CatBoost", best_cb), ("XGBoost", best_xgb),
                     ("LightGBM", best_lgbm), ("Stacking", stacking)]:
    yp_test = model.predict(X_test)
    yp_sr2  = model.predict(X_sr2)
    metrics.append({
        "model":     name,
        "global_r2": r2_score(y_test, yp_test),
        "global_mae": mean_absolute_error(y_test, yp_test),
        "sr2_r2":    r2_score(y_sr2,  yp_sr2),
        "sr2_mae":   mean_absolute_error(y_sr2,  yp_sr2),
    })
    print(f"{name}: Test R²={metrics[-1]['global_r2']:.4f}, MAE={metrics[-1]['global_mae']:.4f} eV"
          f" | Sr2 R²={metrics[-1]['sr2_r2']:.4f}, MAE={metrics[-1]['sr2_mae']:.4f} eV")

pd.DataFrame(metrics).to_csv("models/v2/metrics_comparison.csv", index=False)

# MAPIE conformal prediction on stacking
print("Fitting MAPIE conformal predictor...")
mapie = CrossConformalRegressor(estimator=stacking, method="plus", cv=5, n_jobs=1)
mapie.fit_conformalize(X_train, y_train)
_, y_pis = mapie.predict_interval(X_test)
ci_widths = (y_pis[:, 1, 0] - y_pis[:, 0, 0])
coverage  = np.mean((y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0]))
print(f"MAPIE: test coverage={coverage:.3f}, median CI width={np.median(ci_widths):.3f} eV")

# CLEAN-REBUILD ADDITION: the original script only ever computed/printed coverage on the
# in-distribution test set. But AUDIT_REPORT.md's coverage finding (~0.87-0.90 with a
# working model) is specifically about the Sr2 HOLDOUT (true out-of-domain generalization
# check), not the test set — the two are not interchangeable and RUNBOOK.md's stage-2B
# verification step asks for this number, so compute it.
_, y_pis_sr2 = mapie.predict_interval(X_sr2)
ci_widths_sr2 = (y_pis_sr2[:, 1, 0] - y_pis_sr2[:, 0, 0])
coverage_sr2  = np.mean((y_sr2 >= y_pis_sr2[:, 0, 0]) & (y_sr2 <= y_pis_sr2[:, 1, 0]))
print(f"MAPIE: Sr2 holdout coverage={coverage_sr2:.3f}, median CI width={np.median(ci_widths_sr2):.3f} eV"
      f"  (this is the number to compare against AUDIT_REPORT.md's ~0.87-0.90 expectation)")

# GP surrogate on top-15 features for Bayesian acquisition
feat_importances = best_xgb.feature_importances_
top15_idx = np.argsort(feat_importances)[-15:]

# Sanity check — this is what would have caught the original corruption (src_CMR at 58%
# of total XGBoost importance). Re-run this print any time a model is retrained.
total_imp = feat_importances.sum()
ranked = sorted(zip(feat_names, feat_importances), key=lambda t: -t[1])
print("\nXGBoost top-15 feature importances (sanity check):")
for name, imp in ranked[:15]:
    frac = imp / total_imp if total_imp > 0 else 0
    flag = "  <-- FLAG: >40% of total importance, investigate before trusting this model" if frac > 0.40 else ""
    print(f"  {name:30s} {imp:.4f}  ({frac:.1%}){flag}")

gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=5,
                               normalize_y=True, alpha=0.01)
gp.fit(X_train[:, top15_idx], y_train)

# Save models
with open("models/v2/mapie.pkl",    "wb") as f: pickle.dump(mapie, f)
with open("models/v2/gp.pkl",      "wb") as f: pickle.dump(gp, f)
with open("models/v2/top15_idx.pkl","wb") as f: pickle.dump(top15_idx, f)
with open("models/v2/stacking.pkl","wb") as f: pickle.dump(stacking, f)
with open("models/v2/catboost.pkl","wb") as f: pickle.dump(best_cb, f)
with open("models/v2/xgboost.pkl", "wb") as f: pickle.dump(best_xgb, f)
with open("models/v2/lgbm.pkl",    "wb") as f: pickle.dump(best_lgbm, f)

open(FLAG, "w").close()
print("V2 Stage 2B complete.")
