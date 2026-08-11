"""
V2 pipeline Stage 4 — screen Sr2BB'O6 candidates with V2 models.
Reads: data/processed/screening_set.parquet (same candidates as V1)
       models/v2/{mapie,gp,top15_idx}.pkl + multitask_nn/
Writes: screening/v2/new_ranked_candidates.csv

CLEAN-REBUILD CHANGES vs. original v2 script:
  - `.cuda()` replaced with device-agnostic `.to(device)` (this server has no GPU) and
    n_jobs capped at 36.
  - The NN-prediction block now tolerates a missing `models/v2/multitask_nn/` entirely
    (no n_feat.pkl, no member_*.pt) instead of crashing — the original unconditionally
    opened n_feat.pkl even when the neutral-prior fallback path below it implied "NN is
    optional." If you decided not to retrain the multitask NN (see RUNBOOK.md §5), this
    script now runs anyway, with p_stable/ehull_pred as neutral priors instead of a hard
    crash.
  - Fixed the neutral-prior fallback values: the original used ehull_mean=0.1, which
    makes discovery_score's hull_pen = max(0, 1 - 0.1*10) = 0.0 for every single
    candidate — the "fallback" silently zeroed the entire ranking rather than being
    neutral. Changed to ehull_mean=0.05 (hull_pen=0.5, a genuine midpoint) so a
    missing-NN run still produces a meaningful (if less informative) ranking.
"""
import sys, pickle, os
sys.path.insert(0, "scripts")
import pandas as pd, numpy as np

FLAG = "V2_STAGE_4_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V2 Stage 4 cached — skipping"); sys.exit(0)

os.makedirs("screening/v2", exist_ok=True)
N_CORES = 36  # shared-server budget

with open("data/v2/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v2/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v2/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
with open("models/v2/mapie.pkl",       "rb") as f: mapie      = pickle.load(f)
with open("models/v2/top15_idx.pkl",   "rb") as f: top15      = pickle.load(f)
with open("models/v2/gp.pkl",         "rb") as f: gp         = pickle.load(f)

from preproc_utils import featurize_df, FEAT_LABELS

screen_df = pd.read_parquet("data/processed/screening_set.parquet")
new_candidates = screen_df[~screen_df["in_prior_screen"]].copy()
print(f"New candidates to screen: {len(new_candidates):,}")

new_candidates["source"] = "SCREEN"
raw_feats = featurize_df(new_candidates, formula_col="formula", n_jobs=N_CORES)

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

# Build source feature block (all zeros for screening)
src_df = pd.DataFrame({c: 0.0 for c in src_names}, index=range(len(new_candidates)))

raw_arr = np.array(raw_feats)
magpie_idx = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]
raw_sel    = raw_arr[:, magpie_idx]
src_block  = src_df[src_names].values
X_screen   = scaler.transform(np.hstack([imputer.transform(raw_sel), src_block]))

y_pred = mapie.predict(X_screen)
_, y_pis = mapie.predict_interval(X_screen)
y_gp_mu, y_gp_std = gp.predict(X_screen[:, top15], return_std=True)

# Multitask NN predictions — optional. If Stage 3 wasn't (re)run, n_feat.pkl won't
# exist; fall back to neutral priors rather than crashing.
n_feat_path = "models/v2/multitask_nn/n_feat.pkl"
preds_bg, preds_stable, preds_ehull = [], [], []
if os.path.exists(n_feat_path):
    import torch
    from multitask_arch import MultiTaskNet
    with open(n_feat_path, "rb") as f: N_FEAT = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xsc_t = torch.tensor(X_screen, dtype=torch.float32).to(device)
    for seed in range(10):
        mpath = f"models/v2/multitask_nn/member_{seed}.pt"
        if not os.path.exists(mpath):
            continue
        m = MultiTaskNet(N_FEAT).to(device)
        m.load_state_dict(torch.load(mpath, map_location=device))
        m.eval()
        with torch.no_grad():
            pb, _, pe, ps = m(Xsc_t)
            preds_bg.append(pb.cpu().numpy().squeeze())
            preds_ehull.append(pe.cpu().numpy().squeeze())
            preds_stable.append(ps.cpu().numpy().squeeze())
else:
    print("NOTE: models/v2/multitask_nn/n_feat.pkl not found — Stage 3 was not run for "
          "this dataset. Proceeding with neutral p_stable/ehull priors (see RUNBOOK.md §5).")

if preds_bg:
    stable_mean = np.stack(preds_stable).mean(0)
    ehull_mean  = np.stack(preds_ehull).mean(0)
else:
    stable_mean = np.full(len(new_candidates), 0.5)
    ehull_mean  = np.full(len(new_candidates), 0.05)  # -> hull_pen=0.5, a real midpoint (see docstring)
    print("WARNING: No NN members found; using neutral priors")

def discovery_score(bg, ci_lo, ci_hi, p_stable, ehull):
    ci_w = max(ci_hi - ci_lo, 0.1)
    pv_overlap = max(0, min(ci_hi, 1.8) - max(ci_lo, 1.2))
    pv_frac    = pv_overlap / ci_w
    metal_pen  = 1.0 if bg > 0.3 else 0.1
    hull_pen   = max(0.0, 1.0 - ehull * 10)
    return pv_frac * p_stable * metal_pen * hull_pen

new_candidates = new_candidates.reset_index(drop=True)
new_candidates["bg_pred"]    = y_pred
new_candidates["bg_ci_lo"]   = y_pis[:, 0, 0]
new_candidates["bg_ci_hi"]   = y_pis[:, 1, 0]
new_candidates["p_stable"]   = stable_mean
new_candidates["ehull_pred"] = ehull_mean
new_candidates["score"] = [
    discovery_score(bg, lo, hi, ps, eh)
    for bg, lo, hi, ps, eh in zip(
        y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0], stable_mean, ehull_mean)
]

ranked = new_candidates.sort_values("score", ascending=False)
ranked.to_csv("screening/v2/new_ranked_candidates.csv", index=False)
print("V2 Top-10 new candidates:")
print(ranked.head(10)[["formula", "bg_pred", "bg_ci_lo", "bg_ci_hi",
                        "p_stable", "score"]].to_string())

open(FLAG, "w").close()
print("V2 Stage 4 complete.")
