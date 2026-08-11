"""
V2 pipeline Stage 3 — Multi-task ResNet ensemble.
Reads: data/v2/{train,test}_featurized.parquet + data/v2/preprocessors
Writes: models/v2/multitask_nn/member_{0..9}.pt + n_feat.pkl

CLEAN-REBUILD CHANGES vs. original v2 script:
  - the original hardcoded `.cuda()` throughout (this ran on a GPU box originally). This
    server has no GPU — every `.cuda()` call has been replaced with `.to(device)`, where
    `device` resolves to CPU here. Also caps torch's CPU thread pool at the 36-core budget.
  - BUG FIX: the original training loss was only `loss_bg + loss_st` — the formation-
    enthalpy (`head_dhf`) and e-hull (`head_ehull`) heads never received a gradient. A
    masked-MSE helper and the `dhf_t`/`ehull_t` target tensors were already defined but
    never used (dead code) — the loss was silently dropped for those two heads at some
    point in this pipeline's history. This matters beyond just "unused capacity": Stage 4
    (`v2_04_screening.py`) feeds the e-hull head's prediction directly into the discovery
    score's stability penalty, so an untrained head was injecting noise into the DFT
    candidate ranking. Now included, masked for rows with NaN formation-enthalpy/e-hull
    labels (not every source database provides both).
"""
import sys, os, pickle
sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd, numpy as np

FLAG = "V2_STAGE_3_COMPLETE.flag"
if os.path.exists(FLAG):
    print("V2 Stage 3 cached — skipping"); sys.exit(0)

os.makedirs("models/v2/multitask_nn", exist_ok=True)

N_CORES = 36  # shared-server budget
torch.set_num_threads(N_CORES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from multitask_arch import MultiTaskNet

with open("data/v2/imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
with open("data/v2/scaler.pkl",        "rb") as f: scaler     = pickle.load(f)
with open("data/v2/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)

train_df = pd.read_parquet("data/v2/train_featurized.parquet")
test_df  = pd.read_parquet("data/v2/test_featurized.parquet")

magpie_names = [f for f in feat_names if not f.startswith("src_")]
src_names    = [f for f in feat_names if f.startswith("src_")]

def _prep(df):
    X_mag = imputer.transform(df[magpie_names].values)
    X_src = df[src_names].values.astype(float)
    return scaler.transform(np.hstack([X_mag, X_src]))

X_train = _prep(train_df)
X_test  = _prep(test_df)

y_bg    = train_df["band_gap"].values
y_dhf   = train_df["formation_energy_per_atom"].values if "formation_energy_per_atom" in train_df.columns else np.full(len(train_df), np.nan)
y_ehull = train_df["energy_above_hull"].values         if "energy_above_hull"         in train_df.columns else np.full(len(train_df), np.nan)
y_stable = (pd.to_numeric(train_df.get("energy_above_hull", pd.Series(dtype=float)),
                           errors="coerce").fillna(1.0) < 0.1).astype(float).values

N_FEAT = X_train.shape[1]
with open("models/v2/multitask_nn/n_feat.pkl", "wb") as f: pickle.dump(N_FEAT, f)
print(f"MultiTask NN: N_FEAT={N_FEAT}, train={len(X_train):,}, test={len(X_test):,}")

def masked_mse(pred, target, device):
    """target: a torch.Tensor already on `device` (batched from the DataLoader)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(pred.squeeze()[mask], target[mask])

Xt        = torch.tensor(X_train, dtype=torch.float32)
yt_bg     = torch.tensor(y_bg,    dtype=torch.float32)
yt_stable = torch.tensor(y_stable, dtype=torch.float32)
dhf_t     = torch.tensor(y_dhf,   dtype=torch.float32)
ehull_t   = torch.tensor(y_ehull, dtype=torch.float32)
dataset   = TensorDataset(Xt, yt_bg, yt_stable, dhf_t, ehull_t)
loader    = DataLoader(dataset, batch_size=256, shuffle=True,
                       num_workers=0, pin_memory=False)

members = []
for seed in range(10):
    torch.manual_seed(seed)
    model = MultiTaskNet(N_FEAT).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)

    for epoch in range(500):
        model.train()
        for Xb, yb_bg, yb_st, yb_dhf, yb_eh in loader:
            Xb = Xb.to(device)
            p_bg, p_dhf, p_eh, p_st = model(Xb)
            loss_bg   = F.mse_loss(p_bg.squeeze(), yb_bg.to(device))
            loss_st   = F.binary_cross_entropy(p_st.squeeze(), yb_st.to(device))
            loss_dhf  = masked_mse(p_dhf, yb_dhf.to(device), device)
            loss_eh   = masked_mse(p_eh,  yb_eh.to(device),  device)
            loss = loss_bg + loss_st + loss_dhf + loss_eh
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xt_dev = torch.tensor(X_train, dtype=torch.float32).to(device)
                p_bg_tr, _, _, _ = model(Xt_dev)
                tr_mae = float(torch.mean(torch.abs(p_bg_tr.squeeze() - yt_bg.to(device))))
            print(f"  Seed {seed} epoch {epoch:3d}: train MAE={tr_mae:.4f} eV")

    torch.save(model.state_dict(), f"models/v2/multitask_nn/member_{seed}.pt")
    members.append(seed)
    print(f"  Saved member_{seed}.pt")

# Final evaluation on test
model.eval()
Xtest_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_bg = test_df["band_gap"].values
preds = []
for seed in range(10):
    m = MultiTaskNet(N_FEAT).to(device)
    m.load_state_dict(torch.load(f"models/v2/multitask_nn/member_{seed}.pt", map_location=device))
    m.eval()
    with torch.no_grad():
        pb, _, _, _ = m(Xtest_t)
        preds.append(pb.cpu().numpy().squeeze())

y_ens = np.stack(preds).mean(0)
from sklearn.metrics import r2_score, mean_absolute_error
print(f"NN ensemble Test R²={r2_score(y_test_bg, y_ens):.4f}, "
      f"MAE={mean_absolute_error(y_test_bg, y_ens):.4f} eV")

open(FLAG, "w").close()
print("V2 Stage 3 complete.")
