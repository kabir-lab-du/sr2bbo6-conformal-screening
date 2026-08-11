"""
V3 pipeline Stage 9 — independent review checks (second-opinion audit, 2026-08-01).

Three empirical questions raised by the code audit, none previously measured:

  CHECK 1 — src-flag sensitivity.
    v2_04/v3_06 predict candidates with ALL src_* = 0, a configuration absent from
    training (every training row has exactly one src flag set). Measure how much the
    296 candidate predictions, the tier assignment, and the DFT Table-3 comparison move
    when predicted under src_OQMD=1 (the majority training convention, 82% of labels)
    or src_MP=1 instead.

  CHECK 2 — novelty intersection.
    v3_05 removed only B==B' collisions. Intersect the remaining candidates with the
    full merged database (train/test/Sr2-holdout) by pymatgen-normalized formula.

  CHECK 3 — within-source label spread.
    v3_08's 0.458 eV noise floor collapses within-source duplicates to their MEDIAN
    before comparing across databases — but Stage 1's dedup keeps an essentially
    arbitrary row (unstable sort by source priority, keep="first"), so within-source
    polymorph/settings spread IS present in the training labels yet EXCLUDED from the
    floor estimate. Measure it, and measure |kept label − within-source median|.

Writes: logs/v3_09_review_checks.log (via launcher), tables/review_src_sensitivity.csv,
        tables/review_within_source_spread.csv
"""
import os, sys, pickle, traceback, re, itertools
sys.path.insert(0, "scripts")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "12")
import pandas as pd, numpy as np

def main():
    from pymatgen.core import Composition
    from utils import compute_bartel_tau, norm_formula
    from preproc_utils import featurize_df, FEAT_LABELS
    from joblib import Parallel, delayed

    with open("data/v2/imputer.pkl", "rb") as f: imputer = pickle.load(f)
    with open("data/v2/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open("data/v2/feature_names.pkl", "rb") as f: feat_names = pickle.load(f)
    with open("models/v2/mapie.pkl", "rb") as f: mapie = pickle.load(f)

    magpie_names = [f for f in feat_names if not f.startswith("src_")]
    src_names = [f for f in feat_names if f.startswith("src_")]
    magpie_idx = [FEAT_LABELS.index(f) for f in magpie_names if f in FEAT_LABELS]

    # ---------- candidate set, replicated exactly as v2_04 + v3_05 select it ----------
    screen_df = pd.read_parquet("data/processed/screening_set.parquet")
    cand = screen_df[~screen_df["in_prior_screen"]].copy()

    def split_bb(f):
        m = re.match(r"Sr2([A-Z][a-z]?)([A-Z][a-z]?)O6", str(f))
        return m.groups() if m else (None, None)
    bb = cand["formula"].apply(lambda f: pd.Series(split_bb(f), index=["B_chk", "Bp_chk"]))
    cand = cand.reset_index(drop=True)
    cand["B_chk"], cand["Bp_chk"] = bb["B_chk"].values, bb["Bp_chk"].values
    cand = cand[cand["B_chk"] != cand["Bp_chk"]].reset_index(drop=True)
    print(f"[setup] candidates after B==B' removal: {len(cand)}")

    raw_feats = np.array(featurize_df(cand, formula_col="formula", n_jobs=12))
    X_mag = imputer.transform(raw_feats[:, magpie_idx])

    DFT_FORMULAS = ["Sr2GaSbO6", "Sr2CrSbO6", "Sr2MnNbO6", "Sr2TeZnO6",
                    "Sr2MnWO6", "Sr2TaReO6", "Sr2VHfO6", "Sr2TiMnO6"]
    HSE = {"Sr2GaSbO6": 1.799, "Sr2CrSbO6": 0.580, "Sr2MnNbO6": 0.086,
           "Sr2TaReO6": 0.179, "Sr2VHfO6": 2.360, "Sr2TiMnO6": 2.470}
    dft_df = pd.DataFrame({"formula": DFT_FORMULAS})
    dft_raw = np.array(featurize_df(dft_df, formula_col="formula", n_jobs=8))
    Xd_mag = imputer.transform(dft_raw[:, magpie_idx])

    def with_src(Xm, active=None):
        src = np.zeros((Xm.shape[0], len(src_names)))
        if active is not None:
            src[:, src_names.index(f"src_{active}")] = 1.0
        return scaler.transform(np.hstack([Xm, src]))

    def predict(X):
        yp = mapie.predict(X)
        _, pis = mapie.predict_interval(X)
        return yp, pis[:, 0, 0], pis[:, 1, 0]

    PV_LO, PV_HI = 1.2, 1.8
    def tiers(yp, lo, hi):
        overlaps = (lo <= PV_HI) & (hi >= PV_LO)
        a1 = overlaps & (yp >= PV_LO) & (yp <= PV_HI)
        a2 = overlaps & ~a1
        return int(a1.sum()), int(a2.sum()), int((~overlaps).sum())

    print("\n================ CHECK 1 — src-flag sensitivity ================")
    variants = {"zeros": None, "OQMD": "OQMD", "MP": "MP"}
    cand_res, dft_res = {}, {}
    for tag, active in variants.items():
        yp, lo, hi = predict(with_src(X_mag, active))
        cand_res[tag] = (yp, lo, hi)
        ypd, lod, hid = predict(with_src(Xd_mag, active))
        dft_res[tag] = (ypd, lod, hid)
        a1, a2, b = tiers(yp, lo, hi)
        print(f"[cand] src={tag:5s}  pred mean {yp.mean():.4f}  min {yp.min():.4f}  "
              f"max {yp.max():.4f} | tiers A1={a1} A2={a2} B={b}")

    # sanity: zeros variant must reproduce the pipeline's stored predictions
    stored = pd.read_csv("screening/v2/new_ranked_candidates.csv")[["formula", "bg_pred"]]
    m = cand[["formula"]].assign(bg_new=cand_res["zeros"][0]).merge(stored, on="formula")
    dmax = (m.bg_new - m.bg_pred).abs().max()
    print(f"[sanity] max |repro - stored| on all-zeros predictions: {dmax:.2e} "
          f"({'OK' if dmax < 1e-4 else 'MISMATCH — investigate'})")

    base = cand_res["zeros"][0]
    rows = []
    for tag in ("OQMD", "MP"):
        d = cand_res[tag][0] - base
        print(f"[cand] Δpred (src_{tag} − zeros): mean {d.mean():+.4f}  "
              f"mean|Δ| {np.abs(d).mean():.4f}  p90|Δ| {np.quantile(np.abs(d), .9):.4f}  "
              f"max|Δ| {np.abs(d).max():.4f} eV")
        bt = tiers(*cand_res["zeros"]); vt = tiers(*cand_res[tag])
        # per-candidate tier flips
        def tier_of(yp, lo, hi):
            overlaps = (lo <= PV_HI) & (hi >= PV_LO)
            return np.where(~overlaps, "B", np.where((yp >= PV_LO) & (yp <= PV_HI), "A1", "A2"))
        flips = (tier_of(*cand_res["zeros"]) != tier_of(*cand_res[tag])).sum()
        print(f"[cand] tier counts zeros(A1,A2,B)={bt} -> src_{tag}={vt}; "
              f"candidates changing tier: {flips}/{len(cand)}")
    for i, f in enumerate(DFT_FORMULAS):
        r = {"formula": f, "HSE06": HSE.get(f)}
        for tag in variants:
            yp, lo, hi = dft_res[tag]
            r[f"pred_{tag}"] = round(float(yp[i]), 4)
            r[f"lo_{tag}"] = round(float(lo[i]), 4)
            r[f"hi_{tag}"] = round(float(hi[i]), 4)
            if f in HSE:
                r[f"inCI_{tag}"] = bool(lo[i] <= HSE[f] <= hi[i])
        rows.append(r)
    dft_tab = pd.DataFrame(rows)
    print("\n[DFT] predictions under each src convention:")
    print(dft_tab.to_string(index=False))
    for tag in variants:
        cov = sum(1 for r in rows if r.get(f"inCI_{tag}") is True)
        print(f"[DFT] HSE06-in-CI coverage under src={tag}: {cov}/6")
    dft_tab.to_csv("tables/review_src_sensitivity.csv", index=False)

    print("\n================ CHECK 2 — novelty intersection ================")
    cand["formula_norm"] = [norm_formula(f) for f in cand["formula"]]
    full = pd.read_parquet("data/v2/full_A2BBO6.parquet")[["formula_norm", "band_gap", "source"]]
    splits = {}
    for s in ("train", "test", "sr2_holdout"):
        splits[s] = set(pd.read_parquet(f"data/v2/{s}.parquet")["formula_norm"])
    hit = cand.merge(full, on="formula_norm", how="inner")
    print(f"candidates colliding with the merged database: {len(hit)}/{len(cand)}")
    if len(hit):
        yp0 = pd.Series(base, index=cand.index)
        pred_map = dict(zip(cand["formula_norm"], base))
        for _, r in hit.iterrows():
            member = [s for s in splits if r["formula_norm"] in splits[s]]
            print(f"  {r['formula_norm']:14s} db_gap={r['band_gap']:.3f} ({r['source']}) "
                  f"in={','.join(member) or '?'}  ML_pred={pred_map.get(r['formula_norm'], float('nan')):.3f}")

    print("\n================ CHECK 3 — within-source label spread ================")
    SOURCES = {k: f"data/v2_sources/{k}.parquet" for k in
               ("mp_all", "oqmd_all", "jarvis_all", "cmr_all", "aflow_all", "optimade")}
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
    raw = raw[raw["source"] != "CMR"]

    def is_A2BBO6(formula):
        try:
            comp = Composition(formula).reduced_composition
            if not (3 <= len(list(comp.elements)) <= 5):
                return False
            o = comp["O"]
            cat = sum(v for k, v in comp.items() if str(k) != "O")
            return abs(cat / o - (4 / 6)) < 0.05
        except Exception:
            return False

    raw = raw[raw["formula_pretty"].apply(is_A2BBO6)]
    raw = raw[raw["is_metal"].fillna(False) == False]
    raw = raw[raw["band_gap"].fillna(0) >= 0.05]
    taus = Parallel(n_jobs=12)(delayed(compute_bartel_tau)(f) for f in raw["formula_pretty"])
    raw = raw[np.array(taus) <= 6.0]
    raw = raw[raw["band_gap"].notna()].copy()
    raw["formula_norm"] = raw["formula_pretty"].apply(norm_formula)
    print(f"pre-dedup rows after Stage-1 filters: {len(raw):,}")

    g = raw.groupby(["formula_norm", "source"])["band_gap"]
    ws = g.agg(n="count", med="median", lo="min", hi="max", sd="std").reset_index()
    ws["range"] = ws["hi"] - ws["lo"]
    multi = ws[ws.n >= 2]
    print(f"(composition, source) groups: {len(ws):,}; with >=2 entries: {len(multi):,}")
    for src, gg in multi.groupby("source"):
        print(f"  within-{src:9s} n_groups={len(gg):4d}  mean range {gg['range'].mean():.4f}  "
              f"median {gg['range'].median():.4f}  p90 {gg['range'].quantile(.9):.4f} eV  "
              f"(mean entries/comp {gg.n.mean():.1f})")
    print(f"  within-source ALL  mean range {multi['range'].mean():.4f}  "
          f"median {multi['range'].median():.4f}  p90 {multi['range'].quantile(.9):.4f} eV")
    ws.to_csv("tables/review_within_source_spread.csv", index=False)

    # the labels the model actually trains on vs the within-source median
    kept = full.rename(columns={"band_gap": "kept_gap"})
    j = kept.merge(ws, on=["formula_norm", "source"], how="left")
    jm = j[j.n >= 2].copy()
    jm["dev"] = (jm.kept_gap - jm.med).abs()
    n_tot = len(kept)
    print(f"\nfinal labels whose winning source has >=2 entries for that composition: "
          f"{len(jm):,}/{n_tot:,} ({len(jm)/n_tot:.1%})")
    print(f"|kept label − within-source median|: mean {jm.dev.mean():.4f}  "
          f"median {jm.dev.median():.4f}  p90 {jm.dev.quantile(.9):.4f}  max {jm.dev.max():.4f} eV")
    for thr in (0.1, 0.25, 0.5):
        print(f"  fraction of those deviating > {thr} eV: {(jm.dev > thr).mean():.1%} "
              f"({int((jm.dev > thr).sum())} labels)")
    print(f"as a fraction of ALL {n_tot:,} labels: >0.25 eV arbitrary-selection deviation on "
          f"{(jm.dev > 0.25).sum() / n_tot:.1%} of the training+test+holdout set")

    ehc = raw.groupby("source")["energy_above_hull"].apply(lambda s: s.notna().mean())
    print("\nenergy_above_hull availability (pre-dedup rows), for a ground-state-dedup fix:")
    for src, v in ehc.items():
        print(f"  {src:9s} {v:.1%}")

try:
    main()
except Exception:
    traceback.print_exc()
print("REVIEW_CHECKS_DONE")
