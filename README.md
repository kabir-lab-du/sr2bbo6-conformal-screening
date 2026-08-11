# Conformal-prediction band-gap screening of Sr₂BB′O₆ double perovskites

Code, data, and DFT inputs for:

> Md. Mohiuddin, Alamgir Kabir\*, Jannatul Ferdousi,
> *Conformal prediction sets honest limits on composition-based band-gap
> screening of Sr₂BB′O₆ double perovskite oxides* (submitted, 2026).
> \*Corresponding author: alamgir.kabir@du.ac.bd

The paper trains a stacked gradient-boosted ensemble (CatBoost / XGBoost /
LightGBM → ridge meta-learner) on 6,295 ground-state band gaps merged from
four DFT databases, wraps it in CV+ conformal prediction with class-conditional
(Mondrian) recalibration, and asks whether the calibrated intervals are narrow
enough to *select* photovoltaic candidates — they are not, and the paper
quantifies why (label-fidelity, not regression quality).

## Layout

```
scripts/    Full pipeline, in execution order:
            v4_01_merge_split_gs.py   merge sources, ground-state dedup, Sr2 holdout
            v4_02a_featurize.py       Magpie + provenance features, VIF pruning (55)
            v4_02b_train_fixed.py     Optuna-tuned GBDTs + stacking + CV+ conformal
            v4_03_screen_tier.py      candidate screening and tiering
            v4_04_mondrian.py         class-conditional (Mondrian) recalibration
            v4_05_loco.py             grouped (leave-one-cluster-out) validation
            v4_06_oqmd_only.py        single-convention ablation
            v4_07_export_pub_data.py  publication data export
            v3_05..v3_09_*.py         tiering, DFT comparison, figures,
                                      label-noise floor, review checks
            v2_*.py                   earlier-generation pipeline kept for provenance
data/       Source pulls (parquet), processed matrices, feature metadata (~15 MB)
results/    Per-sample predictions with conformal intervals, metrics, LOCO folds,
            Mondrian summaries, tiered candidate lists, feature importances
tables/     Label-noise floor analysis (incl. the 1,663-row pairwise table),
            DFT-vs-ML comparison
dft/        VASP input files (INCAR, KPOINTS, POSCAR, chain/run scripts) for the
            in-house HSE06 anchor calculations (see note below)
figures/    Scripts regenerating every figure in the paper and SI
```

## Reproducing

```bash
conda create -n sr2bbo6 python=3.10
conda activate sr2bbo6
pip install -r requirements.txt
cd scripts && python v4_01_merge_split_gs.py && python v4_02a_featurize.py && ...
```

Two practical notes:

- **Thread counts matter.** On a shared many-core machine, keep GBDT thread
  counts at ~8 (`n_jobs`/`thread_count`); with only ~5k training rows, 96
  OpenMP threads is orders of magnitude *slower* (benchmark in the SI).
- **Trained model binaries** (~551 MB) exceed GitHub limits and are not in this
  repository; all models are exactly reproducible from `scripts/` + `data/`
  (seeds fixed). A versioned archive with the binaries will be deposited with a
  DOI upon acceptance of the article.

## DFT inputs

`dft/` contains complete VASP 6.4.3 inputs for the HSE06 anchor calculations
(PBE(+U) relaxation with FM/AFM competition where open-shell, then HSE06 static
on the winning order). **POTCAR files are not redistributable** under the VASP
licence; the potentials used are PAW-PBE `Sr_sv, Ga_d, Sb, Mn_pv, W_sv, Mo_sv,
Sn_d, Al, In_d, O` (standard VASP 5.4 recommended set).

## Data provenance and licences

The merged dataset draws on the Materials Project (CC-BY 4.0), OQMD, AFLOW,
and JARVIS-DFT, retrieved partly via OPTIMADE federation; per-row provenance is
retained in the `src` columns. Please cite the original databases when reusing
the merged data. Code in this repository is MIT-licensed (see `LICENSE`).
