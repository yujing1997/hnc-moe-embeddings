# HNC MoE Embeddings — Downstream Survival Analysis

Code accompanying the manuscript comparing structured LLM-extracted clinical
features (Model A) against free-text embedding models (B1–B6, generated from
pathology and consultation reports) for survival prediction (OS, DFS, LFS,
DistFS) in a head-and-neck cancer (HNC) cohort.

## Contents

- `src/01_generate_embeddings.py` — generate dual (pathology + consultation)
  embeddings for each embedding model.
- `src/02_survival_comparison_mice.py` — main survival analysis pipeline:
  Cox proportional hazards models, MICE (M=5) multiple imputation, Harrell
  bootstrap optimism correction (B=1000), likelihood-ratio tests, Meng–Rubin
  D2 pooling.
- `src/analysis_pca_k_sensitivity.py` — sensitivity analysis over PCA
  component count (k).
- `src/analysis_time_dependent_discrimination.py` — time-dependent AUC /
  discrimination analysis.
- `src/analysis_pid_multimodal.py` — Partial Information Decomposition (PID)
  of pathology vs. consultation (vs. LLM-clinical) modalities.
- `src/analysis_interpretability_associations.py` — interpretability /
  feature-association analyses.
- `src/plotting/` — figure generation scripts (KM curves, forest plots, PCA
  variance, model comparison heatmaps, UMAP, embedding similarity, etc.).

## Data availability

Patient-level embeddings, raw clinical data, and generated output figures are
not included in this repository due to size and privacy considerations.
Scripts expect the following (not tracked) directory layout at the repo root:

```
embeddings/<model>/<model>_dual_embeddings.pkl
outputs_all_fields/all_fields_processed.csv
```

## Environment

See individual scripts for required packages (pandas, numpy, scikit-learn,
lifelines, matplotlib, imml).
