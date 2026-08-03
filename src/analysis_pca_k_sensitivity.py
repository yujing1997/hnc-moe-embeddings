#!/usr/bin/env python3
"""
Supplementary analysis: sensitivity to PCA dimensionality k
------------------------------------------------------------
Evaluates Cox-model discrimination as k_total (PCA components) is varied
from 3 to 15 using the same 1:2 pathology:consultation allocation as the
main analysis (k_path : k_cons pairs: 1:2, 2:4, 3:6, 4:8, 5:10).

All four pre-specified endpoints are evaluated (DFS, LFS, OS, DistFS).

Rigour matches the main MICE analysis (02_survival_comparison_mice.py):
  * MICE M=5 (sample_posterior=True) for full-cohort apparent C,
    pooled via logit-scale Rubin's rules (Marshall et al. 2009)
  * Harrell bootstrap B=1000: re-fit PCA on boot sample, apply to
    original cohort; pool M optimism estimates per iteration via mean
  * Corrected C = apparent C - mean bootstrap optimism
  * 95% CI = percentile([apparent - optimism_b for b=1..B], [2.5,97.5])

Outputs (src/plotting/outputs/supplementary/):
  pca_k_sensitivity_{outcome}.csv          (per outcome)
  pca_k_sensitivity.csv                    (combined)
  pca_k_sensitivity_combined.png/.pdf      (2x2 panel, all outcomes)
  pca_k_sensitivity_{outcome}.png/.pdf     (per-outcome panels)

Usage:
  python src/analysis_pca_k_sensitivity.py
  python src/analysis_pca_k_sensitivity.py --replot-only
  python src/analysis_pca_k_sensitivity.py --n-boot 200   # fast test
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ========================== Paths ==========================

BASE          = Path('/Data/Yujing/HNC_OutcomePred/Reports_Agents/Revision1/'
                     'Downstream_Analysis/moe_embeddings_comp')
OUTCOMES_PATH = Path('/Data/Yujing/HNC_OutcomePred/Reports_Agents/Revision1/'
                     'Downstream_Analysis/outputs_all_fields/all_fields_processed.csv')
EMB_DIR       = BASE / 'embeddings'
OUTDIR        = BASE / 'src' / 'plotting' / 'outputs' / 'supplementary'
OUTDIR.mkdir(parents=True, exist_ok=True)
MAIN_RESULTS_PATH = (BASE / 'results_v2_permodality' / 'full_mice'
                     / 'full_mice_m5_b1000' / 'embedding_comparison_results.csv')

# ========================== Hyper-parameters ==========================

M_MICE        = 5        # proper multiple imputation datasets
MICE_MAX_ITER = 10       # chained-equation rounds per dataset
MICE_SEED     = 42       # base seed; actual seed = MICE_SEED + m
N_BOOT        = 1000     # Harrell optimism bootstrap iterations
PENALIZER     = 0.1      # Ridge penaliser for CoxPHFitter

# 1:2 (pathology:consultation) ratio, mirroring the main analysis k_path=1, k_cons=2.
# Each pair doubles the components while preserving the ratio.
K_PAIRS = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]   # (k_path, k_cons)

# ========================== Feature definitions ==========================

BASELINE_FEATURES = [
    'Age_Numeric', 'Sex_Binary',
    'T_Stage_Numeric', 'N_Stage_Numeric',
    'has_pathology', 'has_consultation',
]
MICE_IMPUTE_COLS = [
    'Age_Numeric', 'Sex_Binary',
    'T_Stage_Numeric', 'N_Stage_Numeric',
]

# ========================== Arms ==========================

EMBEDDING_ARMS = {
    'B1: Longformer':   'longformer/longformer_dual_embeddings.pkl',
    'B2: Nomic':        'nomic/nomic_dual_embeddings.pkl',
    'B3: Llama':        'llama/llama_dual_embeddings.pkl',
    'B4: SMB-1.7B':     'smb/smb_dual_embeddings.pkl',
    'B5: SMB-8B-Qwen':  'smb8b_qwen3_mo/smb8b_qwen3_mo_dual_embeddings.pkl',
    'B6: SMB-8B-Llama': 'smb8b_llama3_cl/smb8b_llama3_cl_dual_embeddings.pkl',
}

BASELINE_COLOR = '#6c757d'
ARM_COLORS = {
    'Baseline':         '#6c757d',
    'Model A':          '#d62828',
    'B1: Longformer':   '#2a9d8f',
    'B2: Nomic':        '#e76f51',
    'B3: Llama':        '#e9c46a',
    'B4: SMB-1.7B':     '#264653',
    'B5: SMB-8B-Qwen':  '#7209b7',
    'B6: SMB-8B-Llama': '#3a0ca3',
}

DISPLAY_LABELS = {
    'Baseline':         'Baseline',
    'Model A':          'Model A: LLM-Extr',
    'B1: Longformer':   'B1: Longformer',
    'B2: Nomic':        'B2: Nomic',
    'B3: Llama':        'B3: Llama 3.3 70B',
    'B4: SMB-1.7B':     'B4: SMB-1.7B',
    'B5: SMB-8B-Qwen':  'B5: SMB-8B-Qwen',
    'B6: SMB-8B-Llama': 'B6: SMB-8B-Llama',
}

OUTCOME_SPECS = {
    'DFS':    {'time_col': 'DFS',    'event_col': 'Failure',              'label': 'Disease-Free Survival'},
    'LFS':    {'time_col': 'LFS',    'event_col': 'Locoregional_Failure', 'label': 'Locoregional FS'},
    'OS':     {'time_col': 'OS',     'event_col': 'Death',                'label': 'Overall Survival'},
    'DistFS': {'time_col': 'DistFS', 'event_col': 'distant failure',      'label': 'Distant FS'},
}

# ========================== Style ==========================

_available  = [f.name for f in fm.fontManager.ttflist]
FONT_FAMILY = 'Times New Roman' if 'Times New Roman' in _available else 'DejaVu Serif'

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        [FONT_FAMILY, 'DejaVu Serif'],
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9,
    'savefig.dpi':       400,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'figure.facecolor':  'white',
})

# ========================== Data loading ==========================

def load_outcomes() -> pd.DataFrame:
    df = pd.read_csv(OUTCOMES_PATH)
    df['patient_id'] = df['patient_id'].astype(str)
    df['Age_Numeric'] = pd.to_numeric(df.get('Age', np.nan), errors='coerce')
    df['Sex_Binary'] = (df.get('sex', pd.Series(dtype=str)).astype(str)
                         .apply(lambda x: 1.0 if x.lower().startswith('m')
                                else (0.0 if x.lower().startswith('f') else np.nan)))
    df['T_Stage_Numeric'] = (df.get('T stage', pd.Series(dtype=str)).astype(str)
                              .apply(lambda x: int(x[1]) if isinstance(x, str)
                                     and x.startswith('T') and len(x) > 1
                                     and x[1].isdigit() else np.nan))
    df['N_Stage_Numeric'] = (df.get('N stage', pd.Series(dtype=str)).astype(str)
                              .apply(lambda x: int(x[1]) if isinstance(x, str)
                                     and x.startswith('N') and len(x) > 1
                                     and x[1].isdigit() else np.nan))
    return df


def load_embeddings() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for label, rel in EMBEDDING_ARMS.items():
        p = EMB_DIR / rel
        if not p.exists():
            print(f'  [WARN] Not found: {p}. Arm {label} skipped.')
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        out[label] = {str(k): v for k, v in d.items()}
    return out

# ========================== MICE helpers ==========================

def mice_impute_multiple(
    df: pd.DataFrame,
    impute_cols: list[str],
    M: int = M_MICE,
    random_state: int = MICE_SEED,
) -> list[pd.DataFrame]:
    """
    Return M stochastic imputed copies (sample_posterior=True, proper MI).
    Mirrors mice_impute_multiple() in 02_survival_comparison_mice.py.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    safe_impute  = [c for c in impute_cols
                    if c in numeric_cols and c not in ('Time', 'Event')]
    n_missing = df[safe_impute].isna().sum().sum() if safe_impute else 0
    if n_missing == 0:
        return [df.copy() for _ in range(M)]

    datasets = []
    for m in range(M):
        imp    = IterativeImputer(max_iter=MICE_MAX_ITER,
                                  random_state=random_state + m,
                                  sample_posterior=True)
        result = df.copy()
        arr    = imp.fit_transform(result[numeric_cols])
        idf    = pd.DataFrame(arr, columns=numeric_cols, index=result.index)
        for col in safe_impute:
            result[col] = idf[col]
        for col in ['Sex_Binary', 'T_Stage_Numeric', 'N_Stage_Numeric']:
            if col in safe_impute:
                result[col] = result[col].round().clip(lower=0)
        datasets.append(result)
    return datasets


def _logit(p: float) -> float:
    return float(np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6))))

def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def rubin_pool_c(c_vals: list[float]) -> float:
    """Pool C-indices on logit scale (Marshall et al. 2009), back-transform."""
    valid = [v for v in c_vals if np.isfinite(v)]
    return _expit(np.mean([_logit(v) for v in valid])) if valid else np.nan

# ========================== PCA helpers ==========================

def fit_pca_on_mat(mat: np.ndarray, k_path: int, k_cons: int):
    """Fit per-modality PCA; return (pca_path, pca_cons, pcs)."""
    half = mat.shape[1] // 2
    kp   = min(k_path, mat[:, :half].shape[1], mat.shape[0] - 1)
    kc   = min(k_cons, mat[:, half:].shape[1], mat.shape[0] - 1)
    pp   = PCA(n_components=kp, random_state=42).fit(mat[:, :half])
    pc   = PCA(n_components=kc, random_state=42).fit(mat[:, half:])
    pcs  = np.concatenate([pp.transform(mat[:, :half]),
                           pc.transform(mat[:, half:])], axis=1)
    return pp, pc, pcs, kp, kc

def apply_pca(mat: np.ndarray, pca_path, pca_cons) -> np.ndarray:
    half = mat.shape[1] // 2
    return np.concatenate([pca_path.transform(mat[:, :half]),
                           pca_cons.transform(mat[:, half:])], axis=1)

def pcs_to_df(pcs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])

# ========================== Cox eval ==========================

def fit_eval_c(df: pd.DataFrame, feat_cols: list[str]) -> tuple[float, float]:
    """Fit Cox on df; return (C-index, AIC_partial)."""
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df[feat_cols + ['Time', 'Event']], duration_col='Time', event_col='Event')
    risk = cph.predict_partial_hazard(df[feat_cols]).values
    cidx = float(concordance_index(df['Time'], -risk, df['Event']))
    return cidx, float(cph.AIC_partial_)

# ========================== Core: optimism bootstrap for PCA k ==========================

def bootstrap_pca_k(
    imputed_list: list[pd.DataFrame],   # M imputed dfs with Time + Event
    mat: np.ndarray,                     # embedding matrix (N x D)
    base_feat_cols: list[str],           # BASELINE_FEATURES in df
    k_path: int,
    k_cons: int,
    n_boot: int,
    random_state: int = 42,
) -> tuple[float, float, float, float, int, int]:
    """
    Optimism-corrected C for embedding arm at (k_path, k_cons).

    Returns: (apparent_c, corrected_c, ci_lower, ci_upper, k_path_used, k_cons_used)

    Protocol (mirrors main analysis Val-MI):
      Apparent C: M-pooled Rubin's rules on logit scale.
      Bootstrap (b=1..B):
        - Draw boot_idx; re-fit PCA on mat[boot_idx]
        - For each m: boot_df = imputed_m.iloc[boot_idx] + boot PCs
                      orig_df = imputed_m         + orig PCs
          fit Cox on boot_df -> C_train_m, C_test_m -> optimism_m
        - optimism_b = mean(optimism_m for m=1..M)
      Corrected C = apparent - mean(optimism_b)
      95% CI = percentile([apparent - optimism_b for b], [2.5, 97.5])
    """
    M   = len(imputed_list)
    n   = len(imputed_list[0])
    rng = np.random.default_rng(random_state)

    # ── Apparent C: full cohort, M imputations ────────────────────────────────
    pp_full, pc_full, pcs_full, kp_used, kc_used = fit_pca_on_mat(mat, k_path, k_cons)
    pcs_df_full = pcs_to_df(pcs_full)
    feat_cols   = base_feat_cols + list(pcs_df_full.columns)

    app_c_per_m = []
    for use_m in imputed_list:
        dfm = pd.concat([use_m.reset_index(drop=True),
                         pcs_df_full.reset_index(drop=True)], axis=1)
        try:
            c, _ = fit_eval_c(dfm, feat_cols)
            app_c_per_m.append(c)
        except Exception:
            continue
    apparent_c = rubin_pool_c(app_c_per_m)

    # ── Bootstrap optimism ────────────────────────────────────────────────────
    boot_opts: list[float] = []

    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        boot_mat = mat[boot_idx]

        # Re-fit PCA on bootstrap sample
        pp_b, pc_b, boot_pcs, _, _ = fit_pca_on_mat(boot_mat, k_path, k_cons)
        orig_pcs  = apply_pca(mat, pp_b, pc_b)
        boot_pcs_df = pcs_to_df(boot_pcs)
        orig_pcs_df = pcs_to_df(orig_pcs)
        boot_feat_cols = base_feat_cols + list(boot_pcs_df.columns)

        m_opts: list[float] = []
        for m_idx in range(M):
            use_m = imputed_list[m_idx].reset_index(drop=True)
            boot_df = pd.concat([use_m.iloc[boot_idx].reset_index(drop=True),
                                 boot_pcs_df.reset_index(drop=True)], axis=1)
            orig_df = pd.concat([use_m,
                                 orig_pcs_df.reset_index(drop=True)], axis=1)
            try:
                cph_b = CoxPHFitter(penalizer=PENALIZER)
                cph_b.fit(boot_df[boot_feat_cols + ['Time', 'Event']],
                          duration_col='Time', event_col='Event')
                c_train = concordance_index(
                    boot_df['Time'],
                    -cph_b.predict_partial_hazard(boot_df[boot_feat_cols]).values,
                    boot_df['Event'])
                c_test  = concordance_index(
                    orig_df['Time'],
                    -cph_b.predict_partial_hazard(orig_df[boot_feat_cols]).values,
                    orig_df['Event'])
                m_opts.append(float(c_train - c_test))
            except Exception:
                continue
        if m_opts:
            boot_opts.append(float(np.mean(m_opts)))

    if boot_opts:
        mean_opt   = float(np.mean(boot_opts))
        corrected  = float(np.clip(apparent_c - mean_opt, 0.3, 1.0))
        boot_corr  = [apparent_c - o for o in boot_opts]
        ci_lo      = float(np.percentile(boot_corr, 2.5))
        ci_hi      = float(np.percentile(boot_corr, 97.5))
    else:
        corrected = apparent_c
        ci_lo = ci_hi = np.nan

    return apparent_c, corrected, ci_lo, ci_hi, kp_used, kc_used


def bootstrap_baseline_c(
    imputed_list: list[pd.DataFrame],
    base_feat_cols: list[str],
    n_boot: int,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    """Optimism-corrected C for Baseline (no PCA). Same MI+bootstrap protocol."""
    M   = len(imputed_list)
    n   = len(imputed_list[0])
    rng = np.random.default_rng(random_state)

    # Apparent C
    app_c_m = []
    for use_m in imputed_list:
        try:
            c, _ = fit_eval_c(use_m, base_feat_cols)
            app_c_m.append(c)
        except Exception:
            continue
    apparent_c = rubin_pool_c(app_c_m)

    # Bootstrap
    boot_opts: list[float] = []
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        m_opts: list[float] = []
        for m_idx in range(M):
            use_m   = imputed_list[m_idx].reset_index(drop=True)
            boot_df = use_m.iloc[boot_idx].reset_index(drop=True)
            try:
                cph_b = CoxPHFitter(penalizer=PENALIZER)
                cph_b.fit(boot_df[base_feat_cols + ['Time', 'Event']],
                          duration_col='Time', event_col='Event')
                c_train = concordance_index(
                    boot_df['Time'],
                    -cph_b.predict_partial_hazard(boot_df[base_feat_cols]).values,
                    boot_df['Event'])
                c_test  = concordance_index(
                    use_m['Time'],
                    -cph_b.predict_partial_hazard(use_m[base_feat_cols]).values,
                    use_m['Event'])
                m_opts.append(float(c_train - c_test))
            except Exception:
                continue
        if m_opts:
            boot_opts.append(float(np.mean(m_opts)))

    if boot_opts:
        mean_opt  = float(np.mean(boot_opts))
        corrected = float(np.clip(apparent_c - mean_opt, 0.3, 1.0))
        bc        = [apparent_c - o for o in boot_opts]
        ci_lo     = float(np.percentile(bc, 2.5))
        ci_hi     = float(np.percentile(bc, 97.5))
    else:
        corrected = apparent_c
        ci_lo = ci_hi = np.nan
    return apparent_c, corrected, ci_lo, ci_hi

# ========================== Run one outcome ==========================

def run_outcome(
    df: pd.DataFrame,
    emb: dict,
    outcome_label: str,
    time_col: str,
    event_col: str,
    n_boot: int,
) -> pd.DataFrame | None:
    print(f'\n{"="*60}')
    print(f'Outcome: {outcome_label}  ({time_col} / {event_col})')
    print(f'{"="*60}')

    if time_col not in df.columns or event_col not in df.columns:
        print(f'  Skipping — columns not found.')
        return None

    # Build working dataframe
    needed  = ['patient_id', time_col, event_col] + BASELINE_FEATURES[:4]
    avail   = [c for c in needed if c in df.columns]
    use     = df[avail].copy()
    use     = use.rename(columns={time_col: 'Time', event_col: 'Event'})
    use['Event'] = pd.to_numeric(use['Event'], errors='coerce')
    use     = use[(use['Time'] > 0) & (use['Time'] <= 240)
                  & (use['Event'].isin([0, 1]))].copy().reset_index(drop=True)
    use['Event'] = use['Event'].astype(int)

    # Embedding-presence indicators
    first_emb = next(iter(emb.values()), {})
    hp, hc = [], []
    for pid in use['patient_id'].astype(str):
        vec = first_emb.get(pid)
        if vec is None:
            hp.append(0); hc.append(0); continue
        hh = vec.shape[0] // 2
        hp.append(int(np.any(np.abs(vec[:hh]) > 1e-12)))
        hc.append(int(np.any(np.abs(vec[hh:]) > 1e-12)))
    use['has_pathology']    = hp
    use['has_consultation'] = hc

    # Restrict to common patients across all arms
    keep = set(use['patient_id'].astype(str))
    for d in emb.values():
        keep &= set(d.keys())
    use = use[use['patient_id'].astype(str).isin(keep)].copy().reset_index(drop=True)

    if len(use) < 50 or use['Event'].sum() < 10:
        print(f'  Skipping — insufficient patients or events (N={len(use)}).')
        return None
    print(f'  Cohort: N={len(use)}, events={use["Event"].sum()}')

    # MICE M=5
    impute_cols  = [c for c in MICE_IMPUTE_COLS if c in use.columns]
    imputed_list = mice_impute_multiple(use, impute_cols)
    print(f'  MICE M={M_MICE} complete.')

    base_feat_cols = [f for f in BASELINE_FEATURES if f in use.columns]
    pids = use['patient_id'].astype(str).tolist()

    rows: list[dict] = []

    # ── Baseline (no embedding, no PCA) ─────────────────────────────────────
    print('  Computing Baseline (no PCA) ...')
    app_c, corr_c, ci_lo, ci_hi = bootstrap_baseline_c(
        imputed_list, base_feat_cols, n_boot)
    # Get apparent AIC from first imputed dataset
    try:
        _, aic_base = fit_eval_c(imputed_list[0], base_feat_cols)
    except Exception:
        aic_base = np.nan
    rows.append({
        'Outcome': outcome_label, 'Model': 'Baseline',
        'k_total': 0, 'k_path': 0, 'k_cons': 0,
        'C_apparent': round(app_c, 5),  'C_corrected': round(corr_c, 5),
        'CI_lower':   round(ci_lo, 5) if np.isfinite(ci_lo) else np.nan,
        'CI_upper':   round(ci_hi, 5) if np.isfinite(ci_hi) else np.nan,
        'AIC': round(aic_base, 2) if np.isfinite(aic_base) else np.nan,
        'N': len(use), 'N_events': int(use['Event'].sum()),
    })
    print(f'    apparent={app_c:.4f}  corrected={corr_c:.4f}  '
          f'95%CI [{ci_lo:.3f},{ci_hi:.3f}]')

    # ── Embedding arms ──────────────────────────────────────────────────────
    for arm_label, d in emb.items():
        try:
            mat = np.stack([d[pid] for pid in pids])
        except KeyError as e:
            print(f'  [WARN] Missing pid {e} for {arm_label}. Skipping.')
            continue

        print(f'\n  -- {arm_label} --')
        for k_path, k_cons in K_PAIRS:
            k_total = k_path + k_cons
            print(f'    k=({k_path},{k_cons})  total={k_total} ...', end='', flush=True)

            app_c, corr_c, ci_lo, ci_hi, kp_used, kc_used = bootstrap_pca_k(
                imputed_list, mat, base_feat_cols, k_path, k_cons, n_boot)

            # AIC from apparent fit (first imputed dataset)
            try:
                _, _, pcs_full, _, _ = fit_pca_on_mat(mat, k_path, k_cons)
                pcs_df = pcs_to_df(pcs_full)
                dfm    = pd.concat([imputed_list[0].reset_index(drop=True),
                                    pcs_df.reset_index(drop=True)], axis=1)
                _, aic = fit_eval_c(dfm, base_feat_cols + list(pcs_df.columns))
            except Exception:
                aic = np.nan

            rows.append({
                'Outcome': outcome_label, 'Model': arm_label,
                'k_total': int(k_total), 'k_path': int(kp_used), 'k_cons': int(kc_used),
                'C_apparent': round(app_c, 5), 'C_corrected': round(corr_c, 5),
                'CI_lower': round(ci_lo, 5) if np.isfinite(ci_lo) else np.nan,
                'CI_upper': round(ci_hi, 5) if np.isfinite(ci_hi) else np.nan,
                'AIC': round(aic, 2) if np.isfinite(aic) else np.nan,
                'N': len(use), 'N_events': int(use['Event'].sum()),
            })
            print(f'  apparent={app_c:.4f}  corrected={corr_c:.4f}  '
                  f'95%CI [{ci_lo:.3f},{ci_hi:.3f}]')

    out = pd.DataFrame(rows)
    csv_path = OUTDIR / f'pca_k_sensitivity_{outcome_label}.csv'
    out.to_csv(csv_path, index=False)
    print(f'\n  Saved: {csv_path}')

    # Per-outcome plot (apparent vs corrected, side by side)
    _plot_single_outcome(out, outcome_label)
    return out

# ========================== Plotting: single outcome ==========================

def _plot_single_outcome(out: pd.DataFrame, outcome_label: str,
                         ci_style: str = 'band') -> None:
    """Side-by-side apparent vs corrected C-index for one outcome."""
    outdir = OUTDIR if ci_style == 'band' else OUTDIR / 'ci_bars' / 'pca_k_sensitivity'
    outdir.mkdir(parents=True, exist_ok=True)
    model_list   = [m for m in out['Model'].unique() if m != 'Baseline']
    base_corr    = out.loc[out['Model'] == 'Baseline', 'C_corrected'].values
    base_corr_v  = base_corr[0] if len(base_corr) else np.nan
    base_app     = out.loc[out['Model'] == 'Baseline', 'C_apparent'].values
    base_app_v   = base_app[0]  if len(base_app)  else np.nan

    fig, axes    = plt.subplots(1, 2, figsize=(14, 5.5))
    panel_cols   = ['C_apparent', 'C_corrected']
    panel_titles = ['Apparent C-index', f'Optimism-Corrected C-index (B={N_BOOT})']
    base_vals    = [base_app_v, base_corr_v]

    for ax, col, title, bv in zip(axes, panel_cols, panel_titles, base_vals):
        for arm in model_list:
            sub = out[out['Model'] == arm].sort_values('k_total')
            if sub.empty:
                continue
            ax.plot(sub['k_total'], sub[col],
                    marker='o', linewidth=2.0, markersize=5,
                    color=ARM_COLORS.get(arm, '#888'),
                    label=DISPLAY_LABELS.get(arm, arm))
            if col == 'C_corrected':
                ci_lo = sub['CI_lower'].values
                ci_hi = sub['CI_upper'].values
                valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
                if valid.any() and ci_style == 'band':
                    ax.fill_between(sub['k_total'].values[valid],
                                    ci_lo[valid], ci_hi[valid],
                                    color=ARM_COLORS.get(arm, '#888'),
                                    alpha=0.08, linewidth=0)
                elif valid.any() and ci_style == 'bars':
                    xs = sub['k_total'].values[valid]
                    ys = sub[col].values[valid]
                    ax.errorbar(xs, ys,
                                yerr=[ys - ci_lo[valid], ci_hi[valid] - ys],
                                fmt='none', ecolor=ARM_COLORS.get(arm, '#888'),
                                elinewidth=1.0, capsize=2.5, alpha=0.65,
                                zorder=2)

        if np.isfinite(bv):
            ax.axhline(bv, color=BASELINE_COLOR, ls='--', lw=1.5, label='Baseline')
        # Vertical reference at k=3 (main analysis)
        ax.axvline(3, color='#aaaaaa', ls=':', lw=1.2)
        ax.text(3.1, ax.get_ylim()[0] + 0.002, 'Main\nanalysis\nk=3',
                fontsize=7.5, color='#888', va='bottom')

        ax.set_xlabel('Total PCA components (k_path + k_cons)', fontsize=11)
        ax.set_ylabel(col.replace('_', ' '), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([k_path + k_cons for k_path, k_cons in K_PAIRS])
        ax.legend(loc='best', fontsize=8, frameon=True, framealpha=0.9)
        ax.grid(alpha=0.15)

    fig.suptitle(outcome_label, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = outdir / f'pca_k_sensitivity_{outcome_label}.{ext}'
        fig.savefig(p, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outdir}/pca_k_sensitivity_{outcome_label}.png/pdf')


def add_model_a_k3_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Append Model A at k=3 from the main full-MICE results CSV.

    Model A is structured LLM extraction, so it has no PCA-k curve. For visual
    comparison in the PCA-k sensitivity figure, add its already-computed main
    analysis result as a single point at k=3 (the main embedding analysis k).
    """
    if 'Model' in df.columns and (df['Model'] == 'Model A').any():
        return df
    if not MAIN_RESULTS_PATH.exists():
        print(f'  [WARN] Model A reference not found: {MAIN_RESULTS_PATH}')
        return df

    main = pd.read_csv(MAIN_RESULTS_PATH)
    rows = []
    for oc in sorted(df['Outcome'].dropna().unique()):
        mr = main[main['Outcome'] == oc]
        if mr.empty:
            continue
        mr = mr.iloc[0]
        rows.append({
            'Outcome': oc,
            'Model': 'Model A',
            'k_total': 3,
            'k_path': 1,
            'k_cons': 2,
            'C_apparent': round(float(mr.get('ArmA_Structured_apparent_C', np.nan)), 5),
            'C_corrected': round(float(mr.get('ArmA_Structured_corrected_C', np.nan)), 5),
            'CI_lower': round(float(mr.get('ArmA_Structured_C_CI_lower', np.nan)), 5),
            'CI_upper': round(float(mr.get('ArmA_Structured_C_CI_upper', np.nan)), 5),
            'AIC': round(float(mr.get('ArmA_Structured_AIC', np.nan)), 2),
            'N': int(mr.get('N_patients', np.nan)),
            'N_events': int(mr.get('N_events', np.nan)),
        })
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# ========================== Plotting: combined 4-panel ==========================

def plot_combined(combined: pd.DataFrame, ci_style: str = 'band') -> None:
    """
    2x2 panel figure, one subplot per outcome.
    y-axis: optimism-corrected C-index with 95% CI ribbon.
    x-axis: k_total (3, 6, 9, 12, 15).
    Dashed horizontal = Baseline corrected C.
    Dotted vertical at k=3 = main analysis reference.
    """
    outdir = OUTDIR if ci_style == 'band' else OUTDIR / 'ci_bars' / 'pca_k_sensitivity'
    outdir.mkdir(parents=True, exist_ok=True)
    oc_order  = [o for o in ['DFS', 'LFS', 'OS', 'DistFS']
                 if o in combined['Outcome'].unique()]
    k_ticks   = [kp + kc for kp, kc in K_PAIRS]  # [3, 6, 9, 12, 15]
    model_list = list(EMBEDDING_ARMS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat  = axes.flatten()

    for ax_idx, oc in enumerate(oc_order):
        ax  = axes_flat[ax_idx]
        sub = combined[combined['Outcome'] == oc]

        # Baseline reference
        base_row  = sub[sub['Model'] == 'Baseline']
        base_corr = base_row['C_corrected'].values[0] if not base_row.empty else np.nan
        base_ci_lo = base_row['CI_lower'].values[0] if not base_row.empty else np.nan
        base_ci_hi = base_row['CI_upper'].values[0] if not base_row.empty else np.nan

        if np.isfinite(base_corr):
            ax.axhline(base_corr, color=BASELINE_COLOR, ls='--', lw=1.8,
                       label='Baseline', zorder=2)
            if ci_style == 'band' and np.isfinite(base_ci_lo) and np.isfinite(base_ci_hi):
                ax.axhspan(base_ci_lo, base_ci_hi, color=BASELINE_COLOR, alpha=0.06)

        # Vertical reference at k=3 (main analysis); label added after y-limits
        # are set so it is not pushed too low.
        ax.axvline(3, color='#bbbbbb', ls=':', lw=1.2, zorder=1)

        for arm in [m for m in ['Model A'] + model_list if m in sub['Model'].unique()]:
            asub = sub[sub['Model'] == arm].sort_values('k_total')
            if asub.empty:
                continue
            xs    = asub['k_total'].values
            ys    = asub['C_corrected'].values
            ci_lo = asub['CI_lower'].values
            ci_hi = asub['CI_upper'].values
            col   = ARM_COLORS.get(arm, '#888')

            ax.plot(xs, ys, marker='o', markersize=5, lw=2.0,
                    color=col, label=DISPLAY_LABELS.get(arm, arm), zorder=3)
            valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if valid.any() and ci_style == 'band':
                ax.fill_between(xs[valid], ci_lo[valid], ci_hi[valid],
                                color=col, alpha=0.08, linewidth=0)
            elif valid.any() and ci_style == 'bars':
                ax.errorbar(xs[valid], ys[valid],
                            yerr=[ys[valid] - ci_lo[valid], ci_hi[valid] - ys[valid]],
                            fmt='none', ecolor=col, elinewidth=1.0,
                            capsize=2.5, alpha=0.65, zorder=2)

        n_val = int(sub['N'].max()) if 'N' in sub.columns and not sub['N'].isna().all() else '?'
        ax.set_title(oc, fontsize=13, fontweight='bold', pad=7)
        ax.set_xlabel('Total PCA components (k_path + k_cons)', fontsize=11)
        ax.set_ylabel(f'Optimism-corrected C-index (95% CI)', fontsize=11)
        ax.set_xticks(k_ticks)
        ax.set_xticklabels([f'{k}\n({kp}+{kc})' for (kp, kc), k
                            in zip(K_PAIRS, k_ticks)], fontsize=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.text(0.02, 0.97, f'N = {n_val}', transform=ax.transAxes,
                fontsize=9, color='#555', va='top', ha='left')
        ax.grid(alpha=0.12)

        # Y-axis padding
        all_lo = sub.loc[sub['Model'] != 'Baseline', 'CI_lower'].dropna()
        all_hi = sub.loc[sub['Model'] != 'Baseline', 'CI_upper'].dropna()
        ylo = max(0.38, all_lo.min() - 0.03) if not all_lo.empty else 0.42
        yhi = min(1.00, all_hi.max() + 0.03) if not all_hi.empty else 0.80
        if np.isfinite(base_corr):
            ylo = min(ylo, base_corr - 0.05)
            yhi = max(yhi, base_corr + 0.03)
        ax.set_ylim(ylo, yhi)

        yr = yhi - ylo
        ax.text(3.25, ylo + 0.10 * yr, 'k=3\n(main)',
                fontsize=8, color='black', va='bottom')

        if ci_style == 'bars' and ax_idx == 0:
            handles, labels_l = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels_l, loc='upper left', fontsize=8,
                          frameon=True, framealpha=0.92)
        elif ci_style == 'bars':
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    for ax_idx in range(len(oc_order), 4):
        axes_flat[ax_idx].set_visible(False)

    # Shared legend for the band version only. The CI-bar version keeps a
    # single compact legend inside the top-left panel to avoid excess bottom
    # whitespace.
    handles, labels_l = axes_flat[0].get_legend_handles_labels()
    if handles and ci_style == 'band':
        fig.legend(handles, labels_l,
                   loc='lower center', ncol=min(4, len(handles)),
                   fontsize=9, frameon=True, framealpha=0.92,
                   bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        f'PCA-k Sensitivity\n'
        f'Optimism-Corrected C-index  \u00b7  MICE M={M_MICE}  \u00b7  B={N_BOOT} bootstrap',
        fontsize=12, fontweight='bold', y=1.01)

    rect = [0, 0.04, 1, 1] if ci_style == 'band' else [0, 0.02, 1, 1]
    fig.tight_layout(rect=rect)
    for ext in ('png', 'pdf'):
        p = outdir / f'pca_k_sensitivity_combined.{ext}'
        fig.savefig(p, dpi=400, bbox_inches='tight')
        print(f'Saved combined: {p}')
    plt.close(fig)

# ========================== Main ==========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot-only', action='store_true',
                        help='Read existing CSV(s) and replot without bootstrapping.')
    parser.add_argument('--n-boot', type=int, default=N_BOOT,
                        help=f'Bootstrap iterations (default {N_BOOT})')
    parser.add_argument('--outcomes', nargs='+', default=list(OUTCOME_SPECS.keys()),
                        help='Outcomes to compute (default: all four)')
    args = parser.parse_args()

    combined_csv = OUTDIR / 'pca_k_sensitivity.csv'

    if args.replot_only:
        frames = []
        for oc in args.outcomes:
            p = OUTDIR / f'pca_k_sensitivity_{oc}.csv'
            if p.exists():
                one = add_model_a_k3_reference(pd.read_csv(p))
                one.to_csv(p, index=False)
                _plot_single_outcome(one, oc)
                _plot_single_outcome(one, oc, ci_style='bars')
                frames.append(one)
            else:
                print(f'  [WARN] Not found: {p}')
        if not frames:
            raise FileNotFoundError('No per-outcome CSVs found. Run without --replot-only first.')
        combined = pd.concat(frames, ignore_index=True)
        combined = add_model_a_k3_reference(combined)
        combined.to_csv(combined_csv, index=False)
        plot_combined(combined)
        plot_combined(combined, ci_style='bars')
        return

    df_raw = load_outcomes()
    emb    = load_embeddings()

    all_frames = []
    for oc_label in args.outcomes:
        cfg = OUTCOME_SPECS.get(oc_label)
        if cfg is None:
            print(f'  [WARN] Unknown outcome: {oc_label}. Skipping.')
            continue
        result = run_outcome(
            df_raw, emb,
            oc_label, cfg['time_col'], cfg['event_col'],
            n_boot=args.n_boot,
        )
        if result is not None:
            all_frames.append(result)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = add_model_a_k3_reference(combined)
        combined.to_csv(combined_csv, index=False)
        print(f'\nSaved combined CSV: {combined_csv}')
        for oc_label in args.outcomes:
            one = combined[combined['Outcome'] == oc_label]
            if not one.empty:
                one.to_csv(OUTDIR / f'pca_k_sensitivity_{oc_label}.csv', index=False)
                _plot_single_outcome(one, oc_label)
                _plot_single_outcome(one, oc_label, ci_style='bars')
        plot_combined(combined)
        plot_combined(combined, ci_style='bars')

    print('\nAll outputs in:', OUTDIR)


if __name__ == '__main__':
    main()
