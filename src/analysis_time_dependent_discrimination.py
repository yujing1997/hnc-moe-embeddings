#!/usr/bin/env python3
"""
Supplementary analysis: time-dependent (horizon-censored) discrimination
-------------------------------------------------------------------------
Evaluates Cox-model discrimination at administrative censoring horizons
(12, 24, 36, 60 months) for all four pre-specified outcomes and all arms
(Baseline, Model A, B1-B6 embedding arms).

Rigour matches the main MICE analysis (02_survival_comparison_mice.py):
  * Multiple imputation: M=5 stochastic MICE (sample_posterior=True)
  * Apparent C: pooled over M imputations via logit-scale Rubin's rules
    (Marshall et al. 2009, BMC Med Res Methodol 9:57)
  * Bootstrap optimism: B=1000 Harrell bootstrap (fit on boot, eval on
    original; paired m<->m across M pre-imputed datasets)
  * PCA (k_path=1, k_cons=2) is RE-FIT on each bootstrap sample and
    applied to the original cohort -- same protocol as the main script
  * 95% CI from percentile distribution of bootstrap-corrected C-indices

Outputs (src/plotting/outputs/supplementary/):
  time_dependent_discrimination.csv
  time_dependent_discrimination.png   (2x2 panel, one subplot per outcome)

Usage:
  python src/analysis_time_dependent_discrimination.py
  python src/analysis_time_dependent_discrimination.py --replot-only
  python src/analysis_time_dependent_discrimination.py --n-boot 200   # fast test
"""

from __future__ import annotations
import os
# Pin BLAS/OpenMP to a single thread per process BEFORE importing numpy so that
# joblib process-parallelism does not oversubscribe cores (each worker would
# otherwise spawn its own BLAS thread pool). Must precede numpy import.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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

# ========================== Hyper-parameters ==========================

M_MICE        = 5       # proper multiple imputation datasets
MICE_MAX_ITER = 10      # MICE chained-equation rounds per dataset
MICE_SEED     = 42      # base seed; actual seed = MICE_SEED + m
N_BOOT        = 1000    # Harrell optimism bootstrap iterations
PENALIZER     = 0.1     # Ridge penaliser for CoxPHFitter
K_PATH        = 1       # PCA dims from pathology half  (main analysis default)
K_CONS        = 2       # PCA dims from consultation half
HORIZONS      = [12, 24, 36, 60]   # administrative censoring horizons (months)

# ========================== Feature definitions ==========================

BASELINE_FEATURES = [
    'Age_Numeric', 'Sex_Binary',
    'T_Stage_Numeric', 'N_Stage_Numeric',
    'has_pathology', 'has_consultation',
]
LLM_FEATURES      = ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary']
MICE_IMPUTE_COLS  = [
    'Age_Numeric', 'Sex_Binary',
    'T_Stage_Numeric', 'N_Stage_Numeric',
    'HPV_Binary', 'Smoking_Binary', 'CCS_Binary',
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

COLORS = {
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

MODEL_ORDER = ['Baseline', 'Model A'] + list(EMBEDDING_ARMS.keys())

OUTCOMES = {
    'OS':     {'time_col': 'OS',     'event_col': 'Death',                'label': 'Overall Survival'},
    'DFS':    {'time_col': 'DFS',    'event_col': 'Failure',              'label': 'Disease-Free Survival'},
    'LFS':    {'time_col': 'LFS',    'event_col': 'Locoregional_Failure', 'label': 'Locoregional FS'},
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

    def _hpv(v):
        if pd.isna(v): return np.nan
        v = str(v).lower()
        if any(k in v for k in ['not inferred', 'not available']): return np.nan
        if 'positive' in v: return 1.0
        if 'negative' in v: return 0.0
        return np.nan

    def _smoking(v):
        if pd.isna(v): return np.nan
        v = str(v).lower()
        if 'not inferred' in v or v == '': return np.nan
        if 'never' in v or 'non-smoker' in v: return 0.0
        if any(w in v for w in ['current','former','ex-','quit','ever','smoker','history']): return 1.0
        return np.nan

    def _ccs(v):
        if pd.isna(v): return np.nan
        import re; ns = re.findall(r'\d+', str(v))
        return 1.0 if ns and int(ns[0]) >= 2 else (0.0 if ns else np.nan)

    df['HPV_Binary']     = df['HPV_Status'].apply(_hpv)            if 'HPV_Status'                 in df.columns else np.nan
    df['Smoking_Binary'] = df['Smoking_History'].apply(_smoking)   if 'Smoking_History'             in df.columns else np.nan
    df['CCS_Binary']     = df['Charlson_Comorbidity_Score'].apply(_ccs) if 'Charlson_Comorbidity_Score' in df.columns else np.nan
    if 'CCS_Category' in df.columns and df['CCS_Binary'].isna().all():
        df['CCS_Binary'] = df['CCS_Category'].apply(
            lambda x: 1.0 if x == 'CCS 2+' else (0.0 if x == 'CCS 0-1' else np.nan))
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
    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    safe_impute   = [c for c in impute_cols
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
        for col in ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary']:
            if col in safe_impute:
                result[col] = result[col].round().clip(lower=0, upper=1)
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

def fit_pca_on_mat(mat: np.ndarray, k_path: int = K_PATH, k_cons: int = K_CONS):
    """Fit per-modality PCA on mat. Returns (pca_path, pca_cons, pcs_array).

    Uses the randomized SVD solver: k is tiny (1-2) relative to the ~1000+
    columns per modality half, so randomized SVD is much faster than full SVD
    and is re-fit once per bootstrap iteration.
    """
    half   = mat.shape[1] // 2
    kp     = min(k_path, mat[:, :half].shape[1], mat.shape[0] - 1)
    kc     = min(k_cons, mat[:, half:].shape[1], mat.shape[0] - 1)
    pp     = PCA(n_components=kp, svd_solver='randomized', random_state=42).fit(mat[:, :half])
    pc     = PCA(n_components=kc, svd_solver='randomized', random_state=42).fit(mat[:, half:])
    pcs    = np.concatenate([pp.transform(mat[:, :half]),
                              pc.transform(mat[:, half:])], axis=1)
    return pp, pc, pcs

def apply_pca(mat: np.ndarray, pca_path, pca_cons) -> np.ndarray:
    """Apply pre-fitted PCA objects to mat (bootstrap or original)."""
    half = mat.shape[1] // 2
    return np.concatenate([pca_path.transform(mat[:, :half]),
                           pca_cons.transform(mat[:, half:])], axis=1)

def pcs_to_df(pcs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])

# ========================== Horizon censoring ==========================

def horizon_censor(df: pd.DataFrame, h: int) -> pd.DataFrame:
    out = df.copy()
    out['Event_h'] = ((out['Time'] <= h) & (out['Event'] == 1)).astype(int)
    out['Time_h']  = np.minimum(out['Time'], h)
    return out

# ========================== Core: optimism bootstrap with MICE ==========================

def _cox_c_safe(df_fit: pd.DataFrame, df_eval: pd.DataFrame, feat_cols: list[str]) -> float | None:
    """Fit Cox on df_fit, evaluate C on df_eval. Returns None on any failure."""
    try:
        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(df_fit[feat_cols + ['Time_h', 'Event_h']],
                duration_col='Time_h', event_col='Event_h')
        risk = cph.predict_partial_hazard(df_eval[feat_cols]).values
        return float(concordance_index(df_eval['Time_h'], -risk, df_eval['Event_h']))
    except Exception:
        return None


def compute_arm_results(
    imputed_list:  list[pd.DataFrame],  # M imputed dfs: base_feats + llm + Time + Event
    feat_cols:     list[str],           # non-PCA features for this arm
    mat:           np.ndarray | None,   # embedding matrix (N x D); None = no PCA
    horizons:      list[int],
    n_boot:        int,
    random_state:  int = 42,
    label:         str = '',
    n_jobs:        int = 1,
) -> pd.DataFrame:
    """
    Compute horizon-censored C-indices for a single arm using Harrell's
    optimism bootstrap with MI-pooling.

    Protocol (mirrors main analysis Val-MI):
      Apparent C: pool M apparent Cox C-indices via logit-scale Rubin's rules.
      Bootstrap (b=1..B):
        * Draw boot_idx (same indices applied to all M imputed datasets)
        * [Embedding arm] Re-fit PCA on boot mat; apply to original mat
        * For each m: build boot and orig horizon-censored dfs;
          fit Cox on boot, eval on boot (C_train_m) and orig (C_test_m)
        * optimism_b(h) = mean over m of (C_train_m - C_test_m)
      Corrected C(h) = apparent_C(h) - mean(optimism_b)
      95% CI(h) = percentile([apparent_C - optimism_b for b], [2.5, 97.5])
    """
    M   = len(imputed_list)
    n   = len(imputed_list[0])
    # Independent, reproducible child seeds — one per bootstrap iteration — so
    # results are identical regardless of n_jobs or execution order.
    boot_seeds = np.random.SeedSequence(random_state).spawn(n_boot)

    # ── 1. Apparent C: full cohort, M imputations, Rubin pooling ─────────────
    if mat is not None:
        pca_path_full, pca_cons_full, pcs_full = fit_pca_on_mat(mat)
        pcs_df_full   = pcs_to_df(pcs_full)
        full_feat_cols = feat_cols + list(pcs_df_full.columns)
    else:
        pcs_df_full    = None
        full_feat_cols = feat_cols

    app_c_by_h: dict[int, float] = {}
    nev_by_h:   dict[int, int]   = {}

    for h in horizons:
        app_c_m, nev_m = [], []
        for use_m in imputed_list:
            dfm = use_m.copy().reset_index(drop=True)
            if pcs_df_full is not None:
                dfm = pd.concat([dfm, pcs_df_full.reset_index(drop=True)], axis=1)
            hd = horizon_censor(dfm, h)
            nev_m.append(int(hd['Event_h'].sum()))
            if hd['Event_h'].sum() < 15:
                continue
            c = _cox_c_safe(hd, hd, full_feat_cols)
            if c is not None:
                app_c_m.append(c)
        app_c_by_h[h] = rubin_pool_c(app_c_m)
        nev_by_h[h]   = int(np.mean(nev_m)) if nev_m else 0

    # ── 2. Bootstrap optimism (B iterations, paired m<->m) ───────────────────
    def _one_bootstrap(seed) -> dict[int, float | None]:
        """One optimism-bootstrap iteration → {h: mean optimism or None}."""
        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, n, size=n)

        # [Embedding] Re-fit PCA on bootstrap mat, apply to original
        if mat is not None:
            boot_mat = mat[boot_idx]
            pp_b, pc_b, boot_pcs = fit_pca_on_mat(boot_mat)
            orig_pcs = apply_pca(mat, pp_b, pc_b)
            boot_pcs_df = pcs_to_df(boot_pcs)
            orig_pcs_df = pcs_to_df(orig_pcs)
            boot_feat_cols = feat_cols + list(boot_pcs_df.columns)
        else:
            boot_feat_cols = feat_cols

        per_h_m: dict[int, list[float]] = {h: [] for h in horizons}

        for m_idx in range(M):
            use_m = imputed_list[m_idx].reset_index(drop=True)
            boot_base = use_m.iloc[boot_idx].reset_index(drop=True)
            orig_base = use_m.copy()

            if mat is not None:
                boot_base = pd.concat([boot_base, boot_pcs_df.reset_index(drop=True)], axis=1)
                orig_base = pd.concat([orig_base.reset_index(drop=True),
                                       orig_pcs_df.reset_index(drop=True)], axis=1)

            for h in horizons:
                boot_hd = horizon_censor(boot_base, h)
                orig_hd = horizon_censor(orig_base, h)
                if boot_hd['Event_h'].sum() < 10:
                    continue
                try:
                    cph_b = CoxPHFitter(penalizer=PENALIZER)
                    cph_b.fit(boot_hd[boot_feat_cols + ['Time_h', 'Event_h']],
                              duration_col='Time_h', event_col='Event_h')
                    c_train = concordance_index(
                        boot_hd['Time_h'],
                        -cph_b.predict_partial_hazard(boot_hd[boot_feat_cols]).values,
                        boot_hd['Event_h'])
                    c_test  = concordance_index(
                        orig_hd['Time_h'],
                        -cph_b.predict_partial_hazard(orig_hd[boot_feat_cols]).values,
                        orig_hd['Event_h'])
                    per_h_m[h].append(float(c_train - c_test))
                except Exception:
                    continue

        return {h: (float(np.mean(per_h_m[h])) if per_h_m[h] else None)
                for h in horizons}

    if n_jobs != 1:
        boot_dicts = Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto')(
            delayed(_one_bootstrap)(s) for s in boot_seeds)
    else:
        boot_dicts = [_one_bootstrap(s) for s in boot_seeds]

    boot_opt: dict[int, list[float]] = {h: [] for h in horizons}
    for bd in boot_dicts:
        for h in horizons:
            if bd[h] is not None:
                boot_opt[h].append(bd[h])


    # ── 3. Aggregate into output rows ─────────────────────────────────────────
    rows = []
    for h in horizons:
        app_c = app_c_by_h.get(h, np.nan)
        opts  = boot_opt[h]
        if opts:
            mean_opt  = float(np.mean(opts))
            corr_c    = float(np.clip(app_c - mean_opt, 0.35, 1.0))
            boot_corr = [app_c - o for o in opts]
            ci_lo     = float(np.percentile(boot_corr, 2.5))
            ci_hi     = float(np.percentile(boot_corr, 97.5))
        else:
            mean_opt  = np.nan
            corr_c    = app_c
            ci_lo = ci_hi = np.nan

        if label:
            print(f'      h={h:3d}m  apparent={app_c:.4f}  corrected={corr_c:.4f}  '
                  f'95%CI [{ci_lo:.3f},{ci_hi:.3f}]  events={nev_by_h.get(h,0)}')
        rows.append({
            'HorizonMonths': h,
            'N_events_h':   nev_by_h.get(h, 0),
            'C_apparent':   round(float(app_c), 5)  if np.isfinite(app_c)  else np.nan,
            'C_corrected':  round(float(corr_c), 5) if np.isfinite(corr_c) else np.nan,
            'CI_lower':     round(float(ci_lo), 5)  if np.isfinite(ci_lo)  else np.nan,
            'CI_upper':     round(float(ci_hi), 5)  if np.isfinite(ci_hi)  else np.nan,
            'Mean_optimism':round(float(mean_opt),5) if np.isfinite(mean_opt) else np.nan,
        })
    return pd.DataFrame(rows)

# ========================== Plotting ==========================

def plot_time_dependent(df: pd.DataFrame, outdir: Path, ci_style: str = 'band') -> None:
    """
    2x2 panel, one subplot per outcome.
    x-axis: horizon months; y-axis: optimism-corrected C (95% CI ribbon).
    Baseline = dashed gray, Model A = thick solid red (#d62828), embeddings = colored.
    """
    save_dir = outdir if ci_style == 'band' else outdir / 'ci_bars' / 'time_dependent_discrimination'
    save_dir.mkdir(parents=True, exist_ok=True)
    outcome_order = [o for o in ['DFS', 'LFS', 'OS', 'DistFS']
                     if o in df['Outcome'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
    axes_flat  = axes.flatten()

    ls_kw = {
        'Baseline': dict(lw=1.8, ls='--', zorder=2),
        'Model A':  dict(lw=3.0, ls='-',  zorder=5),
    }

    for ax_idx, oc in enumerate(outcome_order):
        ax   = axes_flat[ax_idx]
        sub  = df[df['Outcome'] == oc]
        n_pt = int(sub['N_patients'].max()) if 'N_patients' in sub.columns else '?'

        # Horizon event counts (median across arms)
        horizon_ev = {}
        for h in HORIZONS:
            hdf = sub[sub['HorizonMonths'] == h]
            if not hdf.empty:
                horizon_ev[h] = int(hdf['N_events_h'].median())

        for ml in MODEL_ORDER:
            msub = sub[sub['Model'] == ml].sort_values('HorizonMonths')
            if msub.empty or msub['C_corrected'].isna().all():
                continue
            xs    = msub['HorizonMonths'].values
            ys    = msub['C_corrected'].values
            ci_lo = msub['CI_lower'].values
            ci_hi = msub['CI_upper'].values
            col   = COLORS.get(ml, '#888')
            kw    = ls_kw.get(ml, dict(lw=1.8, ls='-', zorder=3))

            ax.plot(xs, ys, marker='o', markersize=5, color=col,
                    label=DISPLAY_LABELS.get(ml, ml), **kw)
            valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if valid.any() and ci_style == 'band':
                ax.fill_between(xs[valid],
                                np.clip(ci_lo[valid], 0.35, 1.0),
                                np.clip(ci_hi[valid], 0.35, 1.0),
                                color=col, alpha=0.10, linewidth=0)
            elif valid.any() and ci_style == 'bars':
                ax.errorbar(xs[valid], ys[valid],
                            yerr=[ys[valid] - ci_lo[valid], ci_hi[valid] - ys[valid]],
                            fmt='none', ecolor=col, elinewidth=1.0,
                            capsize=2.5, alpha=0.65, zorder=kw.get('zorder', 3) - 0.1)

        # Reference lines
        ax.axhline(0.5, color='#cccccc', ls=':', lw=1.0, zorder=1, label='_nolegend_')
        for h in HORIZONS:
            ax.axvline(h, color='#eeeeee', lw=0.6, zorder=0)

        ax.set_title(oc, fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('Follow-up horizon (months)', fontsize=12)
        ax.set_ylabel('Optimism-corrected C-index (95% CI)', fontsize=12)
        ax.set_xticks(HORIZONS)
        ax.tick_params(labelsize=11)

        # Patient count (top-left corner)
        ax.text(0.03, 0.97, f'N = {n_pt}',
                transform=ax.transAxes, fontsize=10, color='#444',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

        # Events-per-horizon strip below x-axis
        ev_parts = [f'{h}m: {horizon_ev.get(h,"?")} ev' for h in HORIZONS]
        ax.text(0.5, -0.15, '  |  '.join(ev_parts),
                transform=ax.transAxes, fontsize=9, color='#888',
                ha='center', va='top')

        # Y-axis limits
        all_lo = sub['CI_lower'].dropna()
        all_hi = sub['CI_upper'].dropna()
        ylo = max(0.35, all_lo.min() - 0.03) if not all_lo.empty else 0.40
        yhi = min(1.00, all_hi.max() + 0.03) if not all_hi.empty else 0.85
        ax.set_ylim(ylo, yhi)

    for ax_idx in range(len(outcome_order), 4):
        axes_flat[ax_idx].set_visible(False)

    # Shared legend below figure
    handles, labels_l = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend_y = -0.015 if ci_style == 'bars' else -0.06
        fig.legend(handles, labels_l,
                   loc='lower center', ncol=min(4, len(handles)),
                   fontsize=10, frameon=True, framealpha=0.92,
                   bbox_to_anchor=(0.5, legend_y))

    fig.suptitle(
        'TD Discrimination\n'
        f'Optimism-corrected C-index  \u00b7  MICE M={M_MICE}  \u00b7  B={N_BOOT} bootstrap',
        fontsize=13, fontweight='bold', y=1.01)

    rect = [0, 0.07, 1, 1] if ci_style == 'bars' else [0, 0.04, 1, 1]
    fig.tight_layout(rect=rect)
    for ext in ('png', 'pdf'):
        p = save_dir / f'time_dependent_discrimination.{ext}'
        fig.savefig(p, dpi=400, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)

    for oc in outcome_order:
        _plot_single_time_dependent(df, oc, outdir, ci_style=ci_style)


def _plot_single_time_dependent(df: pd.DataFrame, oc: str, outdir: Path,
                                ci_style: str = 'band') -> None:
    """Save one outcome-specific time-dependent discrimination figure."""
    save_dir = outdir if ci_style == 'band' else outdir / 'ci_bars' / 'time_dependent_discrimination'
    save_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df['Outcome'] == oc]
    if sub.empty:
        return
    n_pt = int(sub['N_patients'].max()) if 'N_patients' in sub.columns else '?'

    horizon_ev = {}
    for h in HORIZONS:
        hdf = sub[sub['HorizonMonths'] == h]
        if not hdf.empty:
            horizon_ev[h] = int(hdf['N_events_h'].median())

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.2))
    ls_kw = {
        'Baseline': dict(lw=1.8, ls='--', zorder=2),
        'Model A':  dict(lw=3.0, ls='-',  zorder=5),
    }

    for ml in MODEL_ORDER:
        msub = sub[sub['Model'] == ml].sort_values('HorizonMonths')
        if msub.empty or msub['C_corrected'].isna().all():
            continue
        xs    = msub['HorizonMonths'].values
        ys    = msub['C_corrected'].values
        ci_lo = msub['CI_lower'].values
        ci_hi = msub['CI_upper'].values
        col   = COLORS.get(ml, '#888')
        kw    = ls_kw.get(ml, dict(lw=1.8, ls='-', zorder=3))

        ax.plot(xs, ys, marker='o', markersize=5, color=col,
            label=DISPLAY_LABELS.get(ml, ml), **kw)
        valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if valid.any() and ci_style == 'band':
            ax.fill_between(xs[valid],
                            np.clip(ci_lo[valid], 0.35, 1.0),
                            np.clip(ci_hi[valid], 0.35, 1.0),
                            color=col, alpha=0.10, linewidth=0)
        elif valid.any() and ci_style == 'bars':
            ax.errorbar(xs[valid], ys[valid],
                yerr=[ys[valid] - ci_lo[valid], ci_hi[valid] - ys[valid]],
                fmt='none', ecolor=col, elinewidth=1.0,
                capsize=2.5, alpha=0.65, zorder=kw.get('zorder', 3) - 0.1)

    ax.axhline(0.5, color='#cccccc', ls=':', lw=1.0, zorder=1)
    for h in HORIZONS:
        ax.axvline(h, color='#eeeeee', lw=0.6, zorder=0)

    ax.set_title(oc, fontsize=14, fontweight='bold', pad=8)
    ax.set_xlabel('Follow-up horizon (months)', fontsize=12)
    ax.set_ylabel('Optimism-corrected C-index (95% CI)', fontsize=12)
    ax.set_xticks(HORIZONS)
    ax.tick_params(labelsize=11)
    ax.text(0.03, 0.97, f'N = {n_pt}', transform=ax.transAxes,
            fontsize=10, color='#444', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    ev_parts = [f'{h}m: {horizon_ev.get(h,"?")} ev' for h in HORIZONS]
    ax.text(0.5, -0.14, '  |  '.join(ev_parts), transform=ax.transAxes,
            fontsize=9, color='#888', ha='center', va='top')

    all_lo = sub['CI_lower'].dropna()
    all_hi = sub['CI_upper'].dropna()
    ylo = max(0.35, all_lo.min() - 0.03) if not all_lo.empty else 0.40
    yhi = min(1.00, all_hi.max() + 0.03) if not all_hi.empty else 0.85
    ax.set_ylim(ylo, yhi)
    ax.legend(loc='best', fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(alpha=0.12)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = save_dir / f'time_dependent_discrimination_{oc}.{ext}'
        fig.savefig(p, dpi=400, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)

# ========================== Main ==========================

def main() -> None:
    _default_jobs = max(1, (os.cpu_count() or 2) // 2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot-only', action='store_true')
    parser.add_argument('--n-boot', type=int, default=N_BOOT,
                        help=f'Bootstrap iterations (default {N_BOOT})')
    parser.add_argument('--m-mice', type=int, default=M_MICE,
                        help=f'MICE datasets (default {M_MICE})')
    parser.add_argument('--n-jobs', type=int, default=_default_jobs,
                        help=f'Parallel workers for bootstrap loop '
                             f'(default {_default_jobs} = half of {os.cpu_count()} cores; '
                             f'use 1 for serial)')
    args = parser.parse_args()

    csv_path = OUTDIR / 'time_dependent_discrimination.csv'

    if args.replot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f'Run without --replot-only first: {csv_path}')
        result = pd.read_csv(csv_path)
        plot_time_dependent(result, OUTDIR)
        plot_time_dependent(result, OUTDIR, ci_style='bars')
        return

    print(f'=== Time-dependent discrimination  M={args.m_mice}  B={args.n_boot}  '
          f'n_jobs={args.n_jobs} ===\n')

    df_raw = load_outcomes()
    emb    = load_embeddings()

    all_rows: list[pd.DataFrame] = []

    for oc_name, ocfg in OUTCOMES.items():
        time_col  = ocfg['time_col']
        event_col = ocfg['event_col']
        print(f'\n{"="*60}')
        print(f'OUTCOME: {oc_name}  ({time_col} / {event_col})')
        print(f'{"="*60}')

        # Build working dataframe
        needed = (['patient_id', time_col, event_col]
                  + BASELINE_FEATURES[:4] + LLM_FEATURES)
        avail  = [c for c in needed if c in df_raw.columns]
        use    = df_raw[avail].copy()
        use    = use.rename(columns={time_col: 'Time', event_col: 'Event'})
        use['Event'] = pd.to_numeric(use['Event'], errors='coerce')
        use = use[(use['Time'] > 0) & (use['Time'] <= 240)
                  & (use['Event'].isin([0.0, 1.0]))].copy().reset_index(drop=True)
        use['Event'] = use['Event'].astype(int)

        # Embedding-presence indicators from first available embedding dict
        first_emb = next(iter(emb.values()), {})
        has_path, has_cons = [], []
        for pid in use['patient_id'].astype(str):
            vec = first_emb.get(pid)
            if vec is None:
                has_path.append(0); has_cons.append(0); continue
            hh = vec.shape[0] // 2
            has_path.append(int(np.any(np.abs(vec[:hh]) > 1e-12)))
            has_cons.append(int(np.any(np.abs(vec[hh:]) > 1e-12)))
        use['has_pathology']    = has_path
        use['has_consultation'] = has_cons

        # Restrict to patients present in ALL embedding arms
        keep = set(use['patient_id'].astype(str))
        for d in emb.values():
            keep &= set(d.keys())
        use = use[use['patient_id'].astype(str).isin(keep)].copy().reset_index(drop=True)

        if len(use) == 0 or use['Event'].sum() == 0:
            print('  [SKIP] No valid patients or events.')
            continue
        print(f'  Cohort: N={len(use)}, events={use["Event"].sum()}')

        # MICE M=5 on full cohort (baseline + LLM features together)
        impute_cols  = [c for c in MICE_IMPUTE_COLS if c in use.columns]
        imputed_list = mice_impute_multiple(use, impute_cols, M=args.m_mice)
        print(f'  MICE M={args.m_mice} complete.')

        # Embedding matrices aligned to use row order
        pids = use['patient_id'].astype(str).tolist()
        arm_mats: dict[str, np.ndarray] = {}
        for arm_label, d in emb.items():
            try:
                arm_mats[arm_label] = np.stack([d[pid] for pid in pids])
            except KeyError as e:
                print(f'  [WARN] Missing pid {e} for {arm_label}.')

        # Feature columns
        base_cols = [f for f in BASELINE_FEATURES if f in use.columns]
        llm_cols  = [f for f in LLM_FEATURES      if f in use.columns]

        arms_to_run = {
            'Baseline': (base_cols, None),
            'Model A':  (base_cols + llm_cols, None),
        }
        for arm_label in arm_mats:
            arms_to_run[arm_label] = (base_cols, arm_mats[arm_label])

        for arm_label, (feat_cols, mat) in arms_to_run.items():
            print(f'\n  -- {arm_label} --')
            arm_df = compute_arm_results(
                imputed_list  = imputed_list,
                feat_cols     = feat_cols,
                mat           = mat,
                horizons      = HORIZONS,
                n_boot        = args.n_boot,
                random_state  = 42,
                label         = arm_label,
                n_jobs        = args.n_jobs,
            )
            arm_df['Outcome']    = oc_name
            arm_df['Model']      = arm_label
            arm_df['N_patients'] = len(use)
            all_rows.append(arm_df)

    if not all_rows:
        print('No results generated.')
        return

    result   = pd.concat(all_rows, ignore_index=True)
    col_order = ['Outcome', 'Model', 'HorizonMonths', 'N_patients', 'N_events_h',
                 'C_apparent', 'C_corrected', 'CI_lower', 'CI_upper', 'Mean_optimism']
    result   = result[[c for c in col_order if c in result.columns]]
    result   = result.sort_values(['Outcome', 'Model', 'HorizonMonths'])
    result.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    plot_time_dependent(result, OUTDIR)
    plot_time_dependent(result, OUTDIR, ci_style='bars')
    print('\nDone.')


if __name__ == '__main__':
    main()
