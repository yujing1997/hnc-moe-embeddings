#!/usr/bin/env python3
"""
02_survival_comparison_mice.py — Cox + PCA + Proper Multiple Imputation + Bootstrap
======================================================================================
Refactored from 02_survival_comparison.py to implement proper multiple imputation
(M=5, sample_posterior=True) nested inside Harrell's bootstrap optimism correction.

# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY NOTE  (Section 2.x Missing-Data Methods)
# ─────────────────────────────────────────────────────────────────────────────
# Missing values in four clinical baseline covariates (age, sex, T-stage,
# N-stage) and, in the full-imputation analysis mode, three LLM-extracted
# features (HPV status, smoking history, Charlson Comorbidity Score) were
# handled by Multiple Imputation by Chained Equations (MICE; van Buuren 2018).
# We generated M = 5 imputed datasets using scikit-learn's IterativeImputer
# with sample_posterior=True (BayesianRidge estimator, max_iter=10), drawing
# one posterior sample per cell so that each dataset is stochastically distinct.
# All available numeric predictors — including time-to-event, event indicators,
# and all structured LLM features — were used as auxiliary variables in the
# imputation model; only the target columns were overwritten.
#
# Predictive performance was estimated by Harrell's bootstrap optimism
# correction (B = 1 000 resamples; Harrell 1996; Steyerberg 2019, Ch. 17)
# with the "Val-MI" protocol of Wahl et al. (2016) and Musoro et al. (2014):
# within each bootstrap resample b, the raw (pre-imputation) rows are drawn
# first, then imputed M times independently; likewise, the original cohort is
# imputed M times once before the loop (fixed seeds). For each m in 1..M the
# bootstrap-fitted model is evaluated on the m-th imputation of the original
# data (paired m-to-m, never crossed). Rubin's combining rules then pool the
# M per-imputation metrics into a single Rubin-pooled estimate for resample b.
# The 1 000 Rubin-pooled test-C values form the bootstrap distribution from
# which mean optimism and 95 % percentile CIs are derived. This two-level
# variance decomposition (imputation variance collapsed within b by Rubin's
# rules; sampling variance across b by the bootstrap) is the recommended
# approach for combining internal validation with MI (Wahl 2016; Musoro 2014).
#
# For likelihood-ratio tests between nested Cox models, we apply the Meng–Rubin
# D₂ statistic (Meng & Rubin 1992), which combines the M LRT statistics into
# an F-test with degrees of freedom adjusted for the fraction of missing
# information.  The fraction of missing information (FMI) is reported for
# each LLM-extracted coefficient.
#
# References
# ----------
# Rubin DB (1987) Multiple Imputation for Nonresponse in Surveys. Wiley.
# Meng X-L, Rubin DB (1992) Biometrika 79:103-111.
# Harrell FE et al. (1996) Stat Med 15:361-387.
# Marshall A et al. (2009) BMC Med Res Methodol 9:57.
# Wahl S et al. (2016) BMC Med Res Methodol 16:144.
# Musoro JZ et al. (2014) BMC Med Res Methodol 14:116.
# van Buuren S (2018) Flexible Imputation of Missing Data, 2nd ed. CRC Press.
# Steyerberg EW (2019) Clinical Prediction Models, 2nd ed. Springer, Ch. 17.
# ─────────────────────────────────────────────────────────────────────────────

# CHANGELOG vs. 02_survival_comparison.py
# ----------------------------------------
# mice_impute()           → removed; replaced by mice_impute_multiple()
# mice_impute_multiple()  → NEW: returns M stochastic imputed DataFrames
#                           (sample_posterior=True, seed per imputation)
# meng_rubin_d2()         → NEW: Meng-Rubin D2 pooled F-test for MI LRTs
# _single_bootstrap()     → renamed _single_bootstrap_mice(); now accepts
#                           complete_df_list (M pre-imputed originals), draws
#                           M bootstrap imputations internally, pools with
#                           Rubin's rules before returning
# run_analysis_for_outcome() → updated: pre-computes M original imputations,
#                           calls _single_bootstrap_mice, updated aggregation
#                           (Rubin-inside-bootstrap for CI propagation),
#                           FMI + B/W sanity-check reporting, full-cohort D2 LRT
# OUTPUT_DIR              → hardcoded to full_mice_m5_b1000 subdir
# _run_and_save()         → updated to save comprehensive .pkl alongside CSV
# ─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import pickle
import time
import logging
import warnings
import numpy as np
import pandas as pd

# ── NumPy pickle compatibility shim ─────────────────────────────────────────
# Embedding pkl files saved with NumPy 2.0–2.3 reference numpy._core sub-
# modules (e.g. numpy._core.numeric, numpy._core.multiarray) as importable
# paths. NumPy 2.4 reorganised _core internals; those paths may no longer be
# directly importable, breaking pickle.load. Register aliases as needed.
def _install_numpy_pickle_compat():
    import sys, types, numpy._core as _npc
    for _name in ('numpy._core.numeric', 'numpy._core.multiarray',
                  'numpy._core.umath', 'numpy._core.fromnumeric',
                  'numpy._core.shape_base'):
        if _name not in sys.modules:
            _m = types.ModuleType(_name)
            _m.__dict__.update(_npc.__dict__)
            sys.modules[_name] = _m
_install_numpy_pickle_compat()
del _install_numpy_pickle_compat
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================== CONFIGURATION ==========================

BASE_DIR = Path("/Data/Yujing/HNC_OutcomePred/Reports_Agents")
EMBEDDINGS_DIR = (BASE_DIR / "Revision1" / "Downstream_Analysis"
                  / "moe_embeddings_comp" / "embeddings")
OUTCOMES_PATH = (BASE_DIR / "Revision1" / "Downstream_Analysis"
                 / "outputs_all_fields" / "all_fields_processed.csv")
# Output goes to full_mice_m5_b1000 sub-directory
OUTPUT_DIR = (BASE_DIR / "Revision1" / "Downstream_Analysis"
              / "moe_embeddings_comp" / "results_v2_permodality"
              / "full_mice" / "full_mice_m5_b1000")

# Cox model
PENALIZER = 0.1
MAX_FOLLOWUP_MONTHS = 240
PCA_K = 3

# Per-modality PCA (k_path=1, k_consult=2 mirrors HPV / Smoking+CCS split)
PCA_K_PATH = 1
PCA_K_CONSULT = 2

# IBS
IBS_TIME_HORIZON_MONTHS = 24

# ── Multiple Imputation ──────────────────────────────────────────────────────
M_IMPUTATIONS = 5        # number of stochastic imputed datasets
MICE_MAX_ITER = 10       # chained-equation rounds per imputed dataset
MICE_RANDOM_STATE = 42   # base seed; actual seed = base + m (0..M-1)
# Only impute these tabular columns (never Time / Event):
MICE_IMPUTE_COLS = ['Age_Numeric', 'Sex_Binary', 'T_Stage_Numeric', 'N_Stage_Numeric']

# Outcomes
OUTCOMES = {
    'OS':     {'time': 'OS',     'event': 'Death'},
    'DFS':    {'time': 'DFS',    'event': 'Failure'},
    'LFS':    {'time': 'LFS',    'event': 'Locoregional_Failure'},
    'DistFS': {'time': 'DistFS', 'event': 'distant failure'},
}

BASELINE_FEATURES = ['Age_Numeric', 'Sex_Binary', 'T_Stage_Numeric', 'N_Stage_Numeric']
LLM_FEATURES      = ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary']

MODALITY_FEATURES = {
    'all':          ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary'],
    'both_reports': ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary'],
    'consult_only': ['Smoking_Binary', 'CCS_Binary'],
    'path_only':    ['HPV_Binary'],
}
MODALITY_PCA_K = {
    'all':          {'path': PCA_K_PATH, 'consult': PCA_K_CONSULT},
    'both_reports': {'path': PCA_K_PATH, 'consult': PCA_K_CONSULT},
    'consult_only': {'consult': PCA_K_CONSULT},
    'path_only':    {'path': PCA_K_PATH},
}
MISSINGNESS_INDICATORS = ['has_pathology', 'has_consultation']

EMBEDDING_ARMS = {
    'B1_SMB':           'smb/smb_dual_embeddings.pkl',
    'B2_Longformer':    'longformer/longformer_dual_embeddings.pkl',
    'B3_Llama':         'llama/llama_dual_embeddings.pkl',
    'B4_Nomic':         'nomic/nomic_dual_embeddings.pkl',
    'B5_SMB8B_Qwen3':  'smb8b_qwen3_mo/smb8b_qwen3_mo_dual_embeddings.pkl',
    'B6_SMB8B_Llama3': 'smb8b_llama3_cl/smb8b_llama3_cl_dual_embeddings.pkl',
}


# ========================== TEE WRITER ==========================

class TeeWriter:
    """Tee sys.stdout to both terminal and a text file."""

    def __init__(self, filepath: str, mode: str = 'w'):
        self.filepath = filepath
        self.mode = mode
        self._file = None
        self._orig_stdout = None

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._file = open(self.filepath, self.mode, encoding='utf-8')
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._orig_stdout
        if self._file:
            self._file.close()

    def write(self, data):
        self._orig_stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._orig_stdout.flush()
        self._file.flush()


# ========================== TQDM JOBLIB ==========================

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# ========================== DATA LOADING ==========================

def load_outcomes() -> pd.DataFrame:
    df = pd.read_csv(OUTCOMES_PATH)
    logger.info(f"Loaded outcomes: {len(df)} rows, {len(df.columns)} cols")
    df['patient_id'] = df['patient_id'].astype(str)

    df['Age_Numeric'] = pd.to_numeric(df.get('Age', np.nan), errors='coerce')
    df['Sex_Binary'] = df.get('sex', pd.Series(dtype=str)).astype(str).apply(
        lambda x: 1 if x.lower().startswith('m') else (0 if x.lower().startswith('f') else np.nan))
    df['T_Stage_Numeric'] = df.get('T stage', pd.Series(dtype=str)).astype(str).apply(
        lambda x: int(x[1]) if isinstance(x, str) and x.startswith('T')
        and len(x) > 1 and x[1].isdigit() else np.nan)
    df['N_Stage_Numeric'] = df.get('N stage', pd.Series(dtype=str)).astype(str).apply(
        lambda x: int(x[1]) if isinstance(x, str) and x.startswith('N')
        and len(x) > 1 and x[1].isdigit() else np.nan)

    def _hpv(v):
        if pd.isna(v): return np.nan
        v = str(v).lower()
        if 'not inferred' in v or v == '' or 'not available' in v: return np.nan
        if 'positive' in v: return 1
        if 'negative' in v: return 0
        return np.nan

    def _smoking(v):
        if pd.isna(v): return np.nan
        v = str(v).lower()
        if 'not inferred' in v or v == '': return np.nan
        if 'never' in v or 'non-smoker' in v or 'non smoker' in v: return 0
        if any(w in v for w in ['current','former','ex-','quit','ever','smoker','history']): return 1
        return np.nan

    def _ccs(v):
        if pd.isna(v): return np.nan
        import re
        nums = re.findall(r'\d+', str(v))
        if nums:
            return 1 if int(nums[0]) >= 2 else 0
        return np.nan

    df['HPV_Binary']     = df['HPV_Status'].apply(_hpv) if 'HPV_Status' in df.columns else np.nan
    df['Smoking_Binary'] = df['Smoking_History'].apply(_smoking) if 'Smoking_History' in df.columns else np.nan
    df['CCS_Binary']     = df['Charlson_Comorbidity_Score'].apply(_ccs) if 'Charlson_Comorbidity_Score' in df.columns else np.nan
    if 'CCS_Category' in df.columns and df['CCS_Binary'].isna().all():
        df['CCS_Binary'] = df['CCS_Category'].apply(
            lambda x: 1 if x == 'CCS 2+' else (0 if x == 'CCS 0-1' else np.nan))
    return df


def load_embeddings() -> Dict[str, Dict[str, np.ndarray]]:
    loaded = {}
    for arm_name, filename in EMBEDDING_ARMS.items():
        path = EMBEDDINGS_DIR / filename
        if path.exists():
            with open(path, 'rb') as f:
                emb_dict = pickle.load(f)
            emb_dict = {str(k): v for k, v in emb_dict.items()}
            dims = set(v.shape[0] for v in emb_dict.values())
            logger.info(f"Loaded {arm_name}: {len(emb_dict)} patients, dim={dims}")
            loaded[arm_name] = emb_dict
        else:
            logger.warning(f"Not found: {path}. Arm {arm_name} skipped.")
    return loaded


# ========================== MULTIPLE IMPUTATION ==========================

def mice_impute_multiple(
    df: pd.DataFrame,
    impute_cols: List[str],
    M: int = M_IMPUTATIONS,
    random_state: int = MICE_RANDOM_STATE,
) -> List[pd.DataFrame]:
    """Return M stochastically imputed copies of df using MICE.

    Each copy is produced by IterativeImputer with sample_posterior=True and
    a distinct seed (random_state + m for m in 0..M-1), so the M datasets
    differ due to posterior sampling rather than only rounding differences.

    sample_posterior=True: draws each imputed cell from the posterior
    predictive distribution of the BayesianRidge imputation model, giving
    proper multiple imputation as described in van Buuren (2018) §2.4.
    sample_posterior=False (the PREVIOUS code) only returns the posterior
    mean — equivalent to single imputation, which underestimates uncertainty.

    References
    ----------
    van Buuren S (2018) Flexible Imputation of Missing Data, Ch. 2. CRC Press.
    scikit-learn IterativeImputer documentation.
    """
    n_missing = df[impute_cols].isna().sum().sum()
    if n_missing == 0:
        # No missing values: all M datasets are identical copies
        return [df.copy() for _ in range(M)]

    # Use all numeric columns as auxiliary predictors (including Time, Event,
    # LLM features) but only overwrite impute_cols.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Never overwrite time/event — exclude from impute_cols if accidentally included
    safe_impute = [c for c in impute_cols if c in numeric_cols and c not in ('Time', 'Event')]

    datasets = []
    for m in range(M):
        imputer = IterativeImputer(
            max_iter=MICE_MAX_ITER,
            random_state=random_state + m,
            sample_posterior=True,   # ← stochastic posterior draw (proper MI)
        )
        result = df.copy()
        imputed_arr = imputer.fit_transform(result[numeric_cols])
        imputed_df  = pd.DataFrame(imputed_arr, columns=numeric_cols, index=result.index)

        for col in safe_impute:
            result[col] = imputed_df[col]

        # Round binary/ordinal columns back to valid integer codes
        for col in ['Sex_Binary', 'T_Stage_Numeric', 'N_Stage_Numeric']:
            if col in safe_impute and col in result.columns:
                result[col] = result[col].round().clip(lower=0)
        for col in ['HPV_Binary', 'Smoking_Binary', 'CCS_Binary']:
            if col in safe_impute and col in result.columns:
                result[col] = result[col].round().clip(lower=0, upper=1)

        datasets.append(result)

    n_imputed = sum(ds[safe_impute].isna().sum().sum() for ds in datasets)
    logger.info(f"MICE (M={M}, posterior=True): imputed {n_missing} missing cells × M datasets")
    if n_imputed > 0:
        logger.warning(f"  {n_imputed} cells still NaN after imputation — check predictor coverage")
    return datasets


# ========================== MENG-RUBIN D2 LRT ==========================

def meng_rubin_d2(
    lrt_stats: List[float],
    k: int,
    M: int,
) -> Tuple[float, float, float]:
    """Pool M likelihood-ratio test statistics using the Meng-Rubin D2 statistic.

    Returns (D2, p_value, r2) where D2 ~ F(k, nu2).

    Formula (Meng & Rubin 1992, eq. 3.3):
        d_bar = mean(lrt_stats)
        r2    = (M+1) / (k*(M-1)) * Var(sqrt(lrt_stats))
        D2    = (d_bar/k - (1 - 1/M) * r2) / (1 + r2)
        nu2   = k^(-3/M) * (M-1) * (1 + 1/r2)^2

    References
    ----------
    Meng X-L, Rubin DB (1992) Biometrika 79:103-111.
    Marshall A et al. (2009) BMC Med Res Methodol 9:57.
    """
    lrt = np.array([x for x in lrt_stats if np.isfinite(x)], dtype=float)
    if len(lrt) < 2:
        # Fallback: average chi2 → chi2 p-value
        d_bar = float(np.mean(lrt)) if len(lrt) else np.nan
        p = float(1 - stats.chi2.cdf(d_bar, df=k)) if np.isfinite(d_bar) else np.nan
        return float(d_bar / max(k, 1)), p, np.nan

    M_used = len(lrt)
    d_bar  = float(np.mean(lrt))

    sqrt_lrt = np.sqrt(np.maximum(lrt, 0.0))
    var_sqrt  = float(np.var(sqrt_lrt, ddof=1))          # sample variance
    r2 = ((M_used + 1) / (k * (M_used - 1))) * var_sqrt

    if r2 <= 0 or not np.isfinite(r2):
        # Between-imputation variance is negligible → classical chi2 / k
        D2 = d_bar / k
        p  = float(1 - stats.f.cdf(D2, k, 1e6))         # large df ≈ chi2/k
        return D2, p, 0.0

    numerator   = d_bar / k - (1.0 - 1.0 / M_used) * r2
    denominator = 1.0 + r2
    D2 = numerator / denominator

    if D2 <= 0:
        return float(D2), 1.0, float(r2)

    nu2 = (k ** (-3.0 / M_used)) * (M_used - 1) * ((1.0 + 1.0 / r2) ** 2)
    nu2 = max(nu2, 1.0)

    p = float(1 - stats.f.cdf(D2, k, nu2))
    return float(D2), p, float(r2)


# ========================== IBS ==========================

def compute_ibs(cph_model, test_df, train_df,
                time_col='Time', event_col='Event',
                n_times=100, max_time=IBS_TIME_HORIZON_MONTHS):
    """2-year IPCW Integrated Brier Score (vectorised). Lower = better.

    Vectorised over patients (numpy broadcasting) — ~50-100x faster than the
    previous per-patient Python loop. IPCW weights use G(·) estimated on
    train_df (Graf et al. 1999; Gerds & Schumacher 2006).
    """
    feature_cols = [c for c in test_df.columns if c not in [time_col, event_col]]
    t_max = min(max_time, test_df[time_col].max() * 0.95) if max_time else test_df[time_col].max() * 0.9
    times = np.linspace(0.01, t_max, n_times)
    try:
        surv_func = cph_model.predict_survival_function(test_df[feature_cols], times=times)
    except Exception:
        return np.nan

    kmf = KaplanMeierFitter()
    kmf.fit(train_df[time_col], event_observed=1 - train_df[event_col])

    T_arr = test_df[time_col].values          # (n,)
    E_arr = test_df[event_col].values         # (n,)
    # Censoring survival G(Ti) per patient, clipped away from zero
    G_Ti  = np.maximum(np.array([kmf.predict(ti) for ti in T_arr]), 1e-6)  # (n,)

    brier_scores = []
    for j, t in enumerate(times):
        S_t = surv_func.iloc[j].values         # (n,) survival probability at time t
        G_t = float(kmf.predict(t))
        if G_t < 0.01:
            continue

        # Vectorised IPCW Brier score
        mask_event = (T_arr <= t) & (E_arr == 1)   # (n,)
        mask_alive = T_arr > t                      # (n,)

        bs   = 0.0
        wsum = 0.0
        if mask_event.any():
            w_ev  = 1.0 / G_Ti[mask_event]
            bs   += float((w_ev * S_t[mask_event] ** 2).sum())
            wsum += float(w_ev.sum())
        if mask_alive.any():
            w_al  = 1.0 / G_t
            bs   += float(w_al * ((1.0 - S_t[mask_alive]) ** 2).sum())
            wsum += float(w_al * mask_alive.sum())

        if wsum > 0:
            brier_scores.append(bs / wsum)

    if len(brier_scores) < 2:
        return np.nan
    valid_times = times[:len(brier_scores)]
    try:
        ibs = np.trapezoid(brier_scores, valid_times) / (valid_times[-1] - valid_times[0])
    except AttributeError:
        ibs = np.trapz(brier_scores, valid_times) / (valid_times[-1] - valid_times[0])
    return ibs


# ========================== PER-MODALITY PCA ==========================

def _fit_per_modality_pca(modality_chunks, boot_idx=None):
    """Fit PCA per modality half; optionally on a bootstrap subsample."""
    all_boot_pcs, all_orig_pcs, all_var = [], [], []
    for matrix, k in modality_chunks:
        pca = PCA(n_components=k, random_state=42)
        if boot_idx is not None:
            boot_pcs = pca.fit_transform(matrix[boot_idx])
            orig_pcs = pca.transform(matrix)
            all_boot_pcs.append(boot_pcs)
            all_orig_pcs.append(orig_pcs)
        else:
            pcs = pca.fit_transform(matrix)
            all_orig_pcs.append(pcs)
        all_var.append(pca.explained_variance_ratio_)

    if boot_idx is not None:
        return np.hstack(all_boot_pcs), np.hstack(all_orig_pcs), all_var
    return np.hstack(all_orig_pcs), all_var


# ========================== FIT ONE ARM (helper) ==========================

def _fit_arm_on_dataset(
    boot_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    penalizer: float = PENALIZER,
) -> dict:
    """Fit a Cox model on boot_df (pre-sliced); evaluate on orig_df (pre-sliced).

    Callers are responsible for passing DataFrames that already contain only the
    relevant columns (Time, Event, covariates). No further column slicing is done
    here — this avoids the ambiguity of passing arm_cols separately.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(boot_df, duration_col='Time', event_col='Event')
    app_c = float(cph.concordance_index_)

    risk = cph.predict_partial_hazard(orig_df.drop(columns=['Time', 'Event']))
    test_c = float(concordance_index(orig_df['Time'], -risk, orig_df['Event']))

    app_ibs  = compute_ibs(cph, boot_df, boot_df)
    # IPCW censoring KM must be estimated on the evaluation set (Graf et al. 1999;
    # Gerds & Schumacher 2006). Using boot_df here would misweight orig_df patients.
    test_ibs = compute_ibs(cph, orig_df, orig_df)

    beta      = cph.params_.values.copy()                     # shape (p,)
    cov_names = list(cph.params_.index)
    try:
        var_mat = cph.variance_matrix_.values.copy()          # (p, p)
    except Exception:
        var_mat = np.diag(np.ones(len(beta)) * np.nan)

    return {
        'app_c':     app_c,
        'test_c':    test_c,
        'app_ibs':   float(app_ibs)  if np.isfinite(app_ibs)  else np.nan,
        'test_ibs':  float(test_ibs) if np.isfinite(test_ibs) else np.nan,
        'log_lik':   float(cph.log_likelihood_),
        'beta':      beta,
        'var_mat':   var_mat,
        'cov_names': cov_names,
    }


def _rubin_pool(per_m: List[dict]) -> dict:
    """Rubin's combining rules across M imputed-dataset results (scalar metrics).

    References
    ----------
    Rubin DB (1987) Multiple Imputation for Nonresponse in Surveys, §3.1.
    Marshall A et al. (2009) BMC Med Res Methodol 9:57.
    """
    M = len(per_m)

    # ── Scalar averages ───────────────────────────────────────────────────────
    # C-index pooled on logit scale then back-transformed (Marshall et al. 2009;
    # Wood et al. 2015) to avoid boundary effects and satisfy Rubin's normality
    # assumption. Report the untransformed value; difference from arithmetic mean
    # is typically < 0.001 at C ≈ 0.55–0.70.
    def _logit_pool(vals):
        clipped = np.clip(vals, 1e-6, 1.0 - 1e-6)
        return float(1.0 / (1.0 + np.exp(-np.mean(np.log(clipped / (1.0 - clipped))))))
    app_c    = _logit_pool([r['app_c']  for r in per_m])
    test_c   = _logit_pool([r['test_c'] for r in per_m])
    app_ibs  = float(np.nanmean([r['app_ibs']  for r in per_m]))
    test_ibs = float(np.nanmean([r['test_ibs'] for r in per_m]))

    # ── Coefficient-level Rubin's rules ───────────────────────────────────────
    betas    = [r['beta'] for r in per_m]              # list of (p,) arrays
    var_mats = [r['var_mat'] for r in per_m]           # list of (p,p) arrays
    cov_names = per_m[0]['cov_names']

    beta_bar = np.mean(betas, axis=0)                  # (p,)

    # Within-imputation variance: W = (1/M) sum Var_m
    W_bar = np.mean(var_mats, axis=0)                  # (p,p)

    # Between-imputation variance: B = 1/(M-1) sum (beta_m - beta_bar)(...)^T
    # Guard M=1: no between-imputation variance (single imputation, sensitivity run)
    diffs = np.stack([b - beta_bar for b in betas], axis=0)  # (M, p)
    if M > 1:
        B = (1.0 / (M - 1)) * (diffs.T @ diffs)              # (p,p)
    else:
        B = np.zeros_like(W_bar)                              # M=1 → no between-imp variance

    # Total variance: T = W + (1 + 1/M) * B
    T = W_bar + (1.0 + 1.0 / M) * B                         # (p,p)

    # FMI per coefficient (diagonal): (B_diag + B_diag/M) / T_diag
    W_diag = np.diag(W_bar)
    B_diag = np.diag(B)
    T_diag = np.diag(T)
    fmi = np.where(T_diag > 0, (B_diag + B_diag / M) / T_diag, np.nan)

    # B/W ratio (sanity check: should be > 0.01 if MI is doing useful work)
    bw_ratio = np.where(W_diag > 0, B_diag / W_diag, np.nan)

    return {
        'app_c':     app_c,
        'test_c':    test_c,
        'optimism_c': app_c - test_c,
        'app_ibs':   app_ibs,
        'test_ibs':  test_ibs,
        'ibs_optimism': test_ibs - app_ibs if np.isfinite(app_ibs) and np.isfinite(test_ibs) else np.nan,
        'beta':      beta_bar,
        'T_diag':    T_diag,
        'fmi':       fmi,
        'bw_ratio':  bw_ratio,
        'cov_names': cov_names,
        'log_liks':  [r['log_lik'] for r in per_m],   # needed for D2
    }


# ========================== SINGLE BOOTSTRAP ITERATION (MI) ==========================

def _single_bootstrap_mice(
    b: int,
    raw_df: pd.DataFrame,
    complete_df_list: List[pd.DataFrame],    # M pre-imputed originals (fixed)
    baseline_cols: List[str],
    structured_cols: List[str],
    embedding_arms_data: Dict[str, Tuple],
    n_patients: int,
    random_state: int,
    pca_k: int,
    impute_cols: List[str],
    llm_features: List[str],
    M: int = M_IMPUTATIONS,
):
    """
    One bootstrap iteration with MI-inside-bootstrap (Val-MI protocol).

    Protocol (Wahl et al. 2016, §2):
      1. Draw boot_idx from raw_df (pre-imputation rows preserved).
      2. Impute boot_df_raw M times → boot_df_list.
      3. For each m: fit all arms on boot_df_list[m], test on complete_df_list[m].
      4. Pool M per-imputation results with Rubin's rules.
      5. Return one pooled row per arm.

    The paired m↔m' evaluation (same imputation index for train and test)
    avoids cross-pairing bias. complete_df_list is pre-computed once outside
    the bootstrap loop and passed in.

    References
    ----------
    Wahl S et al. (2016) BMC Med Res Methodol 16:144.
    Musoro JZ et al. (2014) BMC Med Res Methodol 14:116.
    """
    warnings.filterwarnings('ignore')
    # Use default_rng for the bootstrap index (modern API, avoids legacy seed collisions)
    rng_boot  = np.random.default_rng(random_state + b)
    boot_idx  = rng_boot.integers(0, n_patients, size=n_patients)
    # Imputation seeds are in a non-overlapping band (offset by 10000) to avoid
    # collisions with the top-of-function seed range (random_state + 0..B-1).
    seed_boot = int(rng_boot.integers(10_000, 1_000_000))

    try:
        boot_df_raw  = raw_df.iloc[boot_idx].reset_index(drop=True)
        boot_df_list = mice_impute_multiple(boot_df_raw, impute_cols, M=M,
                                             random_state=seed_boot)

        baseline_feat = [c for c in baseline_cols if c not in ('Time', 'Event')]
        results       = {}

        # ── Pre-compute bootstrap PCA (same boot_idx for all m) ─────────────
        boot_pca_dict = {}
        orig_pca_dict = {}
        for arm_name, (modality_chunks, pca_col_names) in embedding_arms_data.items():
            bp, op, _ = _fit_per_modality_pca(modality_chunks, boot_idx=boot_idx)
            boot_pca_dict[arm_name] = (bp, pca_col_names)
            orig_pca_dict[arm_name] = (op, pca_col_names)

        # ── Per-arm MI loop ──────────────────────────────────────────────────
        arm_per_m: Dict[str, List[dict]] = {
            'baseline':       [],
            'ArmA_Structured': [],
            **{a: [] for a in embedding_arms_data},
        }

        for m in range(M):
            bd = boot_df_list[m]                  # imputed bootstrap dataset
            od = complete_df_list[m]              # paired imputed original

            # -- baseline --
            base_res = _fit_arm_on_dataset(bd[baseline_cols], od[baseline_cols])
            base_ll  = base_res['log_lik']
            arm_per_m['baseline'].append(base_res)

            # -- Arm A (structured) --
            struct_res = _fit_arm_on_dataset(bd[structured_cols], od[structured_cols])
            lrt_a_m    = 2.0 * (struct_res['log_lik'] - base_ll)
            struct_res['lrt_stat'] = lrt_a_m
            arm_per_m['ArmA_Structured'].append(struct_res)

            # -- embedding arms --
            for arm_name, (modality_chunks, pca_col_names) in embedding_arms_data.items():
                total_k = sum(k for _, k in modality_chunks)
                bp = boot_pca_dict[arm_name][0]   # (N_boot, total_k)
                op = orig_pca_dict[arm_name][0]   # (N_orig, total_k)

                boot_emb = bd[['Time','Event'] + baseline_feat].copy()
                orig_emb = od[['Time','Event'] + baseline_feat].copy()
                for j in range(total_k):
                    boot_emb[pca_col_names[j]] = bp[:, j]
                    orig_emb[pca_col_names[j]] = op[:, j]

                emb_cols = ['Time','Event'] + baseline_feat + pca_col_names
                emb_res  = _fit_arm_on_dataset(boot_emb[emb_cols], orig_emb[emb_cols])
                lrt_e_m  = 2.0 * (emb_res['log_lik'] - base_ll)
                emb_res['lrt_stat'] = lrt_e_m
                arm_per_m[arm_name].append(emb_res)

        # ── Rubin's rules within bootstrap ───────────────────────────────────
        for arm_name, per_m_list in arm_per_m.items():
            if not per_m_list:
                continue
            pooled = _rubin_pool(per_m_list)

            # Meng-Rubin D2 for LRT
            if arm_name != 'baseline':
                lrt_stats = [r['lrt_stat'] for r in per_m_list if np.isfinite(r.get('lrt_stat', np.nan))]
                df_lrt    = len(llm_features) if arm_name == 'ArmA_Structured' else pca_k
                D2, lrt_p, r2 = meng_rubin_d2(lrt_stats, k=df_lrt, M=M)
                pooled['lrt_stat'] = D2
                pooled['lrt_p']    = lrt_p
                pooled['r2_mi']    = r2

            results[arm_name] = pooled

        return results

    except Exception as e:
        logger.debug(f"Bootstrap {b} failed: {e}")
        return None


# ========================== APPARENT C FROM M IMPUTATIONS ==========================

def _apparent_from_mi(
    orig_df_list: List[pd.DataFrame],
    llm_features: List[str],
    embedding_arms_data: Dict,
    baseline_cols: List[str],
    structured_cols: List[str],
    pca_k: int,
    M: int = M_IMPUTATIONS,
) -> dict:
    """Compute apparent (full-cohort) metrics pooled over M imputations.

    Returns dict of {arm_name: pooled_dict} with same keys as _rubin_pool output
    plus 'full_cohort_lrt_p' (Meng-Rubin D2 on original data).
    """
    # Build arm_per_m from known arms (arm_cols_map may have empty embedding entries)
    arm_per_m: Dict[str, List[dict]] = {'baseline': [], 'ArmA_Structured': []}
    arm_per_m.update({a: [] for a in embedding_arms_data})

    baseline_feat = [c for c in baseline_cols if c not in ('Time','Event')]

    # Pre-compute PCA for each embedding arm ONCE (embeddings do not depend on m)
    pca_full: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for arm_name, (modality_chunks, pca_col_names) in embedding_arms_data.items():
        comps, _ = _fit_per_modality_pca(modality_chunks)   # fit on full unimputed cohort
        pca_full[arm_name] = (comps, pca_col_names)

    for m, od in enumerate(orig_df_list):
        base_res = _fit_arm_on_dataset(od[baseline_cols], od[baseline_cols])
        base_ll  = base_res['log_lik']
        arm_per_m['baseline'].append(base_res)

        struct_res = _fit_arm_on_dataset(od[structured_cols], od[structured_cols])
        lrt_a      = 2.0 * (struct_res['log_lik'] - base_ll)
        struct_res['lrt_stat'] = lrt_a
        arm_per_m['ArmA_Structured'].append(struct_res)

        for arm_name, (modality_chunks, pca_col_names) in embedding_arms_data.items():
            total_k   = sum(k for _, k in modality_chunks)
            pca_comps = pca_full[arm_name][0]   # pre-computed; does not change with m
            emb_df    = od[['Time','Event'] + baseline_feat].copy()
            for j in range(total_k):
                emb_df[pca_col_names[j]] = pca_comps[:, j]
            emb_cols = ['Time','Event'] + baseline_feat + pca_col_names
            emb_res  = _fit_arm_on_dataset(emb_df[emb_cols], emb_df[emb_cols])
            lrt_e    = 2.0 * (emb_res['log_lik'] - base_ll)
            emb_res['lrt_stat'] = lrt_e
            arm_per_m[arm_name].append(emb_res)

    apparent = {}
    for arm_name, per_m_list in arm_per_m.items():
        pooled = _rubin_pool(per_m_list)
        if arm_name != 'baseline':
            lrt_stats = [r['lrt_stat'] for r in per_m_list if np.isfinite(r.get('lrt_stat',np.nan))]
            df_lrt    = len(llm_features) if arm_name == 'ArmA_Structured' else pca_k
            D2, lrt_p, r2 = meng_rubin_d2(lrt_stats, k=df_lrt, M=M)
            pooled['full_cohort_lrt_D2'] = D2
            pooled['full_cohort_lrt_p']  = lrt_p
            pooled['full_cohort_r2_mi']  = r2
        apparent[arm_name] = pooled

    return apparent


# ========================== MAIN ANALYSIS ==========================

def run_analysis_for_outcome(
    outcome_key: str,
    outcome_config: dict,
    df: pd.DataFrame,
    all_embeddings: Dict[str, Dict[str, np.ndarray]],
    n_bootstrap: int = 1000,
    n_jobs: int = -1,
    random_state: int = 42,
    analysis_mode: str = 'full_mice',
    subgroup: str = 'all',
    pca_mode: str = 'per_modality',
    M: int = M_IMPUTATIONS,
) -> Optional[dict]:
    """Run MI + bootstrap comparison for one survival outcome."""

    time_col  = outcome_config['time']
    event_col = outcome_config['event']

    print(f"\n{'='*70}")
    print(f"OUTCOME: {outcome_key}  (time={time_col}, event={event_col})")
    print(f"  mode={analysis_mode}  subgroup={subgroup}  M={M}  B={n_bootstrap}")
    print(f"{'='*70}")

    if time_col not in df.columns or event_col not in df.columns:
        print(f"  Skipping: columns not found"); return None

    llm_features = MODALITY_FEATURES.get(subgroup, LLM_FEATURES)
    pca_k        = len(llm_features)

    # ── Build analysis cohort ─────────────────────────────────────────────────
    all_cols    = [time_col, event_col] + BASELINE_FEATURES + llm_features
    analysis_df = df[['patient_id'] + [c for c in all_cols if c in df.columns]].copy()
    analysis_df = analysis_df.rename(columns={time_col: 'Time', event_col: 'Event'})

    if analysis_mode == 'complete_case':
        analysis_df = analysis_df.dropna(subset=['Time','Event'] + llm_features)
        impute_cols = BASELINE_FEATURES
    else:  # full_mice
        analysis_df = analysis_df.dropna(subset=['Time','Event'])
        impute_cols = BASELINE_FEATURES + llm_features

    analysis_df = analysis_df[(analysis_df['Time'] > 0) &
                              (analysis_df['Time'] <= MAX_FOLLOWUP_MONTHS) &
                              analysis_df['Event'].isin([0, 1])]

    n_before  = len(analysis_df)
    n_missing = analysis_df[impute_cols].isna().any(axis=1).sum()
    print(f"  Before imputation ({analysis_mode}): N={n_before}, missing any={n_missing}")
    for col in impute_cols:
        nmiss = analysis_df[col].isna().sum()
        if nmiss > 0:
            print(f"    {col}: {nmiss} missing ({nmiss/n_before*100:.1f}%)")

    # raw_df: pre-imputation (NaN preserved) — used for bootstrap resampling
    raw_df = analysis_df.reset_index(drop=True)

    # ── Subgroup filtering ────────────────────────────────────────────────────
    pids = raw_df['patient_id'].values
    if subgroup in ('both_reports', 'consult_only', 'path_only'):
        first_dual_dict = next(
            (emb for arm, emb in all_embeddings.items()
             if 'dual' in EMBEDDING_ARMS.get(arm, '')), None)
        if first_dual_dict is None:
            print("  No dual arm; falling back to subgroup=all"); subgroup = 'all'
        else:
            emb_dim = next(iter(first_dual_dict.values())).shape[0]
            half    = emb_dim // 2
            def _has_path(pid):
                return pid in first_dual_dict and np.any(first_dual_dict[pid][:half] != 0)
            def _has_consult(pid):
                return pid in first_dual_dict and np.any(first_dual_dict[pid][half:] != 0)
            if subgroup == 'both_reports':
                keep = {p for p in pids if _has_path(p) and _has_consult(p)}
            elif subgroup == 'consult_only':
                keep = {p for p in pids if _has_consult(p)}
            else:
                keep = {p for p in pids if _has_path(p)}
            mask   = np.array([p in keep for p in pids])
            raw_df = raw_df[mask].reset_index(drop=True)
            pids   = raw_df['patient_id'].values
            print(f"  After subgroup filter: N={len(raw_df)}")

    n_patients = len(raw_df)
    n_events   = int(raw_df['Event'].sum())

    if n_patients < 50 or n_events < 10:
        print("  Skipping: insufficient data"); return None

    # ── Missingness indicators (subgroup=all, per_modality only) ─────────────
    use_indicators = (pca_mode == 'per_modality' and subgroup == 'all')
    indicator_cols = []
    if use_indicators:
        first_dual_dict2 = next(
            (emb for arm, emb in all_embeddings.items()
             if 'dual' in EMBEDDING_ARMS.get(arm, '')), None)
        if first_dual_dict2 is not None:
            edim2  = next(iter(first_dual_dict2.values())).shape[0]
            half2  = edim2 // 2
            hp_arr = np.array([1.0 if (p in first_dual_dict2 and
                                       np.any(first_dual_dict2[p][:half2] != 0)) else 0.0
                                for p in pids])
            hc_arr = np.array([1.0 if (p in first_dual_dict2 and
                                       np.any(first_dual_dict2[p][half2:] != 0)) else 0.0
                                for p in pids])
            raw_df['has_pathology']    = hp_arr
            raw_df['has_consultation'] = hc_arr
            indicator_cols = list(MISSINGNESS_INDICATORS)
            print(f"  Indicators: has_path={int(hp_arr.sum())}, "
                  f"has_consult={int(hc_arr.sum())}")

    # ── Build embedding matrices (mean-impute missing modality blocks) ────────
    modality_pca_k     = MODALITY_PCA_K.get(subgroup, {'path': PCA_K_PATH, 'consult': PCA_K_CONSULT})
    embedding_arms_data = {}

    for arm_name, emb_dict in all_embeddings.items():
        available = [p for p in pids if p in emb_dict]
        if len(available) < n_patients * 0.5:
            continue
        emb_dim_full = next(iter(emb_dict.values())).shape[0]
        is_dual      = 'dual' in EMBEDDING_ARMS.get(arm_name, '')

        emb_matrix = np.zeros((n_patients, emb_dim_full), dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in emb_dict:
                emb_matrix[i] = emb_dict[pid]

        modality_chunks = []
        pca_col_names   = []

        if pca_mode == 'per_modality' and is_dual:
            half          = emb_dim_full // 2
            path_mat      = emb_matrix[:, :half].copy()
            consult_mat   = emb_matrix[:, half:].copy()

            if subgroup == 'all':
                hp = np.any(path_mat    != 0, axis=1)
                hc = np.any(consult_mat != 0, axis=1)
                if hp.any()  and (~hp).sum()  > 0: path_mat[~hp]    = path_mat[hp].mean(axis=0)
                if hc.any()  and (~hc).sum()  > 0: consult_mat[~hc] = consult_mat[hc].mean(axis=0)
                n_imp_p = int((~hp).sum()); n_imp_c = int((~hc).sum())
                if n_imp_p > 0 or n_imp_c > 0:
                    print(f"  {arm_name}: mean-imputed {n_imp_p} path, {n_imp_c} consult blocks")

            if 'path' in modality_pca_k:
                kp = modality_pca_k['path']
                modality_chunks.append((path_mat, kp))
                pca_col_names += [f'{arm_name}_path_PC{j+1}' for j in range(kp)]
            if 'consult' in modality_pca_k:
                kc = modality_pca_k['consult']
                modality_chunks.append((consult_mat, kc))
                pca_col_names += [f'{arm_name}_consult_PC{j+1}' for j in range(kc)]
        else:
            modality_chunks = [(emb_matrix, pca_k)]
            pca_col_names   = [f'{arm_name}_PC{j+1}' for j in range(pca_k)]

        total_k = sum(k for _, k in modality_chunks)
        embedding_arms_data[arm_name] = (modality_chunks, pca_col_names)
        print(f"  {arm_name}: {len(available)} pts, total_PCs={total_k}")

    baseline_cols    = ['Time','Event'] + BASELINE_FEATURES + indicator_cols
    structured_cols  = ['Time','Event'] + BASELINE_FEATURES + indicator_cols + llm_features
    baseline_feat    = BASELINE_FEATURES + indicator_cols

    all_feat_cols    = BASELINE_FEATURES + indicator_cols + llm_features
    # raw_analysis_data for bootstrap (contains raw NaNs + indicators)
    raw_analysis     = raw_df[['Time','Event'] + all_feat_cols].copy()

    # ── M imputations of original data (pre-computed, used as test in bootstrap) ─
    print(f"\n  Pre-computing M={M} imputations of original cohort …")
    orig_df_list = mice_impute_multiple(raw_analysis, impute_cols, M=M,
                                        random_state=random_state)
    # Attach indicators to each (they are not imputed, just fixed)
    if indicator_cols:
        for od in orig_df_list:
            od['has_pathology']    = raw_df['has_pathology'].values
            od['has_consultation'] = raw_df['has_consultation'].values

    # ── Apparent (full-cohort) performance from M imputations ────────────────
    print(f"  Computing apparent performance over M={M} imputations …")
    apparent = _apparent_from_mi(
        orig_df_list, llm_features,
        embedding_arms_data, baseline_cols, structured_cols, pca_k, M=M,
    )

    print(f"\n  Baseline:  apparent C = {apparent['baseline']['app_c']:.4f}")
    for arm in ['ArmA_Structured'] + list(embedding_arms_data.keys()):
        if arm in apparent:
            dc  = apparent[arm]['app_c'] - apparent['baseline']['app_c']
            lp  = apparent[arm].get('full_cohort_lrt_p', np.nan)
            ibs = apparent[arm].get('app_ibs', np.nan)
            sig = ' *' if np.isfinite(lp) and lp < 0.05 else ''
            print(f"  {arm:<26} C={apparent[arm]['app_c']:.4f}  ΔC={dc:+.4f}"
                  f"  IBS={ibs:.4f}  LRT(D2) p={lp:.4f}{sig}")
        if arm == 'ArmA_Structured' and arm in apparent:
            for feat, coef, fmi in zip(
                    apparent[arm]['cov_names'],
                    apparent[arm]['beta'],
                    apparent[arm]['fmi']):
                if feat in llm_features:
                    print(f"    {feat}: coef={coef:.3f}  FMI={fmi:.3f}")

    # ── Bootstrap optimism correction (Val-MI) ────────────────────────────────
    print(f"\n  Running B={n_bootstrap} × M={M} bootstrap–MI iterations …")

    with tqdm_joblib(tqdm(total=n_bootstrap, desc=f"  {outcome_key}", ncols=80)):
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(_single_bootstrap_mice)(
                b, raw_analysis, orig_df_list,
                baseline_cols, structured_cols,
                embedding_arms_data, n_patients, random_state,
                pca_k, impute_cols, llm_features, M,
            )
            for b in range(n_bootstrap)
        )

    successful = [r for r in boot_results if r is not None]
    n_ok = len(successful)
    print(f"  Bootstrap success: {n_ok}/{n_bootstrap}")
    if n_ok < n_bootstrap * 0.5:
        print("  WARNING: >50% iterations failed — results unreliable")

    # ── Aggregate across B bootstraps ─────────────────────────────────────────
    all_arms = (['baseline', 'ArmA_Structured'] + list(embedding_arms_data.keys()))

    result_row = {
        'Outcome':             outcome_key,
        'N_patients':          n_patients,
        'N_events':            n_events,
        'N_bootstrap_success': n_ok,
        'M_imputations':       M,
        'pca_mode':            pca_mode,
        'subgroup':            subgroup,
        'n_baseline_vars':     len(baseline_feat),
        'missingness_indicators': ','.join(indicator_cols) if indicator_cols else 'none',
        'analysis_mode':       analysis_mode,
    }

    # Storage for detailed bootstrap arrays (saved in pkl)
    boot_arrays: Dict[str, dict] = {a: {} for a in all_arms}

    for arm in all_arms:
        arm_results = [r[arm] for r in successful if arm in r]
        if not arm_results:
            continue

        app_cs    = np.array([r['app_c']    for r in arm_results])
        test_cs   = np.array([r['test_c']   for r in arm_results])
        opt_cs    = np.array([r['optimism_c'] for r in arm_results])
        test_ibss = np.array([r['test_ibs'] for r in arm_results])
        app_ibss  = np.array([r['app_ibs']  for r in arm_results])

        # Harrell (1996): corrected C = apparent-on-full-cohort − mean(bootstrap optimism)
        # Use full-cohort apparent (from M pre-imputed originals, computed before the loop),
        # NOT the mean of bootstrap apparent_c values, which is a biased estimator.
        app_c_full    = apparent[arm]['app_c'] if arm in apparent else float(np.mean(app_cs))
        mean_optimism = float(np.mean(opt_cs))
        corrected_c   = app_c_full - mean_optimism

        # 95% CI on the optimism-corrected estimate (Harrell 1996; Steyerberg 2019 §17.3.2):
        # CI = percentile of (app_c_full − opt_c_b) across B resamples.
        # This is NOT percentile(test_cs), which omits the apparent-vs-test gap variability.
        corrected_boot = app_c_full - opt_cs
        ci_lo = float(np.percentile(corrected_boot, 2.5))
        ci_hi = float(np.percentile(corrected_boot, 97.5))

        result_row[f'{arm}_apparent_C']           = app_c_full
        result_row[f'{arm}_corrected_C']          = corrected_c
        result_row[f'{arm}_C_CI_lower']           = ci_lo
        result_row[f'{arm}_C_CI_upper']           = ci_hi
        result_row[f'{arm}_mean_optimism_C']      = mean_optimism
        result_row[f'{arm}_bootstrap_mean_app_C'] = float(np.mean(app_cs))   # diagnostic only

        # IBS: mirror the C-index treatment (Harrell 1996 applied to IBS).
        # CI = percentile of (app_ibs_full + opt_ibs_b) across B, not percentile(test_ibss).
        # IBS optimism = test_ibs − app_ibs (positive = test IBS worse than apparent IBS);
        # corrected IBS = apparent_full + mean(optimism_b).
        if arm in apparent and np.isfinite(apparent[arm].get('app_ibs', np.nan)):
            app_ibs_full  = apparent[arm]['app_ibs']
            opt_ibss      = np.array([r['ibs_optimism'] for r in arm_results])
            valid_mask    = np.isfinite(opt_ibss)
            if valid_mask.sum() > 5:
                mean_opt_ibs      = float(np.mean(opt_ibss[valid_mask]))
                corrected_ibs     = app_ibs_full + mean_opt_ibs
                corrected_boot_ibs = app_ibs_full + opt_ibss[valid_mask]
                ibs_ci_lo = float(np.percentile(corrected_boot_ibs, 2.5))
                ibs_ci_hi = float(np.percentile(corrected_boot_ibs, 97.5))
            else:
                corrected_ibs = app_ibs_full
                ibs_ci_lo = ibs_ci_hi = np.nan
            result_row[f'{arm}_apparent_IBS']  = app_ibs_full
            result_row[f'{arm}_corrected_IBS'] = corrected_ibs
            result_row[f'{arm}_IBS_CI_lower']  = ibs_ci_lo
            result_row[f'{arm}_IBS_CI_upper']  = ibs_ci_hi

        # Full-cohort Meng-Rubin D2 LRT (reported as main p-value)
        if arm != 'baseline' and arm in apparent:
            result_row[f'{arm}_fullcohort_LRT_D2']  = apparent[arm].get('full_cohort_lrt_D2', np.nan)
            result_row[f'{arm}_fullcohort_LRT_p']   = apparent[arm].get('full_cohort_lrt_p',  np.nan)
            result_row[f'{arm}_fullcohort_r2_mi']   = apparent[arm].get('full_cohort_r2_mi',  np.nan)

        # Bootstrap LRT significance rate (D2 p<0.05 in each bootstrap)
        if arm != 'baseline':
            lrt_ps = [r['lrt_p'] for r in arm_results if 'lrt_p' in r and np.isfinite(r['lrt_p'])]
            if lrt_ps:
                result_row[f'{arm}_bootstrap_LRT_sig_pct'] = float(np.mean([p < 0.05 for p in lrt_ps]) * 100)

        # ΔC vs baseline — all quantities use full-cohort apparent for consistency
        if arm != 'baseline':
            base_opt_cs   = np.array([s['baseline']['optimism_c'] for s in successful if 'baseline' in s])
            app_base_full = apparent['baseline']['app_c']
            n_paired      = min(len(base_opt_cs), len(opt_cs))
            if n_paired >= 50:
                corrected_base  = app_base_full - float(np.mean(base_opt_cs[:n_paired]))
                corrected_delta = corrected_c - corrected_base   # both use full-cohort apparent
                # Per-bootstrap ΔC for CI (Steyerberg 2019, §17.3.2 applied to difference)
                boot_deltas = (app_c_full - opt_cs[:n_paired]) - (app_base_full - base_opt_cs[:n_paired])
                result_row[f'{arm}_delta_C']          = float(corrected_delta)
                result_row[f'{arm}_delta_C_CI_lower'] = float(np.percentile(boot_deltas, 2.5))
                result_row[f'{arm}_delta_C_CI_upper'] = float(np.percentile(boot_deltas, 97.5))
                boot_arrays[arm]['delta_c'] = boot_deltas

        # Store arrays for pkl
        boot_arrays[arm]['test_c']  = test_cs
        boot_arrays[arm]['app_c']   = app_cs
        boot_arrays[arm]['test_ibs']= test_ibss

        # FMI and B/W ratio for LLM coefficients (Arm A only, sanity check)
        if arm == 'ArmA_Structured' and arm in apparent:
            fmi_vals = apparent[arm]['fmi']
            bw_vals  = apparent[arm]['bw_ratio']
            cnames   = apparent[arm]['cov_names']
            for feat in llm_features:
                if feat in cnames:
                    idx = cnames.index(feat)
                    result_row[f'ArmA_{feat}_FMI']     = float(fmi_vals[idx]) if idx < len(fmi_vals) else np.nan
                    result_row[f'ArmA_{feat}_BW_ratio'] = float(bw_vals[idx])  if idx < len(bw_vals)  else np.nan

    # ── AIC comparison: use mean log-likelihood over M imputations (not first only) ──
    n_bv = len(baseline_feat)
    aic_table = {}
    if 'baseline' in apparent:
        aic_table['baseline'] = -2 * float(np.mean(apparent['baseline']['log_liks'])) + 2 * n_bv
    if 'ArmA_Structured' in apparent:
        aic_table['ArmA_Structured'] = -2 * float(np.mean(apparent['ArmA_Structured']['log_liks'])) + 2 * (n_bv + len(llm_features))
    for arm_name, (modality_chunks_a, _) in embedding_arms_data.items():
        if arm_name in apparent:
            total_k = sum(k for _, k in modality_chunks_a)
            aic_table[arm_name] = -2 * float(np.mean(apparent[arm_name]['log_liks'])) + 2 * (n_bv + total_k)
    result_row['best_AIC_arm'] = min(aic_table, key=aic_table.get) if aic_table else 'NA'
    for an, aic_v in aic_table.items():
        result_row[f'{an}_AIC'] = float(aic_v)

    # ── Pairwise bootstrap ΔC ─────────────────────────────────────────────────
    non_base = [a for a in all_arms if a != 'baseline']
    for i, ai in enumerate(non_base):
        for aj in non_base[i+1:]:
            diffs = [r[ai]['test_c'] - r[aj]['test_c']
                     for r in successful if ai in r and aj in r]
            if len(diffs) >= 50:
                md  = float(np.mean(diffs))
                clo = float(np.percentile(diffs, 2.5))
                chi = float(np.percentile(diffs, 97.5))
                sig = '*' if clo > 0 or chi < 0 else ''
                result_row[f'pairwise_{ai}_vs_{aj}_deltaC']   = md
                result_row[f'pairwise_{ai}_vs_{aj}_CI_lower'] = clo
                result_row[f'pairwise_{ai}_vs_{aj}_CI_upper'] = chi
                result_row[f'pairwise_{ai}_vs_{aj}_sig']      = sig

    # ── Win-count ─────────────────────────────────────────────────────────────
    win_counts = {a: 0 for a in non_base}
    n_valid = 0
    for r in successful:
        cs = {a: r[a]['test_c'] for a in non_base if a in r}
        if cs:
            win_counts[max(cs, key=cs.get)] += 1
            n_valid += 1
    for a, cnt in win_counts.items():
        result_row[f'{a}_win_count'] = cnt
        result_row[f'{a}_win_pct']   = cnt / n_valid * 100 if n_valid else 0
    result_row['bootstrap_win_count_N'] = n_valid

    # ── Sanity check: B/W ratio for LLM coefficients ─────────────────────────
    print(f"\n  ── MI Sanity Check (Arm A, LLM features) ──")
    for feat in llm_features:
        fmi_k = f'ArmA_{feat}_FMI'
        bw_k  = f'ArmA_{feat}_BW_ratio'
        if fmi_k in result_row:
            fmi_v = result_row[fmi_k]
            bw_v  = result_row[bw_k]
            flag  = '  ← WARNING: B/W<0.01, sample_posterior may not be working' if np.isfinite(bw_v) and bw_v < 0.01 else ''
            print(f"    {feat}: FMI={fmi_v:.3f}  B/W_ratio={bw_v:.4f}{flag}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  SUMMARY: {outcome_key} (N={n_patients}, Events={n_events}, M={M})")
    print(f"  {'='*70}")
    print(f"  {'Arm':<26} {'AppC':>8} {'CorrC':>8} {'95%CI':>18} {'ΔC':>8} {'D2-p':>8} {'LRT%':>6}")
    print(f"  {'-'*86}")
    for arm in all_arms:
        if f'{arm}_corrected_C' not in result_row: continue
        app  = result_row[f'{arm}_apparent_C']
        corr = result_row[f'{arm}_corrected_C']
        clo  = result_row[f'{arm}_C_CI_lower']
        chi  = result_row[f'{arm}_C_CI_upper']
        dc   = result_row.get(f'{arm}_delta_C', 0.0)
        lp   = result_row.get(f'{arm}_fullcohort_LRT_p', result_row.get(f'{arm}_original_LRT_p', ''))
        lrt_pct = result_row.get(f'{arm}_bootstrap_LRT_sig_pct', '')
        dc_s   = f'{dc:+.4f}' if arm != 'baseline' else '-'
        lp_s   = f'{lp:.4f}' if isinstance(lp, float) and np.isfinite(lp) else '-'
        sig_s  = ' *' if isinstance(lp, float) and np.isfinite(lp) and lp < 0.05 else ''
        lrt_s  = f'{lrt_pct:.1f}%' if isinstance(lrt_pct, float) else '-'
        print(f"  {arm:<26} {app:>8.4f} {corr:>8.4f} ({clo:.4f}-{chi:.4f}) {dc_s:>8} {lp_s:>8}{sig_s} {lrt_s:>6}")

    # Stash boot_arrays in result_row (popped before CSV save, kept in pkl)
    result_row['_boot_arrays'] = boot_arrays
    result_row['_apparent']    = apparent   # full Rubin-pooled apparent dicts

    return result_row


# ========================== SAVE + RUN ==========================

def _run_and_save(
    outcomes_to_run: dict,
    df: pd.DataFrame,
    all_embeddings: dict,
    analysis_mode: str,
    output_subdir: Path,
    n_bootstrap: int,
    n_jobs: int,
    random_state: int,
    subgroup: str = 'all',
    pca_mode: str = 'per_modality',
    M: int = M_IMPUTATIONS,
):
    output_subdir.mkdir(parents=True, exist_ok=True)
    txt_path = output_subdir / "embedding_comparison_report.txt"

    with TeeWriter(str(txt_path)):
        header = (
            f"Embedding vs. Structured Extraction — MI+Bootstrap Report\n"
            f"{'='*60}\n"
            f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Mode       : {analysis_mode}\n"
            f"M (MICE)   : {M}\n"
            f"B (boot)   : {n_bootstrap}\n"
            f"PCA mode   : {pca_mode}\n"
            f"Subgroup   : {subgroup}\n"
            f"Outcomes   : {', '.join(outcomes_to_run.keys())}\n"
            f"{'='*60}\n"
        )
        print(header)

        all_results = []
        all_pkl     = {}     # comprehensive per-outcome statistics for pkl

        for outcome_key, outcome_config in outcomes_to_run.items():
            try:
                result = run_analysis_for_outcome(
                    outcome_key, outcome_config, df, all_embeddings,
                    n_bootstrap=n_bootstrap, n_jobs=n_jobs,
                    random_state=random_state,
                    analysis_mode=analysis_mode,
                    subgroup=subgroup, pca_mode=pca_mode, M=M,
                )
                if result is None:
                    continue

                # Extract pkl payloads before they are dropped for CSV
                boot_arrays = result.pop('_boot_arrays', {})
                apparent    = result.pop('_apparent', {})

                result['analysis_mode'] = analysis_mode
                all_results.append(result)

                all_pkl[outcome_key] = {
                    'boot_arrays':   boot_arrays,
                    'apparent':      apparent,
                    'result_row':    {k: v for k, v in result.items() if not k.startswith('_')},
                }

            except Exception as e:
                logger.error(f"Error {outcome_key}: {e}", exc_info=True)

        if all_results:
            # ── CSV ──────────────────────────────────────────────────────────
            results_df = pd.DataFrame(all_results)
            csv_path   = output_subdir / "embedding_comparison_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"\nCSV saved: {csv_path}")

            # ── Comprehensive PKL ────────────────────────────────────────────
            pkl_path = output_subdir / "full_statistics.pkl"
            with open(pkl_path, 'wb') as f:
                import sklearn, lifelines as _lf
                pickle.dump({
                    'meta': {
                        'date':            time.strftime('%Y-%m-%d %H:%M:%S'),
                        'analysis_mode':   analysis_mode,
                        'M_imputations':   M,
                        'n_bootstrap':     n_bootstrap,
                        'pca_mode':        pca_mode,
                        'subgroup':        subgroup,
                        'random_state':    random_state,
                        'python':          sys.version,
                        'numpy_version':   np.__version__,
                        'sklearn_version': sklearn.__version__,
                        'lifelines_version': _lf.__version__,
                    },
                    'outcomes': all_pkl,
                }, f)
            print(f"PKL saved : {pkl_path}")
            print(f"\n  PKL contains per-outcome keys: {list(all_pkl.keys())}")
            print(f"  Per outcome: boot_arrays (test_c, app_c, test_ibs, delta_c per arm × B)")
            print(f"               apparent (Rubin-pooled full-cohort metrics incl. FMI, B/W, D2)")
            print(f"               result_row (all CSV columns as dict)")

            # ── Final summary ────────────────────────────────────────────────
            print(f"\n{'='*70}")
            print("FINAL SUMMARY ACROSS ALL OUTCOMES")
            print(f"{'='*70}")
            for _, row in results_df.iterrows():
                oc = row['Outcome']
                n  = int(row['N_patients'])
                ev = int(row['N_events'])
                print(f"\n{oc} (N={n}, Events={ev}, M={M}):")
                for arm in ['ArmA_Structured'] + list(EMBEDDING_ARMS.keys()):
                    if f'{arm}_corrected_C' in row and not pd.isna(row.get(f'{arm}_corrected_C', np.nan)):
                        dc   = row.get(f'{arm}_delta_C', np.nan)
                        dclo = row.get(f'{arm}_delta_C_CI_lower', np.nan)
                        dchi = row.get(f'{arm}_delta_C_CI_upper', np.nan)
                        lp   = row.get(f'{arm}_fullcohort_LRT_p', np.nan)
                        sig  = ' *' if np.isfinite(lp) and lp < 0.05 else ''
                        dc_s = f'ΔC={dc:+.4f} ({dclo:.4f},{dchi:.4f})' if np.isfinite(dc) else ''
                        lp_s = f'D2-p={lp:.4f}' if np.isfinite(lp) else ''
                        print(f"  {arm:<26} {dc_s}  {lp_s}{sig}")
        else:
            print("\nNo results produced.")
        print(f"\nDone. All outputs: {output_subdir}")

    print(f"Report: {txt_path}")


# ========================== MAIN ==========================

def main():
    global EMBEDDINGS_DIR, OUTCOMES_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Survival analysis: proper MICE (M=5) + Harrell bootstrap (B=1000)")
    parser.add_argument("--n_bootstrap",  type=int,   default=1000)
    parser.add_argument("--M",            type=int,   default=M_IMPUTATIONS,
                        help=f"Number of imputations (default {M_IMPUTATIONS})")
    parser.add_argument("--n_jobs",       type=int,   default=-1)
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--outcome",      type=str,   default="all",
                        choices=["all","OS","DFS","LFS","DistFS"])
    parser.add_argument("--analysis_mode",type=str,   default="full_mice",
                        choices=["complete_case","full_mice"])
    parser.add_argument("--subgroup",     type=str,   default="all",
                        choices=["all","both_reports","consult_only","path_only"])
    parser.add_argument("--pca_mode",     type=str,   default="per_modality",
                        choices=["per_modality","joint"])
    parser.add_argument("--arms",         type=str,   nargs="+", default=None)
    parser.add_argument("--output_dir",   type=str,   default=str(OUTPUT_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(EMBEDDINGS_DIR))
    parser.add_argument("--outcomes_path",  type=str, default=str(OUTCOMES_PATH))
    args = parser.parse_args()

    EMBEDDINGS_DIR = Path(args.embeddings_dir)
    OUTCOMES_PATH  = Path(args.outcomes_path)
    OUTPUT_DIR     = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"#  MICE BOOTSTRAP PIPELINE")
    print(f"#  M={args.M} imputations × B={args.n_bootstrap} bootstrap resamples")
    print(f"#  Protocol: Val-MI (Wahl 2016) — MI nested inside bootstrap")
    print(f"#  LRT: Meng-Rubin D2 F-test (Meng & Rubin 1992)")
    print(f"#  Output: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

    df             = load_outcomes()
    all_embeddings = load_embeddings()

    if args.arms:
        all_embeddings = {k: v for k, v in all_embeddings.items() if k in set(args.arms)}
        print(f"Restricted to arms: {list(all_embeddings.keys())}")

    outcomes_to_run = OUTCOMES if args.outcome == "all" else {args.outcome: OUTCOMES[args.outcome]}

    # Runtime estimate
    n_arms = len(all_embeddings) + 2   # +2 for baseline + ArmA
    est_h  = (args.n_bootstrap * args.M * n_arms * 0.002) / 3600
    print(f"Runtime estimate: ~{est_h:.1f}–{est_h*2:.1f} h "
          f"({args.n_bootstrap} boot × {args.M} imputations × {n_arms} arms)")

    if args.analysis_mode == 'full_mice':
        subdir = OUTPUT_DIR
    else:
        subdir = OUTPUT_DIR / args.analysis_mode

    _run_and_save(
        outcomes_to_run, df, all_embeddings,
        analysis_mode=args.analysis_mode,
        output_subdir=subdir,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        subgroup=args.subgroup,
        pca_mode=args.pca_mode,
        M=args.M,
    )

    print(f"\nAll done. Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
