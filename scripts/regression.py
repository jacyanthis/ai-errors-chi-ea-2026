#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import RandomEffects
from scipy import stats

# ----------------------------
# Data loading and preparation
# ----------------------------

def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, '..', 'data', 'data.csv')
    df = pd.read_csv(csv_path)
    df = df[df['feedback_finish'].notna() & (df['feedback_finish'] != '') & (df['feedback_finish'] != False)]
    # Exclude attention-check failures (correct answer is "Weekly")
    df = df[df['post_survey2_attention_check'] == 'Weekly']

    return df


def build_task_level_bids(df):
    rows = []
    for pid, row in df.iterrows():
        for col in (c for c in row.index if c.startswith('p2_t') and c.endswith('_bid')):
            if pd.notna(row[col]):
                task_num = int(col.split('_')[1][1:])  # p2_t{N}_bid -> N
                rows.append({'participant_id': pid, 'task': task_num, 'bid_amount': float(row[col])})
    return pd.DataFrame(rows)

def add_display_order(df_participants, bids_df):
    order_map = {}
    for pid, row in df_participants.iterrows():
        seq = json.loads(row['randomization'])['phase_2_diagram_order']
        for pos, task_idx in enumerate(seq, start=1):
            order_map[(pid, task_idx)] = pos
    bids_df['display_order'] = [order_map[(r.participant_id, r.task)] for r in bids_df.itertuples()]
    return bids_df

def encode_predictors(df):
    enc = df.copy()

    enc['monotonicity'] = enc['monotonicity'].astype(int)
    enc['pre_ee'] = enc[['pre_EE1','pre_EE2','pre_EE3']].astype(float).mean(axis=1)
    enc['pre_pe'] = enc[['pre_PE1','pre_PE2','pre_PE3','pre_PE4']].astype(float).mean(axis=1)

    # Gender: Man / Woman / Other, baseline = most frequent
    def map_gender(x):
        s = str(x).strip()
        if s == 'Man': return 'Man'
        if s == 'Woman': return 'Woman'
        if 'Other' in s: return 'Other'
        raise ValueError(f'{s} not a valid gender category')
    enc['gender_group'] = enc['gender'].apply(map_gender)
    g_baseline = enc['gender_group'].value_counts().idxmax()
    g_dum = pd.get_dummies(enc['gender_group'], prefix='gender', drop_first=False)
    for c in g_dum.columns:
        if c != f'gender_{g_baseline}':
            enc[c] = g_dum[c]

    def map_race(x):
        s = str(x).strip()
        if s in [
            'Hispanic or Latino (White or Caucasian)',
            'Hispanic or Latino (Black or African-American)',
            'Hispanic or Latino (any other race or multiple races)'
        ]: return 'Hispanic'
        if s == 'White or Caucasian': return 'White'
        if s == 'Black or African-American': return 'Black'
        if s == 'Asian or Pacific Islander': return 'Asian'
        if s == 'Native American, Alaska Native, or Aleutian': return 'Native_American'
        if s == 'Other (not Hispanic or Latino)': return 'Other'
        raise ValueError(f'{s} not a valid race category')
    enc['race_recoded'] = enc['race'].apply(map_race)
    
    # Pool low-count race categories (<10) into "Other"
    race_counts = enc['race_recoded'].value_counts()
    rare_race = set(race_counts[race_counts < 10].index)
    enc.loc[enc['race_recoded'].isin(rare_race), 'race_recoded'] = 'Other'
    
    r_dum = pd.get_dummies(enc['race_recoded'], prefix='race', drop_first=False)
    for c in r_dum.columns:
        if c != 'race_White':
            enc[c] = r_dum[c]

    def map_religion(x):
        s = str(x).strip()
        if s in ['Protestant','Mormon','Orthodox','Jewish','Muslim','Buddhist','Hindu','Atheist','Agnostic']:
            return s
        if s == 'Roman Catholic': return 'Roman_Catholic'
        if s == 'Nothing in particular': return 'Nothing'
        if 'Something else' in s: return 'Other'
        raise ValueError(f'{s} not a valid religion category')
    enc['religion_recoded'] = enc['religion'].apply(map_religion)
    
    # Pool low-count religion categories (<10) into "Other"
    rel_counts = enc['religion_recoded'].value_counts()
    rare_rel = set(rel_counts[rel_counts < 10].index)
    enc.loc[enc['religion_recoded'].isin(rare_rel), 'religion_recoded'] = 'Other'
    
    rel_dum = pd.get_dummies(enc['religion_recoded'], prefix='religion', drop_first=False)
    for c in rel_dum.columns:
        if c != 'religion_Protestant':
            enc[c] = rel_dum[c]

    enc['education_Post-grad'] = enc['education'].isin([
        "Master's degree (for example: MA, MS, MEng, MEd, MBA)",
        "Professional degree beyond bachelor's degree (for example: MD, DDS, DVM, LLB, JD)",
        'Doctorate degree (for example: PhD, EdD)'
    ]).astype(int)
    enc['education_Undergrad'] = enc['education'].isin([
        "Bachelor's degree (for example: BA, BS)"
    ]).astype(int)

    enc['income_$50-100k'] = enc['income'].isin(['$50,000-$74,999', '$75,000-$99,999']).astype(int)
    enc['income_>$100k'] = enc['income'].isin([
        '$100,000-$124,999', '$125,000-$149,999', '$150,000-$199,999', '$200,000-$249,999', '$250,000 or more'
    ]).astype(int)

    return enc

# ----------------
# Panel regression
# ----------------

def run_panel_regression(bids, enc, out_handle, task_filter=None, label='FULL SAMPLE',
                         exclude_difficulty=False, exclude_pe=False, exclude_ee=False,
                         outcome_var='bid_amount', use_raw_means=True):
    """
    Fit panel random-effects model and write summary + directional hypothesis tests.

    use_raw_means=True  → report raw sample means (used for bid_amount outcome)
    use_raw_means=False → report model-predicted means at covariate reference levels
    """
    # Predictors to merge from participant-level (plus dummies discovered dynamically)
    predictor_cols = ['age','education_Post-grad','education_Undergrad','income_$50-100k','income_>$100k']
    
    # Add PE and EE covariates unless excluded
    if not exclude_pe:
        predictor_cols.append('pre_pe')
    if not exclude_ee:
        predictor_cols.append('pre_ee')
    
    demo_dummies = [c for c in enc.columns
                    if (c.startswith('gender_') or c.startswith('race_') or c.startswith('religion_'))
                    and not c.endswith('_group') and not c.endswith('_recoded')]
    predictor_cols += sorted(demo_dummies)

    # Build task-level frame (optionally filter tasks by difficulty index)
    task = bids.copy()
    if task_filter is not None:
        task = task[task['task'].isin(task_filter)].copy()

    for c in predictor_cols + ['monotonicity','num_errors']:
        task[c] = task['participant_id'].map(enc[c])

    # Task effects (difficulty from task index +1; display order from randomization)
    diff_dum = pd.get_dummies(task['task'].astype(int) + 1, prefix='difficulty', drop_first=True)
    disp_dum = pd.get_dummies(task['display_order'], prefix='display_order', drop_first=True)

    # Treatment coding (categorical num_errors with 1 as reference)
    task['num_errors_3'] = (task['num_errors'] == 3).astype(int)
    task['num_errors_5'] = (task['num_errors'] == 5).astype(int)
    task['num_errors_3_monotonic'] = task['num_errors_3'] * task['monotonicity']
    task['num_errors_5_monotonic'] = task['num_errors_5'] * task['monotonicity']

    main_treat = ['monotonicity','num_errors_3','num_errors_5','num_errors_3_monotonic','num_errors_5_monotonic']
    
    # Include difficulty covariates unless excluded
    include_difficulty = not exclude_difficulty and (task_filter is None)
    features = [task[main_treat + predictor_cols]]
    if include_difficulty:
        features.append(diff_dum)
    features.append(disp_dum)
    X = pd.concat(features, axis=1).astype(float)
    y = task[outcome_var].astype(float)
    ids = task['participant_id']

    X = sm.add_constant(X, has_constant='add')

    # Build panel index
    panel = X.copy()
    panel['y'] = y.values
    panel['entity'] = ids.values
    panel['time'] = panel.groupby('entity').cumcount()
    panel = panel.set_index(['entity','time'])
    y_panel = panel['y']
    X_panel = panel.drop(columns=['y'])

    re = RandomEffects(y_panel, X_panel)
    res = re.fit(cov_type='clustered', cluster_entity=True)

    # -------------------------
    # Output (summary + tests)
    # -------------------------
    out_handle.write('\n' + '='*100 + '\n')
    out_handle.write(f'REGRESSION RUN: {label}\n')
    if task_filter is not None:
        out_handle.write(f'(Subset: task indices in {list(task_filter)})\n')
    if exclude_difficulty:
        out_handle.write('(Excluding task difficulty covariates)\n')
    if exclude_pe:
        out_handle.write('(Excluding pre_pe covariate)\n')
    if exclude_ee:
        out_handle.write('(Excluding pre_ee covariate)\n')
    if outcome_var != 'bid_amount':
        out_handle.write(f'(Outcome variable: {outcome_var})\n')
    out_handle.write('='*100 + '\n\n')
    out_handle.write(res.summary.as_text())

    # Directional hypothesis tests (one-tailed)
    params = res.params
    cov = res.cov
    ix = params.index.get_loc

    def wald_contrast(r, alt):
        est = float(r @ params.values)
        var = float(r @ cov.values @ r)
        se = np.sqrt(var)
        t = est / se
        if alt == "greater":
            p = 1 - stats.t.cdf(t, res.df_resid)
        elif alt == "less":
            p = stats.t.cdf(t, res.df_resid)
        else:
            p = 2 * (1 - stats.t.cdf(abs(t), res.df_resid))
        return est, se, t, p

    # Cell means (raw sample means or model-predicted at covariate reference)
    if use_raw_means:
        mu_1nm = task.loc[(task['num_errors']==1) & (task['monotonicity']==0), outcome_var].mean()
        mu_1m  = task.loc[(task['num_errors']==1) & (task['monotonicity']==1), outcome_var].mean()
        mu_3nm = task.loc[(task['num_errors']==3) & (task['monotonicity']==0), outcome_var].mean()
        mu_3m  = task.loc[(task['num_errors']==3) & (task['monotonicity']==1), outcome_var].mean()
        mu_5nm = task.loc[(task['num_errors']==5) & (task['monotonicity']==0), outcome_var].mean()
        mu_5m  = task.loc[(task['num_errors']==5) & (task['monotonicity']==1), outcome_var].mean()
    else:
        mu_1nm = params['const']
        mu_1m  = params['const'] + params['monotonicity']
        mu_3nm = params['const'] + params['num_errors_3']
        mu_3m  = params['const'] + params['monotonicity'] + params['num_errors_3'] + params['num_errors_3_monotonic']
        mu_5nm = params['const'] + params['num_errors_5']
        mu_5m  = params['const'] + params['monotonicity'] + params['num_errors_5'] + params['num_errors_5_monotonic']

    out_handle.write('\n\n' + '-'*100 + '\n')
    out_handle.write('HYPOTHESIS TESTING (directional, one-tailed)\n')
    out_handle.write('-'*100 + '\n\n')
    if use_raw_means:
        out_handle.write('Raw sample means by condition:\n')
    else:
        out_handle.write('Condition means (at reference levels of covariates):\n')
    out_handle.write(f'  μ_1nm=${mu_1nm:.4f}, μ_1m=${mu_1m:.4f}; '
                     f'μ_3nm=${mu_3nm:.4f}, μ_3m=${mu_3m:.4f}; '
                     f'μ_5nm=${mu_5nm:.4f}, μ_5m=${mu_5m:.4f}\n\n')

    # Counts by cell (for diagnostics)
    n_1_nm = ((task['num_errors']==1) & (task['monotonicity']==0)).sum()
    n_1_m  = ((task['num_errors']==1) & (task['monotonicity']==1)).sum()
    n_3_nm = ((task['num_errors']==3) & (task['monotonicity']==0)).sum()
    n_3_m  = ((task['num_errors']==3) & (task['monotonicity']==1)).sum()
    n_5_nm = ((task['num_errors']==5) & (task['monotonicity']==0)).sum()
    n_5_m  = ((task['num_errors']==5) & (task['monotonicity']==1)).sum()

    tot_1 = n_1_nm + n_1_m
    tot_3 = n_3_nm + n_3_m
    tot_5 = n_5_nm + n_5_m
    tot_35 = tot_3 + tot_5
    tot_all = tot_1 + tot_3 + tot_5

    # Sample-share weights
    w1_nm = n_1_nm / tot_1 if tot_1 else 0.0
    w1_m  = n_1_m  / tot_1 if tot_1 else 0.0
    w3_nm = n_3_nm / tot_3 if tot_3 else 0.0
    w3_m  = n_3_m  / tot_3 if tot_3 else 0.0
    w5_nm = n_5_nm / tot_5 if tot_5 else 0.0
    w5_m  = n_5_m  / tot_5 if tot_5 else 0.0
    w1 = tot_1 / tot_all if tot_all else 0.0
    w3 = tot_3 / tot_all if tot_all else 0.0
    w5 = tot_5 / tot_all if tot_all else 0.0

    out_handle.write('Sample sizes per cell (nm, m):\n')
    out_handle.write(f'  1-error=({n_1_nm},{n_1_m}), 3-error=({n_3_nm},{n_3_m}), 5-error=({n_5_nm},{n_5_m})\n')
    out_handle.write(f'Hypothesis weights (sample-share): w1={w1:.3f}, w3={w3:.3f}, w5={w5:.3f}\n\n')

    # H1: (μ_3 < μ_1) and (μ_5 < μ_3) — one-sided ("less")
    h1a = np.zeros(len(params))
    h1a[ix('monotonicity')]           = (w3_m - w1_m)
    h1a[ix('num_errors_3')]           = (w3_nm + w3_m)
    h1a[ix('num_errors_3_monotonic')] = (w3_m)
    h1a_est, h1a_se, h1a_t, h1a_p = wald_contrast(h1a, "less")

    h1b = np.zeros(len(params))
    h1b[ix('monotonicity')]            = (w5_m - w3_m)
    h1b[ix('num_errors_5')]            = (w5_nm + w5_m)
    h1b[ix('num_errors_3')]            = -(w3_nm + w3_m)
    h1b[ix('num_errors_5_monotonic')]  = (w5_m)
    h1b[ix('num_errors_3_monotonic')]  = -(w3_m)
    h1b_est, h1b_se, h1b_t, h1b_p = wald_contrast(h1b, "less")

    mu_1 = w1_nm*mu_1nm + w1_m*mu_1m
    mu_3 = w3_nm*mu_3nm + w3_m*mu_3m
    mu_5 = w5_nm*mu_5nm + w5_m*mu_5m
    out_handle.write('H1: More errors reduce WTP ("<")\n')
    out_handle.write(f'  μ_1=${mu_1:.4f}, μ_3=${mu_3:.4f}, μ_5=${mu_5:.4f}\n')
    out_handle.write(f'  H1a: μ_3-μ_1 = ${h1a_est:.4f}, SE={h1a_se:.4f}, t={h1a_t:.3f}, p={h1a_p:.4f}\n')
    out_handle.write(f'  H1b: μ_5-μ_3 = ${h1b_est:.4f}, SE={h1b_se:.4f}, t={h1b_t:.3f}, p={h1b_p:.4f}\n')
    out_handle.write(f'  Result: {"SUPPORTED" if (h1a_p<0.05 and h1b_p<0.05) else "NOT SUPPORTED"}\n\n')

    # H2: μ_nm < μ_m (weighted over counts with w1,w3,w5)
    # Contrast: (μ_m^w - μ_nm^w) = w1*(βm) + w3*(βm+β3m) + w5*(βm+β5m)
    h2 = np.zeros(len(params))
    h2[ix('monotonicity')]           = (w1 + w3 + w5)
    h2[ix('num_errors_3_monotonic')] = w3
    h2[ix('num_errors_5_monotonic')] = w5
    h2_est, h2_se, h2_t, h2_p = wald_contrast(h2, "greater")

    mu_nm_w = w1*mu_1nm + w3*mu_3nm + w5*mu_5nm
    mu_m_w  = w1*mu_1m  + w3*mu_3m  + w5*mu_5m
    out_handle.write('H2: Non-monotonic errors reduce WTP (μ_m - μ_nm > 0)\n')
    out_handle.write(f'  μ_nm(weighted)=${mu_nm_w:.4f}, μ_m(weighted)=${mu_m_w:.4f}\n')
    out_handle.write(f'  Diff=${h2_est:.4f}, SE={h2_se:.4f}, t={h2_t:.3f}, p={h2_p:.4f}\n')
    out_handle.write(f'  Result: {"SUPPORTED" if h2_p<0.05 else "NOT SUPPORTED"}\n\n')

    # H3: (μ_(3,5)m - μ_(3,5)nm) > (μ_1m - μ_1nm) → w35_3*β3m + w35_5*β5m > 0
    w35_3 = tot_3 / tot_35 if tot_35 else 0.0
    w35_5 = tot_5 / tot_35 if tot_35 else 0.0
    h3 = np.zeros(len(params))
    h3[ix('num_errors_3_monotonic')] = w35_3
    h3[ix('num_errors_5_monotonic')] = w35_5
    h3_est, h3_se, h3_t, h3_p = wald_contrast(h3, "greater")

    mono_adv_1  = mu_1m - mu_1nm
    mono_adv_3  = mu_3m - mu_3nm
    mono_adv_5  = mu_5m - mu_5nm
    mono_adv_35 = w35_3*mono_adv_3 + w35_5*mono_adv_5
    out_handle.write('H3: Non-monotonic penalty larger for multiple errors (">")\n')
    out_handle.write(f'  Advantage(1)=${mono_adv_1:.4f}, Advantage(3)=${mono_adv_3:.4f}, '
                     f'Advantage(5)=${mono_adv_5:.4f}, Weighted(3,5)=${mono_adv_35:.4f}\n')
    out_handle.write(f'  Excess advantage=${h3_est:.4f}, SE={h3_se:.4f}, t={h3_t:.3f}, p={h3_p:.4f}\n')
    out_handle.write(f'  Result: {"SUPPORTED" if h3_p<0.05 else "NOT SUPPORTED"}\n\n')

    # H4a: μ_3nm - μ_5m < $0.05
    h4_suffix = '"< $0.05"' if use_raw_means else 'with $0.05 margin'
    h4a = np.zeros(len(params))
    h4a[ix('num_errors_3')]           =  1
    h4a[ix('monotonicity')]           = -1
    h4a[ix('num_errors_5')]           = -1
    h4a[ix('num_errors_5_monotonic')] = -1
    h4a_est, h4a_se, _, _ = wald_contrast(h4a, "two-sided")
    t_h4a = (h4a_est - 0.05) / h4a_se
    h4a_p = stats.t.cdf(t_h4a, res.df_resid)
    out_handle.write(f'H4a: 3 non-monotonic vs 5 monotonic ({h4_suffix})\n')
    out_handle.write(f'  Δ=μ_3nm - μ_5m = ${h4a_est:.4f}, SE={h4a_se:.4f}, t={t_h4a:.3f}, p={h4a_p:.4f}\n')
    out_handle.write(f'  Result: {"SUPPORTED" if h4a_p<0.05 else "NOT SUPPORTED"}\n\n')

    # H4b: μ_1nm - μ_3m < $0.05
    h4b = np.zeros(len(params))
    h4b[ix('monotonicity')]           = -1
    h4b[ix('num_errors_3')]           = -1
    h4b[ix('num_errors_3_monotonic')] = -1
    h4b_est, h4b_se, _, _ = wald_contrast(h4b, "two-sided")
    t_h4b = (h4b_est - 0.05) / h4b_se
    h4b_p = stats.t.cdf(t_h4b, res.df_resid)
    out_handle.write(f'H4b: 1 non-monotonic vs 3 monotonic ({h4_suffix})\n')
    out_handle.write(f'  Δ=μ_1nm - μ_3m = ${h4b_est:.4f}, SE={h4b_se:.4f}, t={t_h4b:.3f}, p={h4b_p:.4f}\n')
    out_handle.write(f'  Result: {"SUPPORTED" if h4b_p<0.05 else "NOT SUPPORTED"}\n')

# ----------------------------
# Participant-level analyses
# ----------------------------

def calculate_pe_difference(enc):
    """Calculate pre-post PE difference for each participant"""
    pe_diff = {}
    for idx, row in enc.iterrows():
        # Calculate pre PE average
        pre_cols = ['pre_PE1', 'pre_PE2', 'pre_PE3', 'pre_PE4']
        pre_values = []
        for col in pre_cols:
            if col in row and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        pre_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        # Calculate post PE average
        post_cols = ['post_PE1', 'post_PE2', 'post_PE3', 'post_PE4']
        post_values = []
        for col in post_cols:
            if col in row and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        post_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        # Calculate difference (post - pre) if both are available
        if pre_values and post_values:
            pre_avg = np.mean(pre_values)
            post_avg = np.mean(post_values)
            difference = post_avg - pre_avg
            if np.isfinite(difference):
                pe_diff[idx] = difference
            else:
                pe_diff[idx] = np.nan
        else:
            pe_diff[idx] = np.nan
    
    return pe_diff

def calculate_ee_difference(enc):
    """Calculate pre-post EE difference for each participant"""
    ee_diff = {}
    for idx, row in enc.iterrows():
        # Calculate pre EE average
        pre_cols = ['pre_EE1', 'pre_EE2', 'pre_EE3']
        pre_values = []
        for col in pre_cols:
            if col in row and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        pre_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        # Calculate post EE average
        post_cols = ['post_EE1', 'post_EE2', 'post_EE3']
        post_values = []
        for col in post_cols:
            if col in row and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        post_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        # Calculate difference (post - pre) if both are available
        if pre_values and post_values:
            pre_avg = np.mean(pre_values)
            post_avg = np.mean(post_values)
            difference = post_avg - pre_avg
            if np.isfinite(difference):
                ee_diff[idx] = difference
            else:
                ee_diff[idx] = np.nan
        else:
            ee_diff[idx] = np.nan
    
    return ee_diff

def run_participant_level_analysis(enc, out_handle, outcome_var, label, exclude_pe=False, exclude_ee=False):
    """Run participant-level OLS regression analysis for PE/EE pre-post difference outcomes."""
    # Predictors to include
    predictor_cols = ['age','education_Post-grad','education_Undergrad','income_$50-100k','income_>$100k']
    
    # Add PE and EE covariates unless excluded
    if not exclude_pe:
        predictor_cols.append('pre_pe')
    if not exclude_ee:
        predictor_cols.append('pre_ee')
    
    demo_dummies = [c for c in enc.columns
                    if (c.startswith('gender_') or c.startswith('race_') or c.startswith('religion_'))
                    and not c.endswith('_group') and not c.endswith('_recoded')]
    predictor_cols += sorted(demo_dummies)

    # Build participant-level data
    participant_data = enc[['monotonicity', 'num_errors'] + predictor_cols + [outcome_var]].copy()
    
    # Drop rows with missing outcome
    participant_data = participant_data.dropna(subset=[outcome_var])
    
    if len(participant_data) == 0:
        out_handle.write(f'\n{label}: No valid data for analysis\n')
        return
    
    # Treatment coding (categorical num_errors with 1 as reference)
    participant_data['num_errors_3'] = (participant_data['num_errors'] == 3).astype(int)
    participant_data['num_errors_5'] = (participant_data['num_errors'] == 5).astype(int)
    participant_data['num_errors_3_monotonic'] = participant_data['num_errors_3'] * participant_data['monotonicity']
    participant_data['num_errors_5_monotonic'] = participant_data['num_errors_5'] * participant_data['monotonicity']

    main_treat = ['monotonicity','num_errors_3','num_errors_5','num_errors_3_monotonic','num_errors_5_monotonic']
    
    # Build feature matrix
    X = participant_data[main_treat + predictor_cols].astype(float)
    y = participant_data[outcome_var].astype(float)
    
    X = sm.add_constant(X, has_constant='add')
    
    # Fit OLS model (no random effects needed for participant-level data)
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HC3')  # Use robust standard errors

    # -------------------------
    # Output (summary + tests)
    # -------------------------
    out_handle.write('\n' + '='*100 + '\n')
    out_handle.write(f'PARTICIPANT-LEVEL REGRESSION: {label}\n')
    if exclude_pe:
        out_handle.write('(Excluding pre_pe covariate)\n')
    if exclude_ee:
        out_handle.write('(Excluding pre_ee covariate)\n')
    out_handle.write(f'(Outcome variable: {outcome_var})\n')
    out_handle.write('='*100 + '\n\n')
    out_handle.write(res.summary().as_text())

    # Directional hypothesis tests (one-tailed)
    params = res.params
    cov = res.cov_params()
    
    def wald_contrast(r, alt):
        est = float(r @ params.values)
        var = float(r @ cov.values @ r)
        se = np.sqrt(var)
        t = est / se
        if alt == "greater":
            p = 1 - stats.t.cdf(t, res.df_resid)
        elif alt == "less":
            p = stats.t.cdf(t, res.df_resid)
        else:
            p = 2 * (1 - stats.t.cdf(abs(t), res.df_resid))
        return est, se, t, p

    # Only run hypothesis tests if we have the required parameters
    try:
        ix = params.index.get_loc
        
        # Raw sample means by cell (for reporting)
        mu_1nm = participant_data.loc[(participant_data['num_errors']==1) & (participant_data['monotonicity']==0), outcome_var].mean()
        mu_1m  = participant_data.loc[(participant_data['num_errors']==1) & (participant_data['monotonicity']==1), outcome_var].mean()
        mu_3nm = participant_data.loc[(participant_data['num_errors']==3) & (participant_data['monotonicity']==0), outcome_var].mean()
        mu_3m  = participant_data.loc[(participant_data['num_errors']==3) & (participant_data['monotonicity']==1), outcome_var].mean()
        mu_5nm = participant_data.loc[(participant_data['num_errors']==5) & (participant_data['monotonicity']==0), outcome_var].mean()
        mu_5m  = participant_data.loc[(participant_data['num_errors']==5) & (participant_data['monotonicity']==1), outcome_var].mean()

        out_handle.write('\n\n' + '-'*100 + '\n')
        out_handle.write('HYPOTHESIS TESTING (directional, one-tailed)\n')
        out_handle.write('-'*100 + '\n\n')
        out_handle.write('Raw sample means by condition:\n')
        out_handle.write(f'  μ_1nm={mu_1nm:.4f}, μ_1m={mu_1m:.4f}; '
                         f'μ_3nm={mu_3nm:.4f}, μ_3m={mu_3m:.4f}; '
                         f'μ_5nm={mu_5nm:.4f}, μ_5m={mu_5m:.4f}\n\n')

        # Counts by cell (for diagnostics)
        n_1_nm = ((participant_data['num_errors']==1) & (participant_data['monotonicity']==0)).sum()
        n_1_m  = ((participant_data['num_errors']==1) & (participant_data['monotonicity']==1)).sum()
        n_3_nm = ((participant_data['num_errors']==3) & (participant_data['monotonicity']==0)).sum()
        n_3_m  = ((participant_data['num_errors']==3) & (participant_data['monotonicity']==1)).sum()
        n_5_nm = ((participant_data['num_errors']==5) & (participant_data['monotonicity']==0)).sum()
        n_5_m  = ((participant_data['num_errors']==5) & (participant_data['monotonicity']==1)).sum()

        out_handle.write('Sample sizes per cell (nm, m):\n')
        out_handle.write(f'  1-error=({n_1_nm},{n_1_m}), 3-error=({n_3_nm},{n_3_m}), 5-error=({n_5_nm},{n_5_m})\n\n')

        # H1: More errors reduce the outcome variable — one-sided ("less")
        # For participant-level analysis, use equal weights
        w1_nm, w1_m = 0.5, 0.5
        w3_nm, w3_m = 0.5, 0.5 
        w5_nm, w5_m = 0.5, 0.5

        h1a = np.zeros(len(params))
        h1a[ix('monotonicity')]           = (w3_m - w1_m)
        h1a[ix('num_errors_3')]           = (w3_nm + w3_m)
        h1a[ix('num_errors_3_monotonic')] = (w3_m)
        h1a_est, h1a_se, h1a_t, h1a_p = wald_contrast(h1a, "less")

        # H1b: μ_5 < μ_3
        h1b = np.zeros(len(params))
        h1b[ix('monotonicity')]            = (w5_m - w3_m)
        h1b[ix('num_errors_5')]            = (w5_nm + w5_m)
        h1b[ix('num_errors_3')]            = -(w3_nm + w3_m)
        h1b[ix('num_errors_5_monotonic')]  = (w5_m)
        h1b[ix('num_errors_3_monotonic')]  = -(w3_m)
        h1b_est, h1b_se, h1b_t, h1b_p = wald_contrast(h1b, "less")

        mu_1 = w1_nm*mu_1nm + w1_m*mu_1m
        mu_3 = w3_nm*mu_3nm + w3_m*mu_3m
        mu_5 = w5_nm*mu_5nm + w5_m*mu_5m
        out_handle.write('H1: More errors reduce outcome ("<")\n')
        out_handle.write(f'  μ_1={mu_1:.4f}, μ_3={mu_3:.4f}, μ_5={mu_5:.4f}\n')
        out_handle.write(f'  H1a: μ_3-μ_1 = {h1a_est:.4f}, SE={h1a_se:.4f}, t={h1a_t:.3f}, p={h1a_p:.4f}\n')
        out_handle.write(f'  H1b: μ_5-μ_3 = {h1b_est:.4f}, SE={h1b_se:.4f}, t={h1b_t:.3f}, p={h1b_p:.4f}\n')
        out_handle.write(f'  Result: {"SUPPORTED" if (h1a_p<0.05 and h1b_p<0.05) else "NOT SUPPORTED"}\n\n')

        # H2: Non-monotonic errors reduce outcome more (μ_m > μ_nm)
        w1, w3, w5 = 1/3, 1/3, 1/3  # Equal weights for participant-level
        h2 = np.zeros(len(params))
        h2[ix('monotonicity')]           = (w1 + w3 + w5)
        h2[ix('num_errors_3_monotonic')] = w3
        h2[ix('num_errors_5_monotonic')] = w5
        h2_est, h2_se, h2_t, h2_p = wald_contrast(h2, "greater")

        mu_nm_w = w1*mu_1nm + w3*mu_3nm + w5*mu_5nm
        mu_m_w  = w1*mu_1m  + w3*mu_3m  + w5*mu_5m
        out_handle.write('H2: Non-monotonic errors reduce outcome more (μ_m - μ_nm > 0)\n')
        out_handle.write(f'  μ_nm(weighted)={mu_nm_w:.4f}, μ_m(weighted)={mu_m_w:.4f}\n')
        out_handle.write(f'  Diff={h2_est:.4f}, SE={h2_se:.4f}, t={h2_t:.3f}, p={h2_p:.4f}\n')
        out_handle.write(f'  Result: {"SUPPORTED" if h2_p<0.05 else "NOT SUPPORTED"}\n\n')

        # H3: Non-monotonic penalty larger for multiple errors
        w35_3, w35_5 = 0.5, 0.5  # Equal weights for 3 and 5 errors
        h3 = np.zeros(len(params))
        h3[ix('num_errors_3_monotonic')] = w35_3
        h3[ix('num_errors_5_monotonic')] = w35_5
        h3_est, h3_se, h3_t, h3_p = wald_contrast(h3, "greater")

        mono_adv_1  = mu_1m - mu_1nm
        mono_adv_3  = mu_3m - mu_3nm
        mono_adv_5  = mu_5m - mu_5nm
        mono_adv_35 = w35_3*mono_adv_3 + w35_5*mono_adv_5
        out_handle.write('H3: Non-monotonic penalty larger for multiple errors (">")\n')
        out_handle.write(f'  Advantage(1)={mono_adv_1:.4f}, Advantage(3)={mono_adv_3:.4f}, '
                         f'Advantage(5)={mono_adv_5:.4f}, Weighted(3,5)={mono_adv_35:.4f}\n')
        out_handle.write(f'  Excess advantage={h3_est:.4f}, SE={h3_se:.4f}, t={h3_t:.3f}, p={h3_p:.4f}\n')
        out_handle.write(f'  Result: {"SUPPORTED" if h3_p<0.05 else "NOT SUPPORTED"}\n\n')

    except Exception as e:
        out_handle.write(f'\nHypothesis testing skipped due to error: {e}\n')

# ------
# Main
# ------

def main():
    df = load_data()
    bids = build_task_level_bids(df)
    bids = add_display_order(df, bids)
    enc = encode_predictors(df)
    
    # Calculate additional participant-level outcome variables
    pe_diff = calculate_pe_difference(df)
    ee_diff = calculate_ee_difference(df)
    
    # Add these as participant-level variables to enc
    enc['pe_diff'] = enc.index.map(pe_diff).fillna(0)
    enc['ee_diff'] = enc.index.map(ee_diff).fillna(0)
    
    # Add task-level difficulty scores to bids data
    difficulty_mapping = {
        'one': 6, 'two': 5, 'few': 4, 'many': 3, 'can-but-not-me': 2, 'cannot': 1
    }
    
    # Create task-level difficulty scores
    difficulty_scores = []
    for _, bid_row in bids.iterrows():
        pid = bid_row['participant_id']
        task_num = int(bid_row['task'])  # Ensure integer
        col = f'p2_t{task_num}_diff_estimate'
        
        if col in df.columns:
            participant_row = df.loc[pid]
            if pd.notna(participant_row[col]):
                difficulty_word = str(participant_row[col]).lower().strip()
                if difficulty_word in difficulty_mapping:
                    difficulty_scores.append(difficulty_mapping[difficulty_word])
                else:
                    difficulty_scores.append(np.nan)
            else:
                difficulty_scores.append(np.nan)
        else:
            difficulty_scores.append(np.nan)
    
    bids['difficulty_score'] = difficulty_scores
    
    # Add participant-level outcome variables to the task-level bids data
    for col in ['pe_diff', 'ee_diff']:
        bids[col] = bids['participant_id'].map(enc[col])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'regression.txt')

    with open(out_path, 'w', encoding='utf-8') as f:
        # Full sample run (bid_amount outcome, raw sample means)
        run_panel_regression(bids=bids, enc=enc, out_handle=f, task_filter=None,
                             label='FULL SAMPLE', use_raw_means=True)
        
        # Self-reported task difficulty analysis (model-predicted means, exclude difficulty covariate)
        run_panel_regression(bids=bids, enc=enc, out_handle=f, task_filter=None,
                             label='SELF-REPORTED DIFFICULTY (outcome)',
                             exclude_difficulty=True, outcome_var='difficulty_score',
                             use_raw_means=False)
        
        # PE pre-post difference analysis (exclude PE covariate)
        run_participant_level_analysis(enc, f, 'pe_diff', 'PE PRE-POST DIFFERENCE',
                                       exclude_pe=True)
        
        # EE pre-post difference analysis (exclude EE covariate)
        run_participant_level_analysis(enc, f, 'ee_diff', 'EE PRE-POST DIFFERENCE',
                                       exclude_ee=True)

    print(f'Script completed. Output written to: {out_path}')

if __name__ == "__main__":
    main()
