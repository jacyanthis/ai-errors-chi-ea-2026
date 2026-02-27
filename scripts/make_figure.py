#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def draw_mean_ci(ax, x, mean_val, ci_lo, ci_hi,
                 ci_half_span=0.15, line_width=3, cap_width=None):
    """Draw mean line + CI whiskers on *ax* at horizontal position *x*."""
    if np.isnan(mean_val):
        return
    if cap_width is None:
        cap_width = line_width
    mhs = ci_half_span / 2.0
    ax.plot([x - mhs, x + mhs], [mean_val, mean_val],
            color='black', linewidth=line_width, alpha=0.8)
    if (ci_lo is not None and ci_hi is not None
            and not np.isnan(ci_lo) and not np.isnan(ci_hi)):
        ax.plot([x - ci_half_span, x + ci_half_span], [ci_lo, ci_lo],
                color='black', linewidth=cap_width, alpha=0.6)
        ax.plot([x - ci_half_span, x + ci_half_span], [ci_hi, ci_hi],
                color='black', linewidth=cap_width, alpha=0.6)
        ax.plot([x, x], [ci_lo, ci_hi],
                color='black', linewidth=line_width, alpha=0.6)


# ---------------------------------------------------------------------------
# condition definitions
# ---------------------------------------------------------------------------

CONDITION_ORDER = [
    'mono x 1', 'n_mo x 1',
    'mono x 3', 'n_mo x 3',
    'mono x 5', 'n_mo x 5',
]

CONDITION_COLORS = {
    'n_mo x 5': '#ff1d1d',
    'mono x 5': '#ff7575',
    'n_mo x 3': '#B8860B',
    'mono x 3': '#FFD700',
    'n_mo x 1': '#00ca00',
    'mono x 1': '#77c977',
}

CONDITION_LABELS = {
    'n_mo x 5': '5\nEasy',
    'mono x 5': '5\nHard',
    'n_mo x 3': '3\nEasy',
    'mono x 3': '3\nHard',
    'n_mo x 1': '1\nEasy',
    'mono x 1': '1\nHard',
}


def condition_key(row):
    mono = 'mono' if row['monotonicity'] else 'n_mo'
    ne = row['num_errors']
    if ne >= 5:
        el = '5'
    elif ne >= 3:
        el = '3'
    else:
        el = '1'
    return f'{mono} x {el}'


# ---------------------------------------------------------------------------
# model-adjusted means
# ---------------------------------------------------------------------------

def compute_model_adjusted_means():
    """Fit panel RE model and return model-adjusted cell means and SEs."""
    import statsmodels.api as sm
    from linearmodels.panel import RandomEffects

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from regression import (load_data, build_task_level_bids,
                            add_display_order, encode_predictors)

    df = load_data()
    bids = build_task_level_bids(df)
    bids = add_display_order(df, bids)
    enc = encode_predictors(df)

    task = bids.copy()
    predictor_cols = [
        'age', 'education_Post-grad', 'education_Undergrad',
        'income_$50-100k', 'income_>$100k', 'pre_pe', 'pre_ee',
    ]
    demo_dummies = sorted(
        c for c in enc.columns
        if (c.startswith('gender_') or c.startswith('race_')
            or c.startswith('religion_'))
        and not c.endswith('_group') and not c.endswith('_recoded')
    )
    predictor_cols += demo_dummies

    for c in predictor_cols + ['monotonicity', 'num_errors']:
        task[c] = task['participant_id'].map(enc[c])

    diff_dum = pd.get_dummies(
        task['task'].astype(int) + 1, prefix='difficulty', drop_first=True)
    disp_dum = pd.get_dummies(
        task['display_order'], prefix='display_order', drop_first=True)

    task['num_errors_3'] = (task['num_errors'] == 3).astype(int)
    task['num_errors_5'] = (task['num_errors'] == 5).astype(int)
    task['num_errors_3_monotonic'] = task['num_errors_3'] * task['monotonicity']
    task['num_errors_5_monotonic'] = task['num_errors_5'] * task['monotonicity']

    main_treat = ['monotonicity', 'num_errors_3', 'num_errors_5',
                  'num_errors_3_monotonic', 'num_errors_5_monotonic']

    cov_col_names = (predictor_cols
                     + list(diff_dum.columns)
                     + list(disp_dum.columns))

    X = pd.concat(
        [task[main_treat + predictor_cols], diff_dum, disp_dum],
        axis=1,
    ).astype(float)
    y = task['bid_amount'].astype(float)
    ids = task['participant_id']

    X = sm.add_constant(X, has_constant='add')

    panel = X.copy()
    panel['y'] = y.values
    panel['entity'] = ids.values
    panel['time'] = panel.groupby('entity').cumcount()
    panel = panel.set_index(['entity', 'time'])
    y_panel = panel['y']
    X_panel = panel.drop(columns=['y'])

    re = RandomEffects(y_panel, X_panel)
    res = re.fit(cov_type='clustered', cluster_entity=True)

    params = res.params
    cov_matrix = res.cov
    ix = params.index.get_loc

    cov_means = X[cov_col_names].mean()

    cells = {
        'n_mo x 1': (0, 1),
        'mono x 1': (1, 1),
        'n_mo x 3': (0, 3),
        'mono x 3': (1, 3),
        'n_mo x 5': (0, 5),
        'mono x 5': (1, 5),
    }

    adjusted = {}
    for cond_name, (mono, ne) in cells.items():
        r = np.zeros(len(params))
        r[ix('const')] = 1.0
        r[ix('monotonicity')] = float(mono)
        r[ix('num_errors_3')] = float(ne == 3)
        r[ix('num_errors_5')] = float(ne == 5)
        r[ix('num_errors_3_monotonic')] = float(ne == 3) * mono
        r[ix('num_errors_5_monotonic')] = float(ne == 5) * mono
        for col in cov_col_names:
            r[ix(col)] = float(cov_means[col])
        mean_val = float(r @ params.values)
        se_val = float(np.sqrt(r @ cov_matrix.values @ r))
        adjusted[cond_name] = (mean_val, se_val,
                               mean_val - se_val, mean_val + se_val)

    return adjusted


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from regression import load_data
    d = load_data()

    d['condition'] = d.apply(condition_key, axis=1)

    bid_cols = [f'p2_t{j}_bid' for j in range(8)]

    # --- collect bids per condition (flat list for violin plots) ---
    condition_bids = {c: [] for c in CONDITION_ORDER}

    for _, row in d.iterrows():
        cond = row['condition']
        if cond not in condition_bids:
            continue
        for col in bid_cols:
            v = row[col]
            if pd.notna(v):
                condition_bids[cond].append(v)

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    plot_data, positions, colors = [], [], []
    for i, cond in enumerate(CONDITION_ORDER):
        if condition_bids[cond]:
            plot_data.append(condition_bids[cond])
            positions.append(i)
            colors.append(CONDITION_COLORS[cond])

    if plot_data:
        vp = ax.violinplot(plot_data, positions=positions, widths=0.7)
        for pc, color in zip(vp['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        if 'cbars' in vp:
            vp['cbars'].set_edgecolor('white')
            vp['cbars'].set_linewidth(0.5)
        for part in ('cmedians', 'cmeans', 'cmins', 'cmaxes'):
            if part in vp:
                vp[part].set_visible(False)

    adjusted = compute_model_adjusted_means()
    for i, cond in enumerate(CONDITION_ORDER):
        if cond in adjusted:
            mu, se, lo, hi = adjusted[cond]
            draw_mean_ci(ax, i, mu, lo, hi,
                         ci_half_span=0.12, line_width=2, cap_width=4)

    # axes
    ax.set_ylabel('Bid for AI Tool', fontsize=28, fontweight='bold', labelpad=15)
    ax.set_xlabel('Error Number / Type', fontsize=28, fontweight='bold', labelpad=15)
    ax.set_title('', fontsize=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=24)

    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(labels, fontsize=26, ha='center')
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)

    ax.set_ylim(0.00, 0.49)
    y_ticks = np.arange(0, 0.50, 0.05)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'${t:.2f}' for t in y_ticks])

    plt.tight_layout()

    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'results.png')
    plt.savefig(out, dpi=900, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
