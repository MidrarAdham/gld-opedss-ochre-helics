'''
Author: Midrar Adham
generalization.py — Future day prediction experiment.

Trains the Bayesian estimator and per-device OLS on N training days,
then applies the trained coefficients to future days using raw ON/OFF
states to evaluate how well the method generalizes beyond the
training period.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import DataLoader
from ols import OrdinaryLeastSquare
from bayesian_estimator import BayesianEstimator
from utils import (
    r2_score,
    mape_score,
    get_ground_truth_per_device,
    build_active_ground_truth,
)

if __name__ == '__main__':

    wh_dir          = '../results/wh_cosim/'
    hvac_dir        = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    # ── Configuration ────────────────────────────────────────────────────
    TRAIN_DAYS     = 2
    FUTURE_DAYS    = list(range(3, 31))
    LAMBDA         = 0.01
    CHUNKS_PER_DAY = 144
    EXCLUDE_HVAC   = [
        '../results/hvac_cosim/ochre_load_16.csv',
    ]

    train_day_end = TRAIN_DAYS * 1440
    n_chunks      = TRAIN_DAYS * CHUNKS_PER_DAY

    # ── Load training data ───────────────────────────────────────────────
    wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=train_day_end)
    hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=train_day_end)
    total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=train_day_end)

    wh_loader.load_transformer_data()
    hvac_loader.load_transformer_data()
    feeder_df = total_house_loader.load_transformer_data()

    wh_df   = wh_loader.load_csv_files(threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files(threshold=100.0)

    # ── Bayesian estimation on training days ─────────────────────────────
    estimator      = BayesianEstimator(num_chunks=n_chunks, discount=LAMBDA)
    wh_histories   = estimator.fit_many(all_dfs=wh_df)
    hvac_histories = estimator.fit_many(all_dfs=hvac_df)

    # ── OLS on training days ─────────────────────────────────────────────
    ols = OrdinaryLeastSquare(
        feeder_demand  = feeder_df,
        wh_histories   = wh_histories,
        hvac_histories = hvac_histories,
        wh_all_dfs     = wh_loader.all_dfs,
        hvac_all_dfs   = hvac_loader.all_dfs,
    )
    results       = ols.run(exclude_hvac=EXCLUDE_HVAC)
    hvac_active   = results['per_d_hvac_active']
    kw_per_device = results['per_d_kw_hvac']
    active_cols   = hvac_active.columns
    kw_trained    = kw_per_device[active_cols].values
    gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)

    print(f'\n── Training complete (days 1-{TRAIN_DAYS}) ──────────────────')

    # ── Future day prediction ────────────────────────────────────────────
    future_results = {}

    for future_day in FUTURE_DAYS:
        day_start = (future_day - 1) * 1440
        day_end_f = future_day * 1440

        future_hvac_loader = DataLoader(
            results_dir=hvac_dir,
            day_start=day_start,
            day_end=day_end_f,
        )
        future_hvac_df = future_hvac_loader.load_csv_files(threshold=100.0)

        # Build state matrix from raw ON/OFF states
        future_state_matrix = ols._build_state_matrix(future_hvac_df)

        # Estimate demand: Σ kw_i × ON_i(t)
        future_estimated = future_state_matrix[active_cols].values @ kw_trained

        # Ground truth for future day
        future_gt = build_active_ground_truth(
            hvac_active_cols=active_cols.tolist(),
            hvac_all_dfs=future_hvac_loader.all_dfs,
        )

        r2_future   = r2_score(future_gt, future_estimated)
        mape_future = mape_score(future_gt, future_estimated)

        future_results[future_day] = {
            'r2':              r2_future,
            'mape':            mape_future,
            'future_gt':       future_gt,
            'future_estimated': future_estimated,
            'state_matrix':    future_state_matrix,
        }

        print(f'day={future_day:2d} | R²={r2_future:.3f} | MAPE={mape_future:.1f}%')

    # ── Plot 1: R² and MAPE vs future day ────────────────────────────────
    days      = list(future_results.keys())
    r2_vals   = [future_results[d]['r2']   for d in days]
    mape_vals = [future_results[d]['mape'] for d in days]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(days, r2_vals, marker='o', color='steelblue', linewidth=1.5)
    axes[0].axhline(y=0.88, color='green', linestyle='--', linewidth=1.0,
                    label='R²=0.88 threshold')
    axes[0].set_xlabel('Future Day')
    axes[0].set_ylabel('R²')
    axes[0].set_title(f'R² vs Future Day (trained on days 1-{TRAIN_DAYS})')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(days, mape_vals, marker='o', color='orange', linewidth=1.5)
    axes[1].set_xlabel('Future Day')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title(f'MAPE vs Future Day (trained on days 1-{TRAIN_DAYS})')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'generalization_trained_{TRAIN_DAYS}days.png')
    plt.show()

    # ── Plot 2: Per-device R² vs kw_error for a selected future day ──────
    INSPECT_DAY = 30

    future_state_matrix = future_results[INSPECT_DAY]['state_matrix']
    future_gt_day       = future_results[INSPECT_DAY]['future_gt']
    future_hvac_loader  = DataLoader(
        results_dir=hvac_dir,
        day_start=(INSPECT_DAY - 1) * 1440,
        day_end=INSPECT_DAY * 1440,
    )
    future_hvac_loader.load_csv_files(threshold=100.0)

    per_device_r2 = {}
    for device in active_cols:
        kw_i        = kw_per_device[device]
        states_i    = future_state_matrix[device].values
        estimated_i = kw_i * states_i

        df = future_hvac_loader.all_dfs[device].copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        gt_i  = power.resample('10min').mean().values

        r2_i   = r2_score(gt_i, estimated_i)
        mape_i = mape_score(gt_i, estimated_i)

        short_name = device.split('ochre_load_')[1].replace('.csv', '')
        per_device_r2[device] = {'r2': r2_i, 'mape': mape_i}

    comparison_per_device = pd.DataFrame({
        'kw_estimated':  kw_per_device[active_cols],
        'kw_truth':      gt_per_device[active_cols],
        'r2':            pd.Series({d: per_device_r2[d]['r2']   for d in active_cols}),
        'mape':          pd.Series({d: per_device_r2[d]['mape'] for d in active_cols}),
    })
    comparison_per_device['kw_error_pct'] = (
        (comparison_per_device['kw_estimated'] - comparison_per_device['kw_truth'])
        / comparison_per_device['kw_truth'] * 100
    ).round(1)
    comparison_per_device = comparison_per_device.sort_values('r2', ascending=False)

    print(f'\n── Per-device R² for day {INSPECT_DAY} ─────────────────────')
    print(comparison_per_device[['kw_truth', 'kw_estimated', 'kw_error_pct', 'r2', 'mape']].to_string())

    # Scatter plot
    valid = comparison_per_device.dropna(subset=['r2', 'kw_error_pct'])
    valid = valid[valid['kw_error_pct'].abs() < 150]  # zoom in on well-behaved devices

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid['kw_error_pct'], valid['r2'],
               color='steelblue', s=60, zorder=3)

    for idx, row in valid.iterrows():
        short_name = idx.split('ochre_load_')[1].replace('.csv', '')
        ax.annotate(f'#{short_name}',
                    xy=(row['kw_error_pct'], row['r2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)

    ax.axhline(y=0,    color='red',   linestyle='--', linewidth=1.0, label='R²=0')
    ax.axhline(y=0.88, color='green', linestyle='--', linewidth=1.0, label='R²=0.88 threshold')
    ax.axvline(x=0,    color='gray',  linestyle='--', linewidth=0.8)
    ax.set_xlabel('Rated Power Estimation Error (%)')
    ax.set_ylabel('R²')
    ax.set_title(f'Per-Device R² vs Rated Power Estimation Error — Day {INSPECT_DAY}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'per_device_r2_vs_kw_error_day{INSPECT_DAY}.png')
    plt.show()
