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
    # plt.savefig(f'generalization_trained_{TRAIN_DAYS}days.png')
    # plt.show()

    # ── Diagnose day 15 ───────────────────────────────────────────────────
    DIAGNOSE_DAY = 15
    state_matrix_15 = future_results[DIAGNOSE_DAY]['state_matrix']

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Pick 3 devices to inspect — one good, one bad, one medium
    inspect_devices = [
        '../results/hvac_cosim/ochre_load_13.csv',  # good estimator
        '../results/hvac_cosim/ochre_load_25.csv',  # bad estimator
        '../results/hvac_cosim/ochre_load_1.csv',   # medium
    ]

    for ax, device in zip(axes, inspect_devices):
        short_name  = device.split('ochre_load_')[1].replace('.csv', '')
        kw_i        = kw_per_device[device]
        states_i    = state_matrix_15[device].values
        estimated_i = kw_i * states_i

        # Ground truth for this device on day 15
        future_hvac_loader_15 = DataLoader(
            results_dir=hvac_dir,
            day_start=(DIAGNOSE_DAY - 1) * 1440,
            day_end=DIAGNOSE_DAY * 1440,
        )
        future_hvac_loader_15.load_csv_files(threshold=100.0)
        df = future_hvac_loader_15.all_dfs[device].copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        gt_i  = power.resample('10min').mean().values

        r2_i   = r2_score(gt_i, estimated_i)
        mape_i = mape_score(gt_i, estimated_i)

        ax.plot(gt_i        / 1e3, color='black',     linewidth=1.5, label='Ground Truth')
        ax.plot(estimated_i / 1e3, color='steelblue', linewidth=1.5,
                linestyle='--', label='Estimated')
        ax.set_ylabel('Power [kW]')
        ax.set_title(f'Device #{short_name} | R²={r2_i:.3f} | MAPE={mape_i:.1f}%')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Chunk Index (10-min intervals)')
    plt.suptitle(f'Day {DIAGNOSE_DAY} — Per-Device Diagnosis', fontsize=12)
    plt.tight_layout()
    # plt.savefig(f'diagnosis_day{DIAGNOSE_DAY}.png')
    # plt.show()

    # ── Heatmap: ground truth power consumption (all 30 days) ────────────
    TOTAL_DAYS = 30
    all_hvac_loader = DataLoader(
        results_dir=hvac_dir,
        day_start=0,
        day_end=TOTAL_DAYS * 1440,
    )
    daily_avg = [
        0.950000, 1.662500, 1.695833, 7.316667, 5.612500,
        5.045833, 5.129167, 7.554167, 5.550000, 7.475000,
        6.225000, 4.737500, 3.875000, 10.895833, 12.570833,
        7.450000, 6.979167, 6.600000, 6.741667, 5.254167,
        3.333333, 3.279167, 2.458333, 0.708333, 0.004167,
        4.083333, 10.587500, 8.395833, 6.762500, 7.291667,
        ]
    
    all_hvac_df = all_hvac_loader.load_csv_files(threshold=100.0)

    device_keys = sorted(all_hvac_loader.all_dfs.keys(),
                         key=lambda p: int(p.split('ochre_load_')[1].replace('.csv', '')))

    heatmap_rows = []
    device_labels = []
    for device in device_keys:
        df = all_hvac_loader.all_dfs[device].copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        resampled = power.resample('10min').mean().fillna(0).values
        heatmap_rows.append(resampled / 1e3)
        device_labels.append(device.split('ochre_load_')[1].replace('.csv', ''))

    heatmap_matrix = np.array(heatmap_rows)

    n_chunks_total = TOTAL_DAYS * CHUNKS_PER_DAY
    xtick_positions = [d * CHUNKS_PER_DAY for d in range(TOTAL_DAYS + 1)]
    xtick_labels    = [str(d + 1) for d in range(TOTAL_DAYS)] + ['']

    # ── Heatmap with temperature overlay ─────────────────────────────────
    fig, (ax_temp, ax_heat) = plt.subplots(
        2, 1,
        figsize=(18, max(8, len(device_labels) * 0.35 + 3)),
        gridspec_kw={'height_ratios': [1, 4]},
        sharex=True
    )

    # ── Top panel: daily average temperature ─────────────────────────────
    # daily_avg is 0-indexed, one value per day
    temp_x = [d * CHUNKS_PER_DAY + CHUNKS_PER_DAY // 2 for d in range(TOTAL_DAYS)]
    ax_temp.plot(temp_x, daily_avg, color='tomato', linewidth=2,
                marker='o', markersize=4, label='Daily Avg Temp (°C)')
    ax_temp.fill_between(temp_x, daily_avg, alpha=0.2, color='tomato')
    ax_temp.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_temp.set_ylabel('Temp (°C)', fontsize=9)
    ax_temp.set_title(f'HVAC Ground Truth Power Consumption — All {TOTAL_DAYS} Days')
    ax_temp.legend(fontsize=8)
    ax_temp.grid(True, alpha=0.3)

    # ── Bottom panel: heatmap ─────────────────────────────────────────────
    im = ax_heat.imshow(
        heatmap_matrix,
        aspect='auto',
        interpolation='nearest',
        cmap='plasma',
        vmin=0,
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)
    cbar.set_label('Power [kW]')

    # Mark training window (days 1-2)
    ax_heat.axvspan(0, 2 * CHUNKS_PER_DAY - 0.5,
                    color='cyan', alpha=0.15, label='Training window (days 1-2)')
    ax_heat.legend(fontsize=8, loc='upper right')

    # Mark excluded device #16
    excluded_idx = device_labels.index('16')
    ax_heat.annotate('excluded', xy=(0, excluded_idx),
                    xytext=(5, excluded_idx),
                    fontsize=7, color='white', va='center')

    # Day dividers
    for d in range(1, TOTAL_DAYS):
        ax_heat.axvline(x=d * CHUNKS_PER_DAY - 0.5,
                        color='white', linewidth=0.4, alpha=0.5)

    ax_heat.set_xticks(xtick_positions[:-1])
    ax_heat.set_xticklabels(xtick_labels[:-1], fontsize=8)
    ax_heat.set_xlabel('Day')
    ax_heat.set_yticks(range(len(device_labels)))
    ax_heat.set_yticklabels([f'Device {n}' for n in device_labels], fontsize=7)

    plt.tight_layout()
    plt.savefig('heatmap_hvac_ground_truth_with_temp.png', dpi=150)
    # plt.show()