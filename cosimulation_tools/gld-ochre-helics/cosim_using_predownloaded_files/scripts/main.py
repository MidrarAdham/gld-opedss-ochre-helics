'''
Author: Midrar Adham
Created: Sat May 02 2026

main.py — Core pipeline.
Trains the Bayesian estimator and per-device OLS on N days of data,
evaluates on the same training period, and plots the results.
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
    N_DAYS       = 10
    LAMBDA       = 0.01
    CHUNKS_PER_DAY = 144
    EXCLUDE_HVAC = [
        '../results/hvac_cosim/ochre_load_16.csv',
    ]

    day_end  = N_DAYS * 1440
    n_chunks = N_DAYS * CHUNKS_PER_DAY

    # ── Load data ────────────────────────────────────────────────────────
    wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=day_end)
    hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=day_end)
    total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=day_end)

    wh_ground_truth   = wh_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data()
    feeder_df         = total_house_loader.load_transformer_data()

    wh_df   = wh_loader.load_csv_files(threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files(threshold=100.0)

    # ── Bayesian estimation ──────────────────────────────────────────────
    estimator      = BayesianEstimator(num_chunks=n_chunks, discount=LAMBDA)
    wh_histories   = estimator.fit_many(all_dfs=wh_df)
    hvac_histories = estimator.fit_many(all_dfs=hvac_df)

    # ── OLS ──────────────────────────────────────────────────────────────
    ols = OrdinaryLeastSquare(
        feeder_demand  = feeder_df,
        wh_histories   = wh_histories,
        hvac_histories = hvac_histories,
        wh_all_dfs     = wh_loader.all_dfs,
        hvac_all_dfs   = hvac_loader.all_dfs,
    )
    results = ols.run(exclude_hvac=EXCLUDE_HVAC)

    # ── Evaluation ───────────────────────────────────────────────────────
    hvac_active     = results['per_d_hvac_active']
    kw_per_device   = results['per_d_kw_hvac']
    estimated_total = hvac_active.values @ kw_per_device[hvac_active.columns].values

    gt_active_total = build_active_ground_truth(
        hvac_active_cols=hvac_active.columns.tolist(),
        hvac_all_dfs=hvac_loader.all_dfs,
    )

    r2   = r2_score(gt_active_total, estimated_total)
    mape = mape_score(gt_active_total, estimated_total)

    print(f'\n── Training period results ({N_DAYS} days) ──────────────────')
    print(f'R²:   {r2:.3f}')
    print(f'MAPE: {mape:.1f}%')

    # ── Per-device rated power comparison ────────────────────────────────
    gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)
    comparison = pd.DataFrame({
        'estimated_W': kw_per_device,
        'truth_W':     gt_per_device,
    })
    comparison['error_pct'] = (
        (comparison['estimated_W'] - comparison['truth_W'])
        / comparison['truth_W'] * 100
    ).round(1)
    comparison = comparison.sort_values('truth_W', ascending=False)

    print('\n── Per-device HVAC rated power ──────────────────────────────')
    print(comparison.to_string())

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gt_active_total / 1e3, color='black',     linewidth=1.5,
            label='Ground Truth (active devices)')
    ax.plot(estimated_total / 1e3, color='steelblue', linewidth=1.5,
            linestyle='--', label='Estimated')
    ax.set_ylabel('HVAC Demand [kW]')
    ax.set_xlabel('Chunk Index (10-min intervals)')
    ax.set_title(f'Total HVAC Demand: Estimated vs Ground Truth — {N_DAYS} Days')
    ax.annotate(
        f'R²: {r2:.3f}\nMAPE: {mape:.1f}%',
        xy=(0.01, 0.95), xycoords='axes fraction',
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'hvac_estimated_vs_truth_{N_DAYS}days.png')
    plt.show()