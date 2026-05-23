'''
Author: Midrar Adham
sweep.py — Lambda and day sweep experiment.

Sweeps over different training window lengths and discount factors
to find the optimal configuration for the per-device OLS.
Saves results to CSV so the simulation does not need to be rerun.
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
    build_active_ground_truth,
)

if __name__ == '__main__':

    wh_dir          = '../results/wh_cosim/'
    hvac_dir        = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    EXCLUDE_HVAC = [
        '../results/hvac_cosim/ochre_load_16.csv',
    ]
    CHUNKS_PER_DAY = 144
    DAY_ENDS = [
        1440, 2880, 4320, 5760, 7200,
        8640, 10080, 11520, 12960, 14400,
    ]
    LAMBDAS = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
    CSV_PATH = 'sweep_days_lambda.csv'
    RERUN    = False  # set to True to rerun simulation

    # ── Load or run sweep ─────────────────────────────────────────────────
    if not RERUN:
        sweep_df = pd.read_csv(CSV_PATH)
        print(f'Loaded sweep results from {CSV_PATH}')
    else:
        sweep_results = {}

        for day_end in DAY_ENDS:
            n_days   = day_end // 1440
            n_chunks = n_days * CHUNKS_PER_DAY
            sweep_results[day_end] = {}

            wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=day_end)
            hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=day_end)
            total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=day_end)

            wh_ground_truth   = wh_loader.load_transformer_data()
            hvac_ground_truth = hvac_loader.load_transformer_data()
            feeder_df         = total_house_loader.load_transformer_data()

            wh_df   = wh_loader.load_csv_files(threshold=5000.0)
            hvac_df = hvac_loader.load_csv_files(threshold=100.0)

            for lamb in LAMBDAS:
                estimator      = BayesianEstimator(num_chunks=n_chunks, discount=lamb)
                wh_histories   = estimator.fit_many(all_dfs=wh_df)
                hvac_histories = estimator.fit_many(all_dfs=hvac_df)

                ols = OrdinaryLeastSquare(
                    feeder_demand  = feeder_df,
                    wh_histories   = wh_histories,
                    hvac_histories = hvac_histories,
                    wh_all_dfs     = wh_loader.all_dfs,
                    hvac_all_dfs   = hvac_loader.all_dfs,
                )
                results = ols.run(exclude_hvac=EXCLUDE_HVAC)

                hvac_active     = results['per_d_hvac_active']
                kw_per_device   = results['per_d_kw_hvac']
                estimated_total = hvac_active.values @ kw_per_device[hvac_active.columns].values

                gt_active_total = build_active_ground_truth(
                    hvac_active_cols=hvac_active.columns.tolist(),
                    hvac_all_dfs=hvac_loader.all_dfs,
                )

                r2   = r2_score(gt_active_total, estimated_total)
                mape = mape_score(gt_active_total, estimated_total)

                sweep_results[day_end][lamb] = {'r2': r2, 'mape': mape}
                print(f'days={n_days:2d} | lambda={lamb:.3f} | R²={r2:.3f} | MAPE={mape:.1f}%')

        # Save to CSV
        rows = []
        for day_end, lambda_results in sweep_results.items():
            for lamb, metrics in lambda_results.items():
                rows.append({
                    'days':   day_end // 1440,
                    'lambda': lamb,
                    'r2':     metrics['r2'],
                    'mape':   metrics['mape'],
                })
        sweep_df = pd.DataFrame(rows)
        sweep_df.to_csv(CSV_PATH, index=False)
        print(f'\nSweep results saved to {CSV_PATH}')

    # ── Plot ──────────────────────────────────────────────────────────────
    days    = sweep_df['days'].unique()
    lambdas = sweep_df['lambda'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lamb in lambdas:
        subset    = sweep_df[sweep_df['lambda'] == lamb]
        r2_vals   = subset['r2'].values
        mape_vals = subset['mape'].values
        axes[0].plot(subset['days'], r2_vals,   marker='o', label=f'λ={lamb}')
        axes[1].plot(subset['days'], mape_vals, marker='o', label=f'λ={lamb}')

    axes[0].set_xlabel('Number of Days')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² vs Days of History')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('Number of Days')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('MAPE vs Days of History')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('sweep_days_lambda.png')
    plt.show()
