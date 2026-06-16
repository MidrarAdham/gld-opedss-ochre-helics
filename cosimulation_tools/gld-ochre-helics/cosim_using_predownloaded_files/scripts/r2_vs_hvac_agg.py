'''
Author: Midrar Adham
Created: Thu Jun 11 2026
'''
import os
import random
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
    train_days   = 10
    n_repeats    = 10
    LAMBDA       = 0.01
    exclude_hvac = ['../results/hvac_cosim/ochre_load_16.csv']

    for day in range(1, train_days + 1):



        day_end  = day * 1440
        n_chunks = day * 144

        # ── Load data once ───────────────────────────────────────────────────
        wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=day_end)
        hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=day_end)
        total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=day_end)

        wh_loader.load_transformer_data()
        hvac_loader.load_transformer_data()
        feeder_df = total_house_loader.load_transformer_data()

        wh_df   = wh_loader.load_csv_files(threshold=5000.0)
        hvac_df = hvac_loader.load_csv_files(threshold=100.0)

        # ── Bayesian estimation ──────────────────────────────────────────────
        estimator      = BayesianEstimator(num_chunks=n_chunks, discount=LAMBDA)
        wh_histories   = estimator.fit_many(all_dfs=wh_df)
        hvac_histories = estimator.fit_many(all_dfs=hvac_df)

        # ── Filter to 1-ton and above devices ────────────────────────────────
        one_ton_w = 3517  # 1 ton in Watts

        gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)

        all_devices = [f for f in hvac_loader.all_dfs.keys() if f not in exclude_hvac and gt_per_device[f] >= one_ton_w]
        # all_devices = [f for f in hvac_loader.all_dfs.keys() if f not in exclude_hvac]

        print(f'Devices above 1 ton: {len(all_devices)}')
        for d in all_devices:
            short = d.split('ochre_load_')[1].replace('.csv', '')
            print(f'  Device #{short}: {gt_per_device[d]/1000:.1f} kW')

        # Update fleet sizes to match filtered device count
        fleet_sizes = list(range(1, len(all_devices) + 1))

        print(f'Total active devices: {len(all_devices)}')

        ols = OrdinaryLeastSquare(
            feeder_demand  = feeder_df,
            wh_histories   = wh_histories,
            hvac_histories = hvac_histories,
            wh_all_dfs     = wh_loader.all_dfs,
            hvac_all_dfs   = hvac_loader.all_dfs,
        )

        results       = ols.run(exclude_hvac=exclude_hvac)
        # hvac_active   = results['per_d_hvac_active'] # the bayesian estimated mean matrix
        kw_per_device = results['per_d_kw_hvac'] # estimated kW for each device per two days.
        # active_cols   = hvac_active.columns
        # kw_trained    = kw_per_device[active_cols].values # the ols coefficients
        gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)
        
        print('\nmean matrix training is done\n')

        # ── Run OLS once on all devices ──────────────────────────────────────
        # results       = ols.run(exclude_hvac=exclude_hvac)
        # kw_per_device = results['per_d_kw_hvac']

        # ── Load evaluation day states ────────────────────────────────────────
        eval_hvac_loader = DataLoader(
            results_dir=hvac_dir,
            day_start=3 * 1440,
            day_end=4 * 1440,
        )
        eval_hvac_loader.load_csv_files(threshold=100.0)
        state_matrix = ols._build_state_matrix(eval_hvac_loader.all_dfs)

        # ── Fleet size sweep ──────────────────────────────────────────────────
        sweep_results = {}

        for N in fleet_sizes:
            r2_list = []
            mape_list = []


            for repeat in range(n_repeats):
                # Pick N random devices
                sampled = random.sample(all_devices, N)

                # Ground truth: sum of actual power for sampled devices only
                gt = build_active_ground_truth(
                    hvac_active_cols=sampled,
                    hvac_all_dfs=eval_hvac_loader.all_dfs,
                )

                # Estimate: kw_i × ON_i(t) for sampled devices only
                est = state_matrix[sampled].values @ kw_per_device[sampled].values

                r2 = r2_score(gt, est)
                r2_list.append(r2)
                mape = mape_score(gt, est)
                mape_list.append(mape)

            sweep_results[N] = {
                'r2_mean':   np.mean(r2_list),
                'r2_std':    np.std(r2_list),
                'mape_mean': np.mean(mape_list),
                'mape_std':  np.std(mape_list)
                }
            print(f'N={N:2d} | R2={sweep_results[N]["r2_mean"]:.3f} ± {sweep_results[N]["r2_std"]:.3f} | '
                f'MAPE={sweep_results[N]["mape_mean"]:.1f}% ± {sweep_results[N]["mape_std"]:.1f}%')
            

        # ── Save results ─────────────────────────────────────────────────────
        rows = []
        for N, metrics in sweep_results.items():
            # print(f'N={N:2d} | R²={metrics["r2_mean"]:.3f} ± {metrics["r2_std"]:.3f}')
            rows.append({'N': N, **metrics})
        pd.DataFrame(rows).to_csv(f'r2_vs_fleet_size_unfiltered_day{day}.csv', index=False)
        print(pd.DataFrame(rows))

        # ── Plot ─────────────────────────────────────────────────────────────
        ns       = list(sweep_results.keys())
        r2_means = [sweep_results[N]['r2_mean'] for N in ns]
        r2_stds  = [sweep_results[N]['r2_std']  for N in ns]
        mape_means = [sweep_results[N]['mape_mean'] for N in ns]
        mape_stds  = [sweep_results[N]['mape_std']  for N in ns]


        fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(10, 5))
        ax[0].plot(ns, r2_means, marker='o', color='steelblue', linewidth=1.5)
        # ax2 = ax.twinx()
        # ax2.plot(ns, mape_means, marker='o', color='red', linewidth=1.5)
        ax[0].fill_between(ns,
                        np.array(r2_means) - np.array(r2_stds),
                        np.array(r2_means) + np.array(r2_stds),
                        alpha=0.2, color='steelblue', label='R²: ± 1 std')
        ax[0].axhline(y=0.93, color='green', linestyle='--', linewidth=1.0, label='R²=0.93 target')
        ax[0].axhline(y=0,    color='red',   linestyle='--', linewidth=1.0, label='R²=0')
        ax[0].set_xlabel('Number of HVAC Devices')
        # ax[0].set_ylim(-2, 2)
        ax[0].set_ylabel('R²')
        ax[0].set_title(f'R² vs Fleet Size — Training Day {day}')

        ax[1].plot (ns, mape_means, marker='o', color='red', linewidth=1.5)
        ax[1].fill_between(ns,
                        np.array(mape_means) - np.array(mape_stds),
                        np.array(mape_means) + np.array(mape_stds),
                        alpha=0.2, color='steelblue', label='MAPE: ± 1 std')

        ax[1].set_xlabel('Number of HVAC Devices')
        ax[1].set_ylabel('MAPE (%)')
        # ax[1].set_ylim (-10, 100)
        ax[1].set_title(f'MAPE vs Fleet Size — Training Day {day}')

        # handle the legends for the upper and lower plots
        h1, l1 = ax[0].get_legend_handles_labels()
        h2, l2 = ax[1].get_legend_handles_labels()
        ax[0].legend(h1+h2, l1+l2, loc='upper right')
        ax[1].legend()
        ax[1].grid(True)
        ax[0].grid(True)
        plt.tight_layout()
        # plt.savefig(f'r2_vs_fleet_size_trained_on_{day}_days.png')
        # ax[0].legend()
        # ax[0].grid(True)
        # plt.tight_layout()
        # plt.savefig('r2_vs_fleet_size.png')
        # plt.show()