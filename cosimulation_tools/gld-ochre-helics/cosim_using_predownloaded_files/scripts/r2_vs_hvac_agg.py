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
    train_days   = 1
    n_repeats    = 10
    # fleet_sizes  = [1, 2, 3, 5, 6,7,8,9,10,11,12,13,14,15,16,17, 18, 19,20, 21, 22, 23, 24, 25, 26, 27]
    # fleet_sizes  = [13, 26, 17, 19, 5, 1, 18, 12, 2, 23, 4, 6]
    LAMBDA       = 0.01
    exclude_hvac = ['../results/hvac_cosim/ochre_load_16.csv']

    day_end  = train_days * 1440
    n_chunks = train_days * 144

    # ── Load data once ───────────────────────────────────────────────────
    wh_loader          = DataLoader(results_dir=wh_dir,          day_start=3*1440, day_end=4*1440)
    hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=3*1440, day_end=4*1440)
    total_house_loader = DataLoader(results_dir=total_house_dir, day_start=3*1440, day_end=4*1440)

    wh_loader.load_transformer_data()
    hvac_loader.load_transformer_data()
    feeder_df = total_house_loader.load_transformer_data()

    wh_df   = wh_loader.load_csv_files(threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files(threshold=100.0)

    # ── Bayesian estimation ──────────────────────────────────────────────
    estimator      = BayesianEstimator(num_chunks=n_chunks, discount=LAMBDA)
    wh_histories   = estimator.fit_many(all_dfs=wh_df)
    hvac_histories = estimator.fit_many(all_dfs=hvac_df)

    # ── Get all active devices ───────────────────────────────────────────
    all_devices = [f for f in hvac_loader.all_dfs.keys() if f not in exclude_hvac]

    # ── Filter to 1-ton and above devices ────────────────────────────────
    one_ton_w = 3517  # 1 ton in Watts

    gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)

    # all_devices = [f for f in hvac_loader.all_dfs.keys() if f not in exclude_hvac and gt_per_device[f] >= one_ton_w]
    all_devices = [f for f in hvac_loader.all_dfs.keys() if f not in exclude_hvac]

    print(f'Devices above 1 ton: {len(all_devices)}')
    for d in all_devices:
        short = d.split('ochre_load_')[1].replace('.csv', '')
        print(f'  Device #{short}: {gt_per_device[d]/1000:.1f} kW')

    # Update fleet sizes to match filtered device count
    fleet_sizes = list(range(1, len(all_devices) + 1))

    print(f'Total active devices: {len(all_devices)}')

    # ── Fleet size sweep ─────────────────────────────────────────────────
    sweep_results = {}

    # for N, device_name in enumerate(all_devices):
    #     print(N)
    #     quit()
    for N, device in enumerate(all_devices, start=1):
        r2_list   = []
        mape_list = []
        device_name = []

        for repeat in range(n_repeats):
            sampled_devices = random.sample(all_devices, N)
            

            ols = OrdinaryLeastSquare(
                feeder_demand  = feeder_df,
                wh_histories   = wh_histories,
                hvac_histories = hvac_histories,
                wh_all_dfs     = wh_loader.all_dfs,
                hvac_all_dfs   = hvac_loader.all_dfs,
            )

            results = ols.run(
                exclude_hvac   = exclude_hvac,
                subset_devices = sampled_devices,
            )

            hvac_active     = results['per_d_hvac_active']
            # Skip if no active devices after filtering the posterior mean, see Bayesian class.
            if hvac_active.shape[1] == 0:
                print(f'N={N} repeat={repeat} - Skipped, no active devices')
                continue

            kw_per_device   = results['per_d_kw_hvac']
            estimated_total = hvac_active.values @ kw_per_device[hvac_active.columns].values

            gt_active_total = build_active_ground_truth(
                hvac_active_cols=hvac_active.columns.tolist(),
                hvac_all_dfs=hvac_loader.all_dfs,
            )

            r2   = r2_score(gt_active_total, estimated_total)
            mape = mape_score(gt_active_total, estimated_total)
            short_device = device.split('ochre_load_')[1].replace('.csv', '')


            r2_list.append(r2)
            mape_list.append(mape)
            device_name.append(short_device)
            # quit()
            device_name = list(set(device_name))
        
        if N == 5:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(gt_active_total / 1e3,   color='black',     label='Ground Truth')
            ax.plot(estimated_total / 1e3,   color='steelblue', linestyle='--', label='Estimated')
            ax.set_title(f'N=5 | R²={r2:.3f} | devices: {[d.split("ochre_load_")[1].replace(".csv","") for d in sampled_devices]}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout ()
            plt.savefig ('./testing_five_devices.png')
            plt.show()
            # quit()

        sweep_results[N] = {
            'r2_mean':   np.mean(r2_list),
            'r2_std':    np.std(r2_list),
            'mape_mean': np.mean(mape_list),
            'mape_std':  np.std(mape_list),
            'devices_idx' : device_name[0]
            }
        print(f'N={N:2d} | R2={sweep_results[N]["r2_mean"]:.3f} ± {sweep_results[N]["r2_std"]:.3f} | '
              f'MAPE={sweep_results[N]["mape_mean"]:.1f}% ± {sweep_results[N]["mape_std"]:.1f}%')
        

    # ── Save results ─────────────────────────────────────────────────────
    rows = []
    for N, metrics in sweep_results.items():
        # print(f'N={N:2d} | R²={metrics["r2_mean"]:.3f} ± {metrics["r2_std"]:.3f}')
        rows.append({'N': N, **metrics})
    pd.DataFrame(rows).to_csv('r2_vs_fleet_size.csv', index=False)
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
    ax[0].set_ylim(-2, 2)
    ax[0].set_ylabel('R²')
    ax[0].set_title(f'R² vs Fleet Size — Training Day {train_days}')

    ax[1].plot (ns, mape_means, marker='o', color='red', linewidth=1.5)
    ax[1].fill_between(ns,
                    np.array(mape_means) - np.array(mape_stds),
                    np.array(mape_means) + np.array(mape_stds),
                    alpha=0.2, color='steelblue', label='MAPE: ± 1 std')

    ax[1].set_xlabel('Number of HVAC Devices')
    ax[1].set_ylabel('MAPE (%)')
    ax[1].set_ylim (-100, 1000)
    ax[1].set_title(f'MAPE vs Fleet Size — Training Day {train_days}')

    # handle the legends for the upper and lower plots
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax[1].get_legend_handles_labels()
    ax[0].legend(h1+h2, l1+l2, loc='upper left')
    ax[1].legend()
    ax[1].grid(True)
    ax[0].grid(True)
    plt.tight_layout()
    plt.savefig('r2_vs_fleet_size.png')
    # ax[0].legend()
    # ax[0].grid(True)
    # plt.tight_layout()
    # plt.savefig('r2_vs_fleet_size.png')
    plt.show()