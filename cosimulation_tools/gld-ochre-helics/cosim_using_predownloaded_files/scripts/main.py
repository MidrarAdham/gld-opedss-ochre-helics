'''
Author: Midrar Adham
Created: Sat May 02 2026
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import DataLoader
from ols import OrdinaryLeastSquare
from bayesian_estimator import BayesianEstimator


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_pred = np.asarray(y_pred, dtype=float)
    return 1 - np.sum((y_pred - y_true) ** 2) / \
               np.sum((y_true - y_true.mean()) ** 2)


def get_ground_truth_per_device(all_dfs: dict) -> pd.Series:
    """
    Compute mean ON-power per device from raw per-device dataframes.

    Returns
    -------
    pd.Series
        Index: filenames, values: mean power_out when state == True (Watts).
    """
    gt = {}
    for filename, df in all_dfs.items():
        on_power = df.loc[df['state'] == True, 'power_out']
        gt[filename] = on_power.mean() if len(on_power) > 0 else 0.0
    return pd.Series(gt)


def build_active_ground_truth(
        hvac_active_cols: list,
        hvac_all_dfs: dict,
        ) -> np.ndarray:
    """
    Build a ground truth total demand time series summed only over
    the active devices used in per-device OLS estimation.

    This ensures a fair comparison — excluded devices (e.g. two-state
    devices) are not included in the ground truth sum either.

    Parameters
    ----------
    hvac_active_cols : list
        Filenames of active HVAC devices from per_d_hvac_active.columns.
    hvac_all_dfs : dict
        Output of DataLoader.all_dfs for HVAC data.

    Returns
    -------
    np.ndarray
        Total demand in Watts at each 10-minute chunk.
    """
    gt_total = None

    for filename in hvac_active_cols:
        df = hvac_all_dfs[filename].copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        power_resampled = power.resample('10min').mean().values

        if gt_total is None:
            gt_total = power_resampled
        else:
            gt_total += power_resampled

    return gt_total if gt_total is not None else np.array([])


if __name__ == '__main__':

    wh_dir         = '../results/wh_cosim/'
    hvac_dir       = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    # Two-state devices excluded from per-device OLS
    # (violate the assumption that power_out = 0 when OFF)
    EXCLUDE_HVAC = [
        '../results/hvac_cosim/ochre_load_16.csv',
    ]
    chunks_per_day = 144
    # day_ends = [1440, 2880, 4320, 5760, 7200, 8640, 10080, 11520, 12960, 14400, 15840, 17280, 18720, 20160, 21600, 23040, 24480, 25920, 27360, 28800]
    # day_ends = [1440, 2880, 4320, 5760, 7200, 8640, 17280]
    day_ends = [1440, 17280]
    lambdas = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
    sweep_results = {}

    for day_end in day_ends:
        n_days = day_end // 1440
        n_chunks = n_days * chunks_per_day
        sweep_results [day_end] = {}
        # ── Load 10 days of data ─────────────────────────────────────────────
        wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=day_end)
        hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=day_end)
        total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=day_end)

        wh_ground_truth   = wh_loader.load_transformer_data()
        hvac_ground_truth = hvac_loader.load_transformer_data()
        feeder_df         = total_house_loader.load_transformer_data()

        wh_df   = wh_loader.load_csv_files(threshold=5000.0)
        hvac_df = hvac_loader.load_csv_files(threshold=100.0)

        # ── Bayesian estimation — continuous across 10 days ─────────────────
        for lamb in lambdas:
            estimator = BayesianEstimator(num_chunks=n_chunks, discount=lamb)
            wh_histories   = estimator.fit_many(all_dfs=wh_df)
            hvac_histories = estimator.fit_many(all_dfs=hvac_df)


            # ── Diagnose day 15 collapse ─────────────────────────────────
            if n_days == 15 and lamb == 0.01:
                for filename, history in hvac_histories.items():
                    mean_arr = np.array(history['mean'])
                    alpha_arr = np.array(history['alpha'])
                    beta_arr  = np.array(history['beta'])
                    corrupted = np.any(np.isnan(mean_arr)) or np.any(np.isinf(mean_arr))
                    short_name = filename.split('ochre_load_')[1].replace('.csv', '')
                    print(f"Device {short_name:>4} | "
                        f"mean[-1]: {mean_arr[-1]:.6f} | "
                        f"alpha[-1]: {alpha_arr[-1]:.2f} | "
                        f"beta[-1]: {beta_arr[-1]:.2f} | "
                        f"corrupted: {corrupted}")

            # ── OLS ──────────────────────────────────────────────────────────────
            ols = OrdinaryLeastSquare(
                feeder_demand  = feeder_df,
                wh_histories   = wh_histories,
                hvac_histories = hvac_histories,
                wh_all_dfs     = wh_loader.all_dfs,
                hvac_all_dfs   = hvac_loader.all_dfs,
            )

            results = ols.run(exclude_hvac=EXCLUDE_HVAC)

            # ── Per-device rated power comparison ────────────────────────────────
            gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)

            comparison = pd.DataFrame({
                'estimated_W': results['per_d_kw_hvac'],
                'truth_W':     gt_per_device,
            })
            comparison['error_pct'] = (
                (comparison['estimated_W'] - comparison['truth_W'])
                / comparison['truth_W'] * 100
            ).round(1)
            comparison = comparison.sort_values('truth_W', ascending=False)

            print('\n── Per-device HVAC rated power ──────────────────────────')
            # print(comparison.to_string())

            # ── Total estimated vs active ground truth ───────────────────────────
            hvac_active     = results['per_d_hvac_active']
            kw_per_device   = results['per_d_kw_hvac']
            estimated_total = hvac_active.values @ kw_per_device[hvac_active.columns].values

            gt_active_total = build_active_ground_truth(
                hvac_active_cols=hvac_active.columns.tolist(),
                hvac_all_dfs=hvac_loader.all_dfs,
            )

            r2   = r2_score(gt_active_total, estimated_total)
            mask = gt_active_total > 0
            # mape = np.mean(np.abs((estimated_total - gt_active_total) / gt_active_total) * 100)
            mape = np.mean(np.abs((estimated_total[mask] - gt_active_total[mask]) / gt_active_total[mask]) * 100)

            sweep_results[day_end][lamb] = {'r2': r2, 'mape': mape}

            print(f'days={n_days:2d} | lambda={lamb:.2f} | R²={r2:.3f} | MAPE={mape:.1f}%')

    # ── Plot R² vs days for each lambda ──────────────────────────────────
    rows = []
    for day_end, lambda_results in sweep_results.items():
        for lamb, metrics in lambda_results.items():
            rows.append({
                'days':  day_end // 1440,
                'lambda': lamb,
                'r2':    metrics['r2'],
                'mape':  metrics['mape'],
            })

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv('sweep_days_lambda_long_term_estimation.csv', index=False)
    print('\nSweep results saved to sweep_days_lambda_long_term_estimation.csv')

    days = [d // 1440 for d in day_ends]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lamb in lambdas:
        r2_vals   = [sweep_results[d][lamb]['r2']   for d in day_ends]
        mape_vals = [sweep_results[d][lamb]['mape'] for d in day_ends]
        axes[0].plot(days, r2_vals,   marker='o', label=f'λ={lamb}')
        axes[1].plot(days, mape_vals, marker='o', label=f'λ={lamb}')

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
    plt.savefig('sweep_days_lambda_long_term_estimation.png')
    plt.show()

            # # ── Plot ─────────────────────────────────────────────────────────────
            # fig, ax = plt.subplots(figsize=(12, 5))
            # ax.plot(gt_active_total / 1e3, color='black',     linewidth=1.5,
            #         label='Ground Truth (active devices)')
            # ax.plot(estimated_total / 1e3, color='steelblue', linewidth=1.5,
            #         linestyle='--', label='Estimated')
            # ax.set_ylabel('HVAC Demand [kW]')
            # ax.set_xlabel('Chunk Index (10-min intervals)')
            # ax.set_title(f'Total HVAC Demand: Estimated vs Ground Truth — Active Devices Only - lambda: {lamb}')
            # ax.annotate(
            #     f'R²: {r2:.3f}\nMAPE: {mape:.1f}%',
            #     xy=(0.01, 0.95), xycoords='axes fraction',
            #     fontsize=10, verticalalignment='top',
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            # )
            # ax.legend()
            # plt.tight_layout()
            # plt.savefig (f"results_lambda_{lamb}.png")
            # plt.show()