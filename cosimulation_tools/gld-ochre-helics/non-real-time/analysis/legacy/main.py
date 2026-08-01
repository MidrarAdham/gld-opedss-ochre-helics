'''
Author: Midrar Adham
Created: Sat May 02 2026
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
from ols import OrdinaryLeastSquare
from proposal_viz import ProposalFigures
from bayesian_estimator import BayesianEstimator

def build_binned_lookup(x_hvac, hvac_truth, bin_width=0.5, min_count=4):
    """
    Build lookup table:

        x_hvac bin -> mean HVAC power

    Only bins with at least min_count samples are considered reliable.
    """

    df = pd.DataFrame({
        "x_hvac": x_hvac,
        "hvac_truth": hvac_truth
    })

    df["bin"] = (df["x_hvac"] / bin_width).round() * bin_width

    summary = df.groupby("bin")["hvac_truth"].agg(
        mean="mean",
        count="count"
    )

    reliable_lookup = summary[summary["count"] >= min_count]["mean"]

    return reliable_lookup, summary


def predict_from_binned_lookup(x_hvac, reliable_lookup, bin_width=0.5):
    """
    Predict HVAC power using nearest reliable bin.
    """

    bins = (x_hvac / bin_width).round() * bin_width
    reliable_bins = reliable_lookup.index.values

    predictions = []

    for b in bins:
        nearest_bin = reliable_bins[
            np.argmin(np.abs(reliable_bins - b))
        ]

        predictions.append(reliable_lookup.loc[nearest_bin])

    return np.array(predictions)


def r2_score_manual(y_true, y_pred):
    return 1 - np.sum((y_pred - y_true) ** 2) / np.sum(
        (y_true - y_true.mean()) ** 2
    )


def r2_score(y_true, y_pred):
    y_true = pd.to_numeric(y_true, errors='coerce').values
    y_pred = np.asarray(y_pred, dtype=float)
    return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


if __name__ == '__main__':

    wh_dir = '../results/wh_cosim/'
    hvac_dir = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    # ── Load data ────────────────────────────────────────────────────
    wh_loader = DataLoader(results_dir=wh_dir, day_start=0, day_end=14400)
    hvac_loader = DataLoader(results_dir=hvac_dir, day_start=0, day_end=14400)
    total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=14400)

    wh_ground_truth = wh_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data()
    feeder_df = total_house_loader.load_transformer_data()
    time_col = pd.to_datetime(wh_ground_truth['Time']).dt.strftime('%H:%M')

    # State thresholds used before Bayesian estimation
    wh_df = wh_loader.load_csv_files(threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files(threshold=100.0)

    # ── Bayesian estimation, low discount ────────────────────────────
    estimator_low = BayesianEstimator (num_chunks=1440)
    wh_histories_low = estimator_low.fit_many(all_dfs=wh_df)
    hvac_histories_low = estimator_low.fit_many(all_dfs=hvac_df)

    # ── OLS object ───────────────────────────────────────────────────
    ols = OrdinaryLeastSquare(
        feeder_demand=feeder_df,
        wh_histories=wh_histories_low,
        wh_all_dfs=wh_loader.all_dfs,
        hvac_histories=hvac_histories_low,
        hvac_all_dfs=hvac_loader.all_dfs,
    )

    # filename = list(hvac_loader.all_dfs.keys())[0]
    # print(hvac_loader.all_dfs[filename].columns.tolist())
    # print(hvac_loader.all_dfs[filename].head())
    # quit()
    hvac_truth = pd.to_numeric(
        hvac_ground_truth["power_out"],
        errors="coerce"
        ).values
    
    results_low = ols.run()
    # viz = ProposalFigures (time_col=time_col)
    # viz.fig12_bar_chart_for_each_device (
    #     estimated_data=results_low['per_d_kw_hvac'],
    #     ground_truth_data=hvac_loader.all_dfs)

    # quit()
    
    # Get ground truth from raw data
    gt_dict = {}
    for filename, df in hvac_loader.all_dfs.items():
        if filename != '../results/hvac_cosim/ochre_load_16.csv':
            print(filename)
            on_power = df.loc[df['state'] == True, 'power_out']
            if len(on_power) > 0:
                gt_dict[filename] = on_power.mean()
            else:
                gt_dict[filename] = 0.0

    # quit()
    gt_series = pd.Series(gt_dict)
    # Filter to devices above 1kW ground truth
    # gt_series = pd.Series(hvac_ground_truth_per_device)

    comparison = pd.DataFrame({
        'estimated_W': results_low['per_d_kw_hvac'],
        'truth_W':     gt_series,
    })

    # Keep only devices where truth > 1000 W
    comparison = comparison[comparison['truth_W'] > 1000]
    comparison['error_pct'] = (
        (comparison['estimated_W'] - comparison['truth_W'])
        / comparison['truth_W'] * 100
    ).round(1)

    comparison = comparison.sort_values('truth_W', ascending=False)

    # ── Total estimated vs ground truth ──────────────────────────
    hvac_active = results_low['hvac_active']
    print('\n\nhvac active devices\n\n')
    print(results_low['hvac_active'])
    kw_per_device = results_low['per_d_kw_hvac']

    # Estimated total: Σ kw_i × mean_i(t) for all active devices
    estimated_total = hvac_active.values @ kw_per_device[hvac_active.columns].values

    # Ground truth total from transformer
    hvac_truth = pd.to_numeric(hvac_ground_truth['power_out'], errors='coerce').values

    # Fair ground truth — only active devices
    gt_active_total = np.zeros(len(hvac_active))

    for filename in hvac_active.columns:
        df = hvac_loader.all_dfs[filename].copy()
        df['time'] = pd.to_datetime(df['time'])
        device_power = df.set_index('time')['power_out']
        device_power = pd.to_numeric(device_power, errors='coerce')
        device_power_resampled = device_power.resample('10min').mean().values
        gt_active_total += device_power_resampled

    # Metrics
    r2 = 1 - np.sum((estimated_total - gt_active_total) ** 2) / \
            np.sum((gt_active_total - gt_active_total.mean()) ** 2)
    
    mape = np.mean(np.abs((estimated_total - gt_active_total) / gt_active_total) * 100)

    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gt_active_total / 1e3, color='black',     linewidth=1.5, label='Ground Truth (active devices)')
    ax.plot(estimated_total / 1e3, color='steelblue', linewidth=1.5,
            linestyle='--', label='Estimated')
    ax.set_ylabel('HVAC Demand [kW]')
    ax.set_xlabel('Chunk Index (10-min intervals)')
    ax.set_title('Total HVAC Demand: Estimated vs Ground Truth — Active Devices Only')
    ax.annotate(f'R²: {r2:.3f}\nMAPE: {mape:.1f}%',
                xy=(0.01, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # # Metrics
    # r2 = 1 - np.sum((estimated_total - hvac_truth) ** 2) / \
    #         np.sum((hvac_truth - hvac_truth.mean()) ** 2)
    # mape = np.mean(np.abs((estimated_total - hvac_truth) / hvac_truth) * 100)

    # print(f"R²:   {r2:.3f}")
    # print(f"MAPE: {mape:.1f}%")

    # # Plot
    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.plot(hvac_truth      / 1e3, color='black',    linewidth=1.5, label='Ground Truth')
    # ax.plot(estimated_total / 1e3, color='steelblue', linewidth=1.5, 
    #         linestyle='--', label='Estimated')
    # ax.set_ylabel('HVAC Demand [kW]')
    # ax.set_xlabel('Chunk Index (10-min intervals)')
    # ax.set_title('Total HVAC Demand: Estimated vs Ground Truth (10 days)')
    # ax.annotate(f'R²: {r2:.3f}\nMAPE: {mape:.1f}%',
    #             xy=(0.01, 0.95), xycoords='axes fraction',
    #             fontsize=10, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # Plot
    # import numpy as np

    # labels = [f.split('ochre_load_')[1].replace('.csv','') 
    #         for f in comparison.index]
    # x = np.arange(len(labels))
    # width = 0.35

    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.bar(x - width/2, comparison['truth_W']     / 1000, width, label='Ground Truth', color='black',  alpha=0.7)
    # ax.bar(x + width/2, comparison['estimated_W'] / 1000, width, label='Estimated',    color='steelblue', alpha=0.7)

    # ax.set_ylabel('Rated Power [kW]')
    # ax.set_xlabel('HVAC Device')
    # ax.set_title('Per-Device HVAC Rated Power: Estimated vs Ground Truth')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()