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

def evaluate_metrics(gt_data, estimated_data):
    gt       = np.asarray(gt_data)
    est      = np.asarray(estimated_data)
    gt_mean  = gt.mean()

    ss_res = np.sum((est - gt) ** 2)
    ss_tot = np.sum((gt - gt_mean) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.vlines(x=gt, ymin=gt, ymax=est, color='tomato', alpha=0.6, lw=0.8, label=f'{r"$y - \hat{y}$"}')
    # for g in gt:
    #     ax.plot([g, g], [g, gt_mean], color='steelblue', alpha=0.4, lw=0.8, label=f'{r"$y - \bar{y}$"}')

    # for g, e in zip(gt, est):
        # ax.plot([g, g], [g, e], color='tomato', alpha=0.6, lw=0.8, label=f'{r"$y - \hat{y}$"}')
    # draw ss_tot segments: each gt point → mean(gt)  [blue, total variance]
    ax.vlines (x=gt, ymin=gt_mean, ymax=gt, color='steelblue', alpha=0.5, lw=0.8, label=f'{r"$y - \bar{y}$"}')
    # ax.vlines(gt, gt, gt_mean, color='steelblue', alpha=0.4, lw=0.8)


    # scatter and reference lines
    ax.scatter(gt, est, color='black', s=20, zorder=5, label = f'Ground Truth (y) VS Estimated ({r"$\hat{y}$"})')
    lims = [min(gt.min(), est.min()), max(gt.max(), est.max())]
    # ax.plot(lims, lims, 'k--', lw=1, label='perfect (y = x)')
    ax.axhline(gt_mean, color='steelblue', lw=1, ls=':', label=f'{r"$\bar{y}$"} = {gt_mean:.2f}')

    # proxy patches for legend
    # from matplotlib.patches import Patch
    # ax.legend(handles=[
    #     ax.get_lines()[0],
    #     ax.get_lines()[-1],
    #     Patch(color='steelblue', alpha=0.5, label=f'SS_tot = {ss_tot:.2f}'),
    #     Patch(color='tomato',    alpha=0.6, label=f'SS_res = {ss_res:.2f}'),
    #     ], loc='upper left')

    ax.set_xlabel('Ground truth (W)')
    ax.set_ylabel('Estimated (W)')
    ax.set_title(f'R² = {r2:.3f}  |  SS_res={ss_res:.2f}  SS_tot={ss_tot:.2f}')
    # ax.set_aspect('equal')
    plt.legend ()
    plt.tight_layout()
    plt.savefig ('./r2_investigation_best_case.png', dpi=300)
    plt.show()
    

def analyze_metrics (hvac_dir : str, mean_matrix : pd.DataFrame, per_d_ols_coefficients : pd.DataFrame):
    hvac_loader = DataLoader (
        results_dir=hvac_dir,
        day_start=0,
        day_end=1440,
    )
    target_hvac = '../results/hvac_cosim/ochre_load_25.csv'

    estimated_hvac = mean_matrix.multiply (per_d_ols_coefficients[target_hvac])[target_hvac]
    gt_hvac = hvac_loader.load_csv_files (threshold=100.0)[target_hvac]
    gt_hvac['time'] = pd.to_datetime (gt_hvac['time'])
    power = pd.to_numeric(gt_hvac.set_index ('time')['power_out'], errors='coerce')
    power_resampled = power.resample ('10min').mean ().values

    estimated_hvac = estimated_hvac.tolist()[:144]
    x_axis = list(range(1, len(power_resampled) + 1))

    # calculate r2 for the estiamated and ground truth data:

    evaluate_metrics (gt_data=power_resampled, estimated_data=estimated_hvac)
    # print(hvac_state_matrix)
    
    # quit()

if __name__ == '__main__':

    wh_dir          = '../results/wh_cosim/'
    hvac_dir        = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    # ── Configuration ────────────────────────────────────────────────────
    train_days     = 2
    future_days    = list(range(3, 10))
    LAMBDA         = 0.01
    chunks_per_day = 144
    exclude_hvac   = [
        '../results/hvac_cosim/ochre_load_16.csv',
    ]

    train_day_end = train_days * 1440
    n_chunks      = train_days * chunks_per_day
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

    results       = ols.run(exclude_hvac=exclude_hvac)
    hvac_active   = results['per_d_hvac_active'] # the bayesian estimated mean matrix
    kw_per_device = results['per_d_kw_hvac'] # estimated kW for each device per two days.
    active_cols   = hvac_active.columns
    kw_trained    = kw_per_device[active_cols].values # the ols coefficients
    gt_per_device = get_ground_truth_per_device(hvac_loader.all_dfs)
    
    print('\nmean matrix training is done\n')

    # analyze_metrics (hvac_dir=hvac_dir, mean_matrix=hvac_active, per_d_ols_coefficients=kw_per_device)

    # ── Future day prediction ────────────────────────────────────────────

    future_results = {}

    for future_day in future_days:
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

        r2 = r2_score(future_gt, future_estimated)
        mape_future = mape_score(future_gt, future_estimated)

        future_results[future_day] = {
            'r2':              r2,
            'mape':            mape_future,
            'future_gt':       future_gt,
            'future_estimated': future_estimated,
            'state_matrix':    future_state_matrix,
        }

        print(f'day={future_day:2d} | R²={r2:.3f} | MAPE={mape_future:.1f}%')

    evaluate_metrics(
        gt_data       = future_results[4]['future_gt'],
        estimated_data = future_results[4]['future_estimated'],
        )
    