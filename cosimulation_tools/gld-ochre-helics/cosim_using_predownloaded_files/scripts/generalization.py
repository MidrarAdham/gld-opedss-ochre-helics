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
    '''
    I created this function because I needed to investigate the -ve R2 value. I think it is very straight forward.
    '''
    gt       = np.asarray(gt_data)
    est      = np.asarray(estimated_data)
    gt_mean  = gt.mean()

    ss_res = np.sum((est - gt) ** 2)
    ss_tot = np.sum((gt - gt_mean) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.vlines(x=gt, ymin=gt, ymax=est, color='tomato', alpha=0.6, lw=0.8, label=f'{r"$y - \hat{y}$"}')
    ax.vlines (x=gt, ymin=gt_mean, ymax=gt, color='steelblue', alpha=0.5, lw=0.8, label=f'{r"$y - \bar{y}$"}')


    # scatter and reference lines
    ax.scatter(gt, est, color='black', s=20, zorder=5, label = f'Ground Truth (y) VS Estimated ({r"$\hat{y}$"})')
    lims = [min(gt.min(), est.min()), max(gt.max(), est.max())]

    ax.axhline(gt_mean, color='steelblue', lw=1, ls=':', label=f'{r"$\bar{y}$"} = {gt_mean:.2f}')

    ax.set_xlabel('Ground truth (W)')
    ax.set_ylabel('Estimated (W)')
    ax.set_title(f'R² = {r2:.3f}  |  SS_res={ss_res:.2f}  SS_tot={ss_tot:.2f}')
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
    train_days     = 30
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
    wh_histories   = estimator.fit_many(all_dfs=wh_df) # calculates the beta, mean, alpha, etc parameters
    hvac_histories = estimator.fit_many(all_dfs=hvac_df) # calculates the beta, mean, alpha, etc parameters

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

    '''
    calculate the change in the mean matrix:
    delta M = M_new - M_old
    '''
    delta_M = (hvac_active.diff ()).fillna (0)
    # symmetric color range around zero
    limit = np.max(np.abs(delta_M))

    fig, ax = plt.subplots(figsize=(12, 6))

    mesh = ax.pcolormesh(
        delta_M,
        shading="auto",
        vmin=-limit,
        vmax=limit
    )

    fig.colorbar(mesh, ax=ax, label="Mean change: M_new - M_old")

    ax.set_xlabel("HVAC device")
    ax.set_ylabel("10-minute time step")
    ax.set_title("Difference between new and old mean matrices")

    ax.set_xticks(np.arange(0.5, 28.5, 1))
    ax.set_xticklabels([f"HVAC {i+1}" for i in range(28)], rotation=90)

    plt.tight_layout()
    plt.show()

    time_change = np.sum(np.abs(delta_M), axis=1)

    plt.figure(figsize=(12, 4))

    plt.plot(time_change)

    plt.xlabel("10-minute time step")
    plt.ylabel("Total absolute mean change")
    plt.title("Mean matrix change per time step")

    plt.tight_layout()
    plt.show()

    device_change = np.sum(np.abs(delta_M), axis=0)

    plt.figure(figsize=(10, 4))

    plt.bar(np.arange(1, 27), device_change)

    plt.xlabel("HVAC device")
    plt.ylabel("Total absolute mean change")
    plt.title("Mean matrix change per HVAC device")

    plt.tight_layout()
    plt.show()
    