'''
Author: Midrar Adham
Benchmark: Naive Persistence Forecasting
Purpose: Evaluate persistence forecast (ŷ(t+1) = y(t)) at multiple
         aggregation levels and compare with the Bayesian estimator results.
'''

# ── Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from data_loader import DataLoader

# ── Helper functions ──────────────────────────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    y_true = pd.to_numeric(y_true, errors='coerce')
    if len(y_true) == 0:
        return 0.0
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum ((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean ()) **2 )
    return 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

def mape_score (y_true, y_pred):
    mask = y_true > 0 # avoid dividing by zero
    if mask.sum () == 0:
        return np.nan
    mape = np.mean (np.abs((y_pred[mask] - y_true[mask]) / (y_true[mask])) * 100)
    return mape

def persistence_forecast (signal, horizon):    
    """
    Given a signal array, compute persistence forecast at a given horizon.
    
    Parameters
    ----------
    signal : np.ndarray
    horizon : int — number of chunks to shift
    
    Returns
    -------
    y_true, y_pred, r2, mape
    """
    y_true = signal[horizon:]
    y_pred = signal[:-horizon]
    r2 = r2_score (y_true=y_true, y_pred=y_pred)
    mape = mape_score (y_true=y_true, y_pred=y_pred)

    return y_true, y_pred, r2, mape

def load_data (chunk_num, wh_dir, hvac_dir, total_house_dir):

    wh_loader          = DataLoader(results_dir=wh_dir,          day_start=0, day_end=chunk_num)
    hvac_loader        = DataLoader(results_dir=hvac_dir,        day_start=0, day_end=chunk_num)
    total_house_loader = DataLoader(results_dir=total_house_dir, day_start=0, day_end=chunk_num)


    wh_ground_truth   = wh_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data()
    feeder_df         = total_house_loader.load_transformer_data()

    wh_df   = wh_loader.load_csv_files(threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files(threshold=100.0)

    return wh_ground_truth, hvac_ground_truth, feeder_df, wh_df, hvac_df, wh_loader, hvac_loader

def per_device_persistence_forecast (hvac_df : pd.DataFrame, horizons : list):
    hvac_per_device_results = {}

    for filename, df in hvac_df.items ():
        df = df.copy () # this has a one-minute resolution
        df['time'] = pd.to_datetime(df['time']) 
        power = pd.to_numeric (df.set_index ('time')['power_out'], errors='coerce')
        power_resampled = power.resample('10min').mean().values # now this is a 10-minute chunks, so we have 144 chunks
        device_results = {}
        for h in horizons:
            _,_,r2, mape = persistence_forecast (signal=power_resampled, horizon=h)
            device_results[h] = {'r2':round(r2, 2), 'mape':round(mape, 2)}
        short_name = filename.split('ochre_load_')[1].replace('.csv', '')
        hvac_per_device_results[short_name] = device_results
        # print(f'Device {short_name:>4} | h=1 R²={device_results[1]["r2"]:.3f} | h=144 R²={device_results[144]["r2"]:.3f}')
    
    return hvac_per_device_results