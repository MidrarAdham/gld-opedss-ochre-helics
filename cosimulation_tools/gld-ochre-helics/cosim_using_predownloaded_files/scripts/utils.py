'''
Author: Midrar Adham
Shared utility functions used across all scripts.
'''
import numpy as np
import pandas as pd


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R² score. Returns 0.0 if ss_tot is near zero.
    """
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    # return 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).
    Ignores zero values in y_true to avoid division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAPE in percent. Returns np.nan if no valid values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100)


def get_ground_truth_per_device(all_dfs: dict) -> pd.Series:
    """
    Compute mean ON-power per device from raw per-device dataframes.

    Parameters
    ----------
    all_dfs : dict
        Output of DataLoader.all_dfs.

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

    Parameters
    ----------
    hvac_active_cols : list
        Filenames of active HVAC devices.
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
        power = pd.to_numeric(
            df.set_index('time')['power_out'], errors='coerce'
        )
        power_resampled = power.resample('10min').mean().values

        if gt_total is None:
            gt_total = power_resampled
        else:
            gt_total += power_resampled

    return gt_total if gt_total is not None else np.array([])
