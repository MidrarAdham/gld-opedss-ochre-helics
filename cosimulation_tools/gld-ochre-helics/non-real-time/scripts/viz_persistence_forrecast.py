'''
Author: Midrar Adham
Created: Fri May 22 2026
'''
import pandas as pd
import persistant_forecast as pf
import matplotlib.pyplot as plt

wh_dir         = '../results/wh_cosim/'
hvac_dir       = '../results/hvac_cosim/'
total_house_dir = '../results/total_house_consumption/'

chunks_per_day = 2880

horizons = list(range(1,(chunks_per_day // 10)+1))

wh_ground_truth, hvac_ground_truth, feeder_df, wh_df, hvac_df, wh_loader, hvac_loader = pf.load_data (
        chunk_num=chunks_per_day,wh_dir=wh_dir, hvac_dir=hvac_dir, total_house_dir=total_house_dir
        )

hvac_per_device_results = pf.per_device_persistence_forecast (hvac_df=hvac_df, horizons=horizons)



def plot_per_device_persistence(
    hvac_df: dict,
    hvac_per_device_results: dict,
    selected_device: str = '25',
    selected_horizons: list = [1, 6, 12, 24, 72, 144]
    ):

    # ── Get signal for selected device ───────────────────────────────
    filename = f'../results/hvac_cosim/ochre_load_{selected_device}.csv'
    df = hvac_df[filename].copy()
    df['time'] = pd.to_datetime(df['time'])
    power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
    signal = power.resample('10min').mean().values

    # ── Plot 1: R² vs horizon ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for short_name, device_results in hvac_per_device_results.items():
        r2_vals = [device_results[h]['r2'] for h in range(1, 145)]
        ax1.plot(range(1, 145), r2_vals, linewidth=0.8, alpha=0.5)

    # Highlight selected device
    r2_selected = [hvac_per_device_results[selected_device][h]['r2']
                   for h in range(1, 145)]
    ax1.plot(range(1, 145), r2_selected, linewidth=2.0,
             color='black', label=f'Device {selected_device}')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.0)
    ax1.set_xlabel('Horizon (chunks x 10 min)')
    ax1.set_ylabel('R²')
    ax1.set_title('Persistence Forecast R² vs Horizon — Per Device')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig('persistence_r2_vs_horizon.png')
    plt.show()

    # ── Plot 2: Actual vs predicted at selected horizons ─────────────
    fig2, axes = plt.subplots(3, 2,
                              figsize=(15, 10),
                              sharex=False)

    for ax, h in zip(axes.flatten(), selected_horizons):
        y_true, y_pred, r2, mape = pf.persistence_forecast(signal=signal, horizon=h)
        ax.plot(y_true / 1e3, color='black',     linewidth=1.2, label='Actual')
        ax.plot(y_pred / 1e3, color='steelblue', linewidth=1.2,
                linestyle='--', label='Predicted')
        ax.set_ylabel('Power [kW]')
        ax.set_title(f'h={h} ({h*10} min ahead) | R²={r2:.3f} | MAPE={mape:.1f}%')
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if selected_horizons < 6
    for i in range(len(selected_horizons), 6):
        axes.flatten()[i].axis('off')

    fig2.text(0.5, 0.01, 'Chunk Index', ha='center')
    plt.suptitle(f'Device {selected_device} — Actual vs Persistence Forecast', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'persistence_actual_vs_pred_device_{selected_device}.png')
    plt.show()

def fleet_level_persistence_forecast(
    hvac_df: dict,
    wh_df: dict,
    horizons: list,
    ):
    # ── Build HVAC fleet signal ───────────────────────────────────
    hvac_fleet = None
    for filename, df in hvac_df.items():
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        power_resampled = power.resample('10min').mean().values
        if hvac_fleet is None:
            hvac_fleet = power_resampled
        else:
            hvac_fleet += power_resampled

    # ── Build WH fleet signal ─────────────────────────────────────
    wh_fleet = None
    for filename, df in wh_df.items():
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        power = pd.to_numeric(df.set_index('time')['power_out'], errors='coerce')
        power_resampled = power.resample('10min').mean().values
        if wh_fleet is None:
            wh_fleet = power_resampled
        else:
            wh_fleet += power_resampled

    # ── Run persistence at each horizon ──────────────────────────
    hvac_results = {}
    wh_results   = {}

    for h in horizons:
        _, _, r2_hvac, mape_hvac = pf.persistence_forecast(signal=hvac_fleet, horizon=h)
        _, _, r2_wh,   mape_wh   = pf.persistence_forecast(signal=wh_fleet,   horizon=h)
        hvac_results[h] = {'r2': r2_hvac, 'mape': mape_hvac}
        wh_results[h]   = {'r2': r2_wh,   'mape': mape_wh}
        print(f'h={h:3d} | HVAC R²={r2_hvac:.3f} | WH R²={r2_wh:.3f}')

    return hvac_fleet, wh_fleet, hvac_results, wh_results

def plot_fleet_persistence(
    hvac_fleet_results: dict,
    wh_fleet_results: dict,
    horizons: list,
    ):

    r2_hvac = [hvac_fleet_results[h]['r2'] for h in horizons]
    r2_wh   = [wh_fleet_results[h]['r2']   for h in horizons]
    r2_feeder = [feeder_results[h]['r2'] for h in horizons]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(horizons, r2_feeder, color='green', linewidth=1.5, label='Feeder')
    ax.plot(horizons, r2_hvac, color='steelblue', linewidth=1.5, label='HVAC Fleet')
    ax.plot(horizons, r2_wh,   color='orange',    linewidth=1.5, label='WH Fleet')
    ax.axhline(y=0,    color='red',   linestyle='--', linewidth=1.0, label='R²=0')
    ax.axhline(y=0.931, color='black', linestyle='--', linewidth=1.0, label='Bayesian R²=0.931')
    ax.set_xlabel('Horizon (chunks × 10 min)')
    ax.set_ylabel('R²')
    ax.set_title('Fleet-Level Persistence Forecast R² vs Horizon')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('fleet_persistence_r2_vs_horizon.png')
    plt.show()

def feeder_level_persistence_forecast(
    feeder_df: pd.DataFrame,
    horizons: list,
) -> dict:

    signal = pd.to_numeric(
        feeder_df['power_out'], errors='coerce'
    ).values

    feeder_results = {}

    for h in horizons:
        _, _, r2, mape = pf.persistence_forecast(signal=signal, horizon=h)
        feeder_results[h] = {'r2': r2, 'mape': mape}
        print(f'h={h:3d} | Feeder R²={r2:.3f} | MAPE={mape:.1f}%')

    return signal, feeder_results

hvac_fleet, wh_fleet, hvac_fleet_results, wh_fleet_results = fleet_level_persistence_forecast(
    hvac_df=hvac_df,
    wh_df=wh_df,
    horizons=horizons,
)

feeder_signal, feeder_results = feeder_level_persistence_forecast(
    feeder_df=feeder_df,
    horizons=list(range(1, 145)),
)

plot_fleet_persistence(
    hvac_fleet_results=hvac_fleet_results,
    wh_fleet_results=wh_fleet_results,
    horizons=list(range(1, 145)),
)
plot_per_device_persistence (
    hvac_df=hvac_df,
    hvac_per_device_results=hvac_per_device_results)