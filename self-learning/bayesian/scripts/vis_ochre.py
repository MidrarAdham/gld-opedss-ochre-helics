# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import bayesian_experiment as bay
import sequential_bayesian as bayesian
import time_changing_theta as theta_bayesian
# %%
with open ('./filtered_dataset.txt', 'r') as f:
    input_paths = f.readlines()

input_paths = [s.strip() for s in input_paths if s.strip()]
input_path = f'/Users/midrar/projects/{Path(*input_paths[1].split("/")[3:])}'
df = pd.read_csv (input_path, usecols=['Time', 'HVAC Heating Electric Power (kW)'])

# df['Time'] = pd.to_datetime (df['Time']).dt.strftime ('%d %H:%M')
df['Time'] = pd.to_datetime (df['Time'])

def create_binary_states (df : pd.DataFrame) -> pd.DataFrame:
    df['state'] = (df[df.columns[1]] >= 0.7).astype(int)
    return df

df = create_binary_states (df=df)

theta_values = np.linspace (0.001, 0.999, 1000)

history = theta_bayesian.initialize_history ()

for hour in range(24):
    print(df)
    df_hour = df[df['Time'].dt.hour == hour]
    alpha, beta_param = 1,1
    H, T, n = theta_bayesian.prepare_data (df=df_hour)
    alpha_post = alpha + H
    beta_post = beta_param + T

    stats = theta_bayesian.calculate_stats (alpha=alpha_post, beta_param=beta_post)
    posterior = bay.calculate_posterior_conjugate (theta_values=theta_values,
                                                   alpha_posterior=alpha_post,
                                                   beta_posterior=beta_post)
    history['hour'].append(hour)
    history['H'].append(H)
    history['T'].append(T)
    history['alpha'].append(alpha_post)
    history['beta'].append(beta_post)
    history['theta_mean'].append(stats['mean'])
    history['theta_std'].append(stats['std'])
    history['ci_lower'].append(stats['ci_lower'])
    history['ci_upper'].append(stats['ci_upper'])

plt.figure(figsize=(12, 6))
plt.plot(history['hour'], history['theta_mean'], linewidth=2)
plt.xlabel('Hour of Day')
plt.ylabel(r'$\theta$ (Probability of ON)')
plt.fill_between (history['hour'], history['ci_lower'], history['ci_upper'] ,alpha=0.3)
plt.title('Heating HVAC Activity Throughout the Day')
plt.grid(True)
plt.savefig ('./time-varying-theta_hvac.png', dpi=150)
plt.show()

# %%
def hvac_sequential_bayesian ():
    with open ('./filtered_dataset.txt', 'r') as f:
        input_paths = f.readlines()

    input_paths = [s.strip() for s in input_paths if s.strip()]
    input_path = f'/Users/midrar/projects/{Path(*input_paths[1].split("/")[3:])}'
    df = pd.read_csv (input_path, usecols=['Time', 'HVAC Heating Electric Power (kW)'])

    df['Time'] = pd.to_datetime (df['Time']).dt.strftime ('%d %H:%M')

    def create_binary_states (df : pd.DataFrame) -> pd.DataFrame:
        df['state'] = (df[df.columns[1]] >= 0.7).astype(int)
        return df

    df = create_binary_states (df=df)

    theta_values = np.linspace (0.001, 0.999, 1000)

    window_size, num_chunks = 10, 30

    history = bayesian.sequential_bayesian_implementation (theta_values=theta_values, df=df,
                                                num_chunks=num_chunks,
                                                window_size=window_size)


    bayesian.plot_sequential_learning(history=history)
    bayesian.plot_posterior_evolution (history=history)
# %%
