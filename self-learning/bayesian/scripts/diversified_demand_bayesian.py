import numpy as np
import pandas as pd
# My functions
import bayesian_plots as my_vis
import bayesian_experiment as my_bayesian
import sequential_bayesian as my_seq_bayesian
# %%
def collect_profiles_in_one_df(input_paths):
    x = []

    for input_path in input_paths:
        bldg_id = input_path.split('/')[-3]
        df = my_bayesian.load_wh_data(filepath=input_path)
        df = df.head(1440)

        time_col = df['Time']
        df = df.drop('Time', axis=1)

        # Rename columns to include building ID
        df.columns = [f"bldg_{bldg_id}_{col}" for col in df.columns]

        x.append(df)

    dfs = pd.concat(x, axis=1)
    dfs['Time'] = pd.to_datetime(time_col)

    return dfs

def calculate_diversified_demand (dfs):
    time_col = dfs['Time']
    dfs = dfs.drop('Time',axis=1)

    row_sums = dfs.sum (axis=1)
    dfs['diversified_demand (kW)'] = row_sums
    dfs['Time'] = time_col.to_list()
    return dfs


# %%
if __name__ == "__main__":
    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
    
    input_paths = [s.strip() for s in input_paths if s.strip()]

    time_changing_theta_history = {}
    sequential_bayesian_history = {}

    dfs = collect_profiles_in_one_df (input_paths=input_paths)

    dfs = calculate_diversified_demand (dfs=dfs)
    dfs['state'] = (dfs['diversified_demand (kW)'] > 5).astype(int)
    window_size, num_chunks = 10, 144
    theta_values = np.linspace (0.001, 0.999, 1000)

    seq_step_history = my_seq_bayesian.sequential_bayesian_implementation (theta_values=theta_values,
                                                    df=dfs, num_chunks=num_chunks,
                                                    window_size=window_size)
    
    theta_step_history = my_seq_bayesian.time_varying_theta (theta_values=theta_values, df=dfs)
    
    sequential_bayesian_history = {'portfolio' : seq_step_history}
    time_changing_theta_history = {'portfolio' : theta_step_history}

    

    # my_vis.plot_evolution_comparison (all_histories=sequential_bayesian_history)
    my_vis.plot_all_posteriors_detailed (all_histories=sequential_bayesian_history)
    # my_vis.plot_uncertainty_by_hour (all_history=time_changing_theta_history)
    # my_vis.plot_heterogeneity_distribution (theta_values=theta_values, fitted_params=)