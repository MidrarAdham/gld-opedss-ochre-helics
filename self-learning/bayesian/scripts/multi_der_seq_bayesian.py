# %%
"""
This script runs sequential Bayesian and time-varying theta for multiple DERs (WHs).
"""
# %%
# pypi packages
import numpy as np
# My functions:
import bayesian_plots as my_baysian_vis
import heterogenity_modeling as my_pred
import bayesian_experiment as my_bayesian
import sequential_bayesian as my_seq_bayesian
import time_changing_theta as my_varying_theta

# %%
def plot_comparison_all_wh(all_histories):
    """
    Master comparison function - calls all other comparison functions
    """
    my_baysian_vis.plot_final_theta_comparison(all_histories)
    my_baysian_vis.plot_evolution_comparison(all_histories)
    my_baysian_vis.plot_all_posteriors_detailed (all_histories=all_histories)

def time_varying_theta_plots(all_history, theta_values):
    """
    Master as well - Generate all time-varying theta plots for multiple WHs
    
    :param all_history: Dict of {wh_name: history_dict}
    :param theta_values: Array for PDF calculation
    """
    my_baysian_vis.plot_theta_by_hour(all_history)
    my_baysian_vis.plot_posterior_evolution_by_hour(all_history, theta_values)
    # my_baysian_vis.plot_data_availability(all_history)
    my_baysian_vis.plot_uncertainty_by_hour(all_history)
# %%
if __name__ == '__main__':

    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
        input_paths = [s.strip() for s in input_paths if s.strip()]

     # Get the dataset input files:
    all_histories_seq = {}
    all_histories_time_varying_theta = {}

    # Set the theta values:
    theta_values = np.linspace (0.001, 0.999, 1000)

    # define loop parameters:
    window_size, num_chunks = 10, 144

    for idx, input_file in enumerate (input_paths):

        # Read the input files
        df = my_bayesian.load_wh_data (filepath=input_file)

        # Create binary states
        df = my_bayesian.create_binary_states (df=df, threshold=0.5)
        
        # Run the bayesian implementation:
        seq_history = my_seq_bayesian.sequential_bayesian_implementation (theta_values=theta_values,
                                                                          df=df, num_chunks=num_chunks,
                                                                          window_size=window_size
                                                                          )
        
        theta_history = my_varying_theta.time_varying_theta (theta_values=theta_values, df=df)
        
        bldg_id = input_file.split('/')[-3]
        all_histories_seq [bldg_id] = seq_history
        all_histories_time_varying_theta [bldg_id] = theta_history
    
    # my_baysian_vis.plot_evolution_comparison (all_histories=all_histories_time_varying_theta)


    
    # fitted_params = my_pred.heterogeneity_modeling(all_histories_seq)