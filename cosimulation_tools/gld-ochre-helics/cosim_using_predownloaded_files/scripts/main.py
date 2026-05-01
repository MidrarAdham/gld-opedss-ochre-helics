'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import pandas as pd
from milp import MILP
from viz import visualizer
from data_loader import DataLoader
from ols import OrdinaryLeastSquare
from proposal_viz import ProposalFigures
from bayesian_estimator import BayesianEstimator


if __name__ == '__main__':


    wh_dir = '../results/wh_cosim/'
    hvac_dir = '../results/hvac_cosim/'
    total_house_dir = '../results/total_house_consumption/'

    # Load the data:
    wh_loader = DataLoader (results_dir=wh_dir)
    hvac_loader = DataLoader (results_dir=hvac_dir)
    total_house_loader = DataLoader (results_dir=total_house_dir)

    # retrieve the dataframes
    wh_df = wh_loader.load_csv_files (threshold=5000.0)
    hvac_df = hvac_loader.load_csv_files (threshold=100.0)
    feeder_df = total_house_loader.load_transformer_data ()

    # bayesian estimation:
    estimator_low = BayesianEstimator (discount=0.01)
    wh_histories_low = estimator_low.fit_many (all_dfs=wh_df)
    hvac_histories_low = estimator_low.fit_many (all_dfs=hvac_df)
    
    # ols decomposition:
    ols_low = OrdinaryLeastSquare (
        feeder_demand=feeder_df,
        wh_histories=wh_histories_low,
        wh_all_dfs=wh_loader.all_dfs,
        hvac_histories=hvac_histories_low,
        hvac_all_dfs=hvac_loader.all_dfs
    )
    

    results_low = ols_low.run ()

    # bayesian estimation:
    estimator_high = BayesianEstimator (discount=0.3)
    wh_histories_high = estimator_high.fit_many (all_dfs=wh_df)
    hvac_histories_high = estimator_high.fit_many (all_dfs=hvac_df)
    
    # ols decomposition:
    ols_high = OrdinaryLeastSquare (
        feeder_demand=feeder_df,
        wh_histories=wh_histories_high,
        wh_all_dfs=wh_loader.all_dfs,
        hvac_histories=hvac_histories_high,
        hvac_all_dfs=hvac_loader.all_dfs
    )
    

    results_high = ols_high.run ()


    wh_ground_truth = wh_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data ()

    time_col = pd.to_datetime (wh_ground_truth['Time']).dt.strftime('%H:%M')

    pf = ProposalFigures(time_col=time_col, save_dir='./figures/')

    # Section 1
    # pf.fig1_raw_binary_states(
    #     wh_all_dfs=wh_loader.all_dfs,
    #     hvac_all_dfs=hvac_loader.all_dfs
    # )
    # pf.fig2_posterior_mean_comparison(
    #     histories_low  = wh_histories_low,
    #     histories_high = wh_histories_high,
    #     der_label      = 'Water Heater',
    #     filename       = 'fig2_wh_posterior_mean.png'
    # )

    # pf.fig2_posterior_mean_comparison(
    #     histories_low  = hvac_histories_low,
    #     histories_high = hvac_histories_high,
    #     der_label      = 'HVAC',
    #     filename       = 'fig2_hvac_posterior_mean.png'
    # )
    
    # pf.fig4_switching_event_zoom(
    #     wh_all_dfs       = wh_loader.all_dfs,
    #     wh_histories_low  = wh_histories_low,
    #     wh_histories_high = wh_histories_high,
    #     filename         = 'fig4_switching_event_zoom.png'
    # )

    # pf.fig4_switching_event_zoom(
    #     wh_all_dfs       = hvac_loader.all_dfs,
    #     wh_histories_low  = hvac_histories_low,
    #     wh_histories_high = hvac_histories_high,
    #     filename         = 'fig4_switching_event_zoom_hvac.png'
    # )

    # pf.fig5_switching_event_zoom_with_matrix(
    #     wh_all_dfs       = wh_loader.all_dfs,
    #     wh_histories_low  = wh_histories_low,
    #     wh_histories_high = wh_histories_high,
    #     filename         = 'fig5_switching_event_zoom_wh_with_matrix.pdf'
    # )

    pf.fig5_switching_event_zoom_with_matrix(
        wh_all_dfs       = hvac_loader.all_dfs,
        wh_histories_low  = hvac_histories_low,
        wh_histories_high = hvac_histories_high,
        filename         = 'fig5_switching_event_zoom_hvac_with_matrix.pdf'
    )

    # pf.fig6_aggregated_probability_vectors(
    #     wh_histories_low    = wh_histories_low,
    #     wh_histories_high   = wh_histories_high,
    #     hvac_histories_low  = hvac_histories_low,
    #     hvac_histories_high = hvac_histories_high,
    #     filename            = 'fig6_aggregated_probability_vectors.png'
    # )
    # pf.fig7_input_vectors(
    #     wh_histories_low    = wh_histories_low,
    #     wh_histories_high   = wh_histories_high,
    #     hvac_histories_low  = hvac_histories_low,
    #     hvac_histories_high = hvac_histories_high,
    #     filename            = 'fig7_input_vectors.png'
    # )

    # pf.fig8_ols_comparison(
    #     std_wh_predicted    = results_low['wh_predicted'],
    #     std_hvac_predicted  = results_low['hvac_predicted'],
    #     delta_wh_predicted  = results_low['delta_wh_predicted'],
    #     delta_hvac_predicted = results_low['delta_hvac_predicted'],
    #     wh_ground_truth     = wh_ground_truth['power_out'],
    #     hvac_ground_truth   = hvac_ground_truth['power_out'],
    #     filename            = 'fig8_ols_comparison.png'
    # )

    # pf.fig10_delta_ols_predicted_vs_truth(
    #     wh_predicted=results_low['delta_wh_predicted'],
    #     hvac_predicted=results_low['delta_hvac_predicted'],
    #     wh_ground_truth=wh_ground_truth['power_out'],
    #     hvac_ground_truth=hvac_ground_truth['power_out'],
    #     filename='fig10_delta_ols_predicted_vs_truth.pdf'
    # )
    quit()

    # # Section 2
    # pf.fig5_mean_matrix_heatmap(results_low['wh_mean_matrix'], results_low['hvac_mean_matrix'])
    # pf.fig6_aggregated_probability_vectors(
    #     x_wh=results_low['x_wh'],
    #     x_hvac=results_low['x_hvac']
    # )

    # # Section 3
    # pf.fig7_ols_predicted_vs_truth(
    #     wh_predicted=results_low['wh_predicted'],
    #     hvac_predicted=results_low['hvac_predicted'],
    #     wh_ground_truth=wh_ground_truth['power_out'],
    #     hvac_ground_truth=hvac_ground_truth['power_out']
    # )
    # pf.fig8_ols_residuals(
    #     feeder_minus_background=results_low['feeder_minus_background'],
    #     combined_predicted=results_low['combined_predicted']
    # )

    # # # Section 4
    # pf.fig9_delta_signals(
    #     x_wh=results_low['x_wh'],
    #     x_hvac=results_low['x_hvac']
    # )

    # # Section 4 comparison table
    # comparison_results = {
    #     'Day 2': {'wh_truth': 3273, 'hvac_truth': 10497,
    #             'low_wh': 3289, 'low_hvac': 13169,
    #             'high_wh': 3776, 'high_hvac': 21745},
    #     'Day 3': {'wh_truth': 2674, 'hvac_truth': 11826,
    #             'low_wh': 2669, 'low_hvac': 14853,
    #             'high_wh': 4921, 'high_hvac': 9398},
    #     'Day 4': {'wh_truth': 2800, 'hvac_truth': 6872,
    #             'low_wh': 2764, 'low_hvac': 7682,
    #             'high_wh': 4057, 'high_hvac': 10666},
    #     'Day 5': {'wh_truth': 3059, 'hvac_truth': 5984,
    #             'low_wh': 3377, 'low_hvac': 6273,
    #             'high_wh': 4921, 'high_hvac': 9398},
    # }
    # pf.fig11_discount_comparison_table(results=comparison_results)



    # print("=== Standard OLS ===")
    # print(f"wh_predicted mean:   {results['wh_predicted'].mean():.1f} W")
    # print(f"hvac_predicted mean: {results['hvac_predicted'].mean():.1f} W")

    # print("=== Delta OLS ===")
    # print(f"wh_predicted mean:   {results['delta_wh_predicted'].mean():.1f} W")
    # print(f"hvac_predicted mean: {results['delta_hvac_predicted'].mean():.1f} W")

    # print("=== Ground Truth ===")
    # print(f"wh ground truth:     {wh_ground_truth['power_out'].mean():.1f} W")
    # print(f"hvac ground truth:   {hvac_ground_truth['power_out'].mean():.1f} W")

    # quit()

    # time_col = pd.to_datetime (wh_ground_truth['Time']).dt.strftime('%H:%M')

    # viz = visualizer (time_col=time_col)
    # viz.plot_wh_predicted_vs_ground_truth (
    #     wh_predicted=results['wh_predicted'],
    #     wh_ground_truth=wh_ground_truth['power_out']
    #     )
    
    # viz.plot_hvac_predicted_vs_ground_truth (
    #     hvac_predicted=results['hvac_predicted'],
    #     hvac_ground_truth=hvac_ground_truth['power_out']
    #     )
    
    # viz.plot_ols_residuals(
    #     feeder_minus_background=results['feeder_minus_background'],
    #     combined_predicted=results['combined_predicted']
    # )

    # viz.plot_posterior_variance (
    #     wh_histories=wh_histories,
    #     hvac_histories=hvac_histories
    #     )