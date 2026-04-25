'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import pandas as pd
from viz import visualizer
from data_loader import DataLoader
from ols import OrdinaryLeastSquare
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
    estimator = BayesianEstimator ()
    wh_histories = estimator.fit_many (all_dfs=wh_df)
    hvac_histories = estimator.fit_many (all_dfs=hvac_df)
    
    # ols decomposition:
    ols = OrdinaryLeastSquare (
        feeder_demand=feeder_df,
        wh_histories=wh_histories,
        wh_all_dfs=wh_loader.all_dfs,
        hvac_histories=hvac_histories,
        hvac_all_dfs=hvac_loader.all_dfs
    )

    results = ols.run ()
    print(f"New HVAC mean: {results['hvac_predicted'].mean():.1f} W")


    wh_ground_truth = wh_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data ()

    time_col = pd.to_datetime (wh_ground_truth['Time']).dt.strftime('%H:%M')

    viz = visualizer (time_col=time_col)
    viz.plot_wh_predicted_vs_ground_truth (
        wh_predicted=results['wh_predicted'],
        wh_ground_truth=wh_ground_truth['power_out']
        )
    
    viz.plot_hvac_predicted_vs_ground_truth (
        hvac_predicted=results['hvac_predicted'],
        hvac_ground_truth=hvac_ground_truth['power_out']
        )