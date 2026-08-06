"""
Author: MidrarAdham
Created: Sat Aug 01 2026
"""

"""
main.py

Main entry point for the refactored Bayesian + OLS aggregation workflow.

Current scope:
    Stage 1: simultaneous WH/HVAC OLS
    Stage 2: per-device HVAC OLS

This script should stay simple.

Its job:
    1. define configuration
    2. load data
    3. run Bayesian state estimation
    4. run aggregation OLS
    5. print a small summary

The math is handled by:
    bayesian_estimator.py
    aggregation_ols.py
    matrix_builder.py
    ols.py
"""
from pathlib import Path

from aggregation_ols import AggregationOLS
from bayesian_estimator import BayesianEstimator
from data_loader import DataLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

number_of_days = 1
chunks_per_day = 144
minutes_per_day = 1440

bayesian_discount = 0.01

water_heater_state_threshold_w = 5000.0
hvac_state_threshold_w = 100.0

minimum_hvac_mean = 0.01

# excluded_hvac_devices = [
#     "../results/hvac_cosim/ochre_load_16.csv",
# ]

results_dir = Path("../../results/")
manifest_filename = results_dir / "less_than_five_tons_resstock_2024.csv"

day_start = 0
day_end = number_of_days * minutes_per_day
number_of_chunks = number_of_days * chunks_per_day

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    """
    Print a readable section header.
    """
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


def print_summary(summary: dict) -> None:
    """
    Print the most important outputs from the refactored OLS workflow.
    """
    print_section("Device and time-window counts")
    print("number of WH devices:", summary["number_of_wh_devices"])
    print("number of HVAC devices:", summary["number_of_hvac_devices"])
    print("number of active HVAC devices:", summary["number_of_active_hvac_devices"])
    print("number of time windows:", summary["number_of_time_windows"])

    print_section("Simultaneous WH/HVAC OLS")
    print("r_squared:", summary["simultaneous_r_squared"])
    print()
    print(summary["simultaneous_coefficients"])

    print_section("Per-device HVAC OLS")
    print("r_squared:", summary["per_device_hvac_r_squared"])
    print()
    print(summary["per_device_hvac_coefficients"])


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run the refactored Bayesian + OLS workflow.
    """
    print_section("Configuration")
    print("number_of_days:", number_of_days)
    print("number_of_chunks:", number_of_chunks)
    print("day_start:", day_start)
    print("day_end:", day_end)
    print("bayesian_discount:", bayesian_discount)
    print("water_heater_state_threshold_w:", water_heater_state_threshold_w)
    print("hvac_state_threshold_w:", hvac_state_threshold_w)
    print_section("Create data loaders")

    # home_xml_loader = DataLoader(
    #     results_dir=str(home_xml_dir),
    #     day_start=day_start,
    #     day_end=day_end,
    # )

    water_heater_loader = DataLoader(
        results_dir=str(water_heater_results_dir),
        day_start=day_start,
        day_end=day_end,
    )

    hvac_loader = DataLoader(
        results_dir=str(hvac_results_dir),
        day_start=day_start,
        day_end=day_end,
    )

    total_house_loader = DataLoader(
        results_dir=str(total_house_results_dir),
        day_start=day_start,
        day_end=day_end,
    )

    print("Data loaders created.")

    print_section("Load metadata and measured demand")
    # hvac_sizes = home_xml_loader.get_btu_per_device()
    feeder_demand = total_house_loader.load_transformer_data()

    # These ground-truth signals are not used by the refactored OLS runner yet,
    # but keeping them here makes it easy to add evaluation plots later.
    water_heater_ground_truth = water_heater_loader.load_transformer_data()
    hvac_ground_truth = hvac_loader.load_transformer_data()

    # print("number of HVAC metadata records:", len(hvac_sizes))
    print("feeder_demand rows:", len(feeder_demand))
    print("water_heater_ground_truth rows:", len(water_heater_ground_truth))
    print("hvac_ground_truth rows:", len(hvac_ground_truth))

    print_section("Load device-level data and create binary states")
    water_heater_device_data = water_heater_loader.load_csv_files(
        threshold=water_heater_state_threshold_w,
    )

    hvac_device_data = hvac_loader.load_csv_files(
        threshold=hvac_state_threshold_w,
    )

    print("number of WH device files:", len(water_heater_device_data))
    print("number of HVAC device files:", len(hvac_device_data))

    print_section("Run Bayesian estimator")
    bayesian_estimator = BayesianEstimator(
        num_chunks=number_of_chunks,
        discount=bayesian_discount,
    )

    water_heater_histories = bayesian_estimator.fit_many(
        all_dfs=water_heater_device_data,
    )

    hvac_histories = bayesian_estimator.fit_many(
        all_dfs=hvac_device_data,
    )

    print("number of WH Bayesian histories:", len(water_heater_histories))
    print("number of HVAC Bayesian histories:", len(hvac_histories))

    print_section("Run aggregation OLS")
    aggregation_ols = AggregationOLS()

    aggregation_result = aggregation_ols.run(
        wh_histories=water_heater_histories,
        hvac_histories=hvac_histories,
        feeder_demand=feeder_demand,
        power_column="power_out",
        excluded_hvac_devices=excluded_hvac_devices,
    )

    summary = aggregation_ols.summarize_results(
        aggregation_result=aggregation_result,
    )

    print_summary(summary=summary)

    print_section("Done")
    print("Bayesian + OLS workflow completed!.")


if __name__ == "__main__":
    main()
