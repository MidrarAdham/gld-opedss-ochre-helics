"""
Author: Midrar Adham
Created: Fri May 01 2026
"""

import datetime as dt
import json
import sys
from pathlib import Path
import pandas as pd
import helics

# pandas (pyarrow's parquet reader) must be imported before helics - importing
# helics first segfaults on the first pd.read_parquet call (native library
# conflict between helics' and pyarrow's bundled shared libs).

config_folder = Path(__file__).parent.parent

# Optional CLI arg selects which model's config subfolder to use, e.g.
# `ochre_cosim.py 9500` reads config/9500/*.json instead of config/4node/*.json.
config_variant = sys.argv[1] if len(sys.argv) > 1 else "4node"

load_paths_file = config_folder / "config" / config_variant / "load_paths.json"

master_config = config_folder / "config" / config_variant / "master_cosim_config.json"

ochre_helics_config_file = (
    config_folder / "config" / config_variant / "ochre_helics_config.json"
)


# Only these get published in run_simulation (see the commented-out
# alternatives there) - reading just these columns instead of all 13 keeps
# memory manageable across hundreds of buildings (~13x smaller per file).
# "Water Heating Electric Power (kW)" is deliberately excluded: buildings
# with gas water heaters don't have that column at all (only "Water Heating
# Gas Power (therms/hour)"), so requesting it unconditionally raises
# pyarrow.lib.ArrowInvalid for those buildings.
power_columns = [
    # "Total Electric Power (kW)",
    "Water Heating Electric Power (kW)",
    # "HVAC Cooling Electric Power (kW)",
]


def read_load_paths(load_paths_file: str):
    dfs = {}
    with open(load_paths_file, "r") as f:
        data = json.load(f)

    # dfs = [pd.read_csv (value) for key, value in data.items ()]
    for key, value in data.items():
        try:
            idx = value.split("/")[-4]
            dfs[idx] = pd.read_parquet(value, columns=power_columns)
        except Exception as e:
            continue
    return dfs


def make_helics_federate(config_file: str = "ochre_helics_config.json"):
    """
    Create a HELICS federate from a JSON configuration file
    This sets up the connection to the HELICS broker
    """
    # Load the federate from the JSON config file
    fed = helics.helicsCreateValueFederateFromConfig(str(config_file))

    # Enter initialization mode and wait for other federates
    fed.enter_initializing_mode()
    # print("doing helics federates helicsCreateValueFederateFromConfig")
    return fed


def get_publications(dfs: dict, fed):
    pubs = {}
    for idx in dfs.keys():
        pub_name = f"ochre_house_load_{idx}.constant_power_12"
        # print(pub_name)
        pubs[idx] = fed.get_publication_by_name(pub_name)

    return pubs


def _define_sim_time_settings():
    start_time = dt.datetime(
        2025, 9, 1
    )  # Start date - matches resstock_2024 dataset coverage
    time_res = dt.timedelta(minutes=1)  # Time step = 10 minutes
    duration = dt.timedelta(days=30)  # Simulate 1 day
    sim_times = pd.date_range(
        start_time,
        start_time + duration,
        freq=time_res,
        inclusive="left",
    )
    return sim_times, start_time


def _step_to(time, fed, start_time, offset=0):
    """
    Request the next time step in the co-simulation
    All federates must sync up at each time step
    """
    t_requested = (time - start_time).total_seconds() + offset
    while True:
        t_new = helics.helicsFederateRequestTime(fed, t_requested)
        if t_new >= t_requested:
            return


def run_simulation(fed, dfs, pubs):

    sim_time, start_time = _define_sim_time_settings()

    for t in sim_time:
        # Let's wait for the broker
        _step_to(time=t, fed=fed, start_time=start_time)
        # try:
        for idx in dfs.keys():
            # power_kw = dfs[idx]["Total Electric Power (kW)"].get(t, 0)
            power_kw = dfs[idx]["Water Heating Electric Power (kW)"].get(t, 0)
            # power_kw = dfs[idx]["HVAC Cooling Electric Power (kW)"].get(t, 0)

            pubs[idx].publish(complex(power_kw * 1000, 0))
        # except Exception as e:
        #     continue
            # print(f"Error occurred while processing time step {t}: {e}")


if __name__ == "__main__":
    dfs = read_load_paths(load_paths_file=load_paths_file)
    fed = make_helics_federate(config_file=ochre_helics_config_file)
    pubs = get_publications(dfs=dfs, fed=fed)
    fed.enter_executing_mode()
    run_simulation(fed=fed, dfs=dfs, pubs=pubs)
    fed.finalize()
