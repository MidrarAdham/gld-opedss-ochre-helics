import os
import sys
import json
import helics
import pandas as pd
import datetime as dt
from pathlib import Path

config_folder = Path (__file__).parent.parent

load_paths_file = (config_folder / "config" / "load_paths.json")

master_config = (config_folder / "config" / "master_cosim_config.json")

ochre_helics_config_file = (config_folder / "config" / "ochre_helics_config.json")


def read_load_paths (load_paths_file : str):
    dfs = {}
    with open (load_paths_file, 'r') as f:
        data = json.load (f)
    
    # dfs = [pd.read_csv (value) for key, value in data.items ()]

    for key, value in data.items ():
        df = pd.read_csv (value, parse_dates=["Time"])
        dfs [key] = df.set_index ('Time')

    return dfs

def make_helics_federate(config_file : str ="ochre_helics_config.json"):
    """
    Create a HELICS federate from a JSON configuration file
    This sets up the connection to the HELICS broker
    """
    # Load the federate from the JSON config file
    fed = helics.helicsCreateValueFederateFromConfig(str(config_file))

    # Enter initialization mode and wait for other federates
    fed.enter_initializing_mode()
    return fed

def get_publications (dfs : dict, fed):
    pubs = {}
    for idx in dfs.keys ():
        pub_name = f"ochre_house_{idx}.constant_power_12"
        pubs[idx] = fed.get_publication_by_name(pub_name)
    
    return pubs

def _define_sim_time_settings ():
    start_time = dt.datetime(2025, 1, 1)           # Start date
    time_res = dt.timedelta(minutes=1)            # Time step = 10 minutes
    duration = dt.timedelta(days=30)                # Simulate 1 day
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
        
def run_simulation (fed, dfs, pubs):
    
    sim_time, start_time = _define_sim_time_settings ()

    for t in sim_time:
        # Let's wait for the broker
        _step_to (time=t, fed=fed, start_time=start_time)

        for idx in dfs.keys ():
            # power_kw = dfs[idx]['Total Electric Power (kW)'].get (t, 0)
            power_kw = dfs[idx]['Total Electric Power (kW)'].get (t, 0)
            pubs [idx].publish (complex (power_kw * 1000, 0))
        
        print(f"{t}: published {len(dfs)} loads")

if __name__ == "__main__":
    dfs = read_load_paths (load_paths_file=load_paths_file)
    fed = make_helics_federate (config_file=ochre_helics_config_file)
    pubs = get_publications (dfs=dfs,fed=fed)
    fed.enter_executing_mode ()
    run_simulation (fed=fed, dfs=dfs, pubs=pubs)
    fed.finalize ()

    