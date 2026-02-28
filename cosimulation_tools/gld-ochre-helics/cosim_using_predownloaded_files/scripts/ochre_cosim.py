import os
import sys
import json
import helics
import pandas as pd
import numpy as np
from helics.cli import run
from ochre import Analysis
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
        dfs [key] = pd.read_csv (value)

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


if __name__ == "__main__":
    dfs = read_load_paths (load_paths_file=load_paths_file)
    fed = make_helics_federate (config_file=ochre_helics_config_file)
    fed = get_publications (dfs=dfs,fed=fed)
    fed.enter_executing_mode ()
    