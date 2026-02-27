import os
import sys
import json
import numpy as np
import pandas as pd
from helics.cli import run

# Filenames:
ochre_cosim_file_name = 'ochre_cosim.py'
ochre_fed_name = 'OCHRE_Federate'
gld_fed_name = 'GridLAB-D_Federate'
gld_model_file_name = 'powerflow_4node.glm'
csv_files_dir_name = 'loads_path.json'

# Directories
config_files_dir = './config'
dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/load_profiles/tmp'

# bldg_ids: both of these are used within the same federate
bldg_ids = {'xfmr_50kva' : [385977, 292272, 32387, 188195, 504038, 370750, 258272, 193969]
            # 'xfmr_75kva' : [409590, 366404, 29683, 32387, 186734, 214735, 193969, 177213, 93923]
            }

bldg_ids = [385977, 292272, 32387, 188195, 504038, 370750, 258272, 193969]

def create_loads_json_files (bldg_ids : dict, dataset_dir : str):
    load_paths = {}
    for idx, value in enumerate( bldg_ids):
        if not idx in load_paths:
            load_paths[f"load_{idx}"] = {}
        load_paths[f"load_{idx}"] = f"{dataset_dir}/{value}/up00/ochre.csv"
    
    return load_paths

def create_master_config (ochre_cosim_file_name : str,
                          ochre_fed_name : str, csv_files_dir_name : str):

    ochre_cmd = f"{sys.executable} -u {ochre_cosim_file_name} {ochre_fed_name} {csv_files_dir_name}"
    ochre_federate = {
        "name": "",
        "host": "localhost",
        "directory": ".",
        "exec": ochre_cmd
    }

if __name__ == '__main__':
    create_loads_json_files (bldg_ids=bldg_ids, dataset_dir=dataset_dir)