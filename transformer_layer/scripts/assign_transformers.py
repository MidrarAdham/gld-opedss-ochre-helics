import os
import numpy as np
import pandas as pd
from pathlib import Path
from config import load_config

transformer_sizes = {'75.0': [],
                     '50.0': [],
                     '25.0': []
                     }

def read_datasets (prefix : str, suffix : str) -> pd.DataFrame:
    dfs = []
    for folder in os.listdir (prefix):
        filename = prefix +'/'+ folder +'/' + suffix
        df = pd.read_csv (filename)
        time_col = df['Time']
        df = df.drop (columns=['Time'], axis=1)
        dfs.append(df)
    
    df_concat = pd.concat (dfs, axis=1)
    df_concat['Time'] = time_col

    return df_concat

if __name__ == "__main__":
    cfg = load_config ()
    prefix = cfg['data']['dataset_dir']
    suffix = '/up00/ochre.csv'
    df_concat = read_datasets (prefix=prefix, suffix=suffix)
    
