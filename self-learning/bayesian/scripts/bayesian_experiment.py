# %%
# Import libs
import os
import numpy as np
import pandas as pd
from pathlib import Path
# %%
def read_datasets (input_path : str) -> pd.DataFrame:
    return pd.read_csv (input_path)

# %%
if __name__ == '__main__':

    # 
    root = Path (__file__).resolve ().parents[3]
    
    dataset_dir = root / 'cosimulation_tools' / 'dss-ochre-helics' / 'profiles' / 'one_week_wh_data'
    
    input_files = [file for file in dataset_dir.iterdir()]
    
    for file in input_files:
        df = read_datasets (input_path=file)
        print(df)
    