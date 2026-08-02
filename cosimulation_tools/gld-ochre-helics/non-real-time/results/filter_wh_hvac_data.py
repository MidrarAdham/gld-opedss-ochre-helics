'''
Author: MidrarAdham
Created: Sat Aug 01 2026
'''
import os
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from pprint import pprint as pp

def check_valid_bldgs_manifest () -> bool:
    manifest_name = Path('./datasets_manifest.csv')
    if manifest_name.is_file ():
        return True
    else:
        return False
def check_wh_bldgs_manifest () -> bool:
    manifest_name = Path('./wh_cosim/wh_manifest.csv')
    if manifest_name.is_file ():
        return True
    else:
        return False

def check_hvac_bldgs_manifest () -> bool:
    manifest_name = Path('./hvac_cosim/hvac_manifest.csv')
    if manifest_name.is_file ():
        return True
    else:
        return False

def write_manifest (manifest_filename : str) -> pd.DataFrame:
    path_dir = Path("/mnt/datasets/resstock_2024/cosimulation/")
    upgrade = "up02"
    exist = []

    for bldg_id in os.listdir (path_dir):
        ochre_parquet_files = path_dir / bldg_id / upgrade / "simulation_results_august" / "ochre.parquet"
        if ochre_parquet_files.is_file ():
            if ochre_parquet_files.stat().st_size > 0:
                exist.append ({
                    "path":str(ochre_parquet_files)}
                    )
    df = pd.DataFrame (exist)
    df.to_csv (manifest_filename, index=False)

def extract_wh_bldgs (file_cols, row) -> list[dict]:

    if 'Water Heating Electric Power (kW)' in file_cols and not 'HVAC Cooling Electric Power (kW)' in file_cols:
            return row.path

def extract_hvac_bldgs (file_cols, row) -> list[dict]:

    if 'HVAC Cooling Electric Power (kW)' in file_cols:
        return row.path
        

def write_wh_hvac_manifests (full_df : pd.DataFrame):
    wh_bldgs_list = []
    hvac_bldgs_list = []
    wh_manifest_filename = Path ('./wh_cosim/wh_manifest.csv')
    hvac_manifest_filename = Path ('./hvac_cosim/hvac_manifest.csv')

    count_x = 0
    count_y = 0
    for row in full_df.itertuples (index=False):
        try:
            parquet_file = pq.ParquetFile(row.path)
            file_cols = parquet_file.schema.names
            wh_row = extract_wh_bldgs (file_cols=file_cols, row=row)
            hvac_row = extract_hvac_bldgs (file_cols=file_cols, row=row)
            wh_bldgs_list.append ({"path" : wh_row})
            hvac_bldgs_list.append ({"path": hvac_row})

        except Exception as e:
            continue
    
    # if count_y != len (full_df):
    #     print("Not all files have HVAC cols. Need further investigation. Quitting!")
    #     quit()

    

    # check overlap
    # if count_x != count_y:
        # overlap = list(set(wh_bldg_list) & set(hvac_bldg_list))
        # print('\n\nCSV files with both HVAC and water heater')
        # print(overlap)
        # pass

    wh_df = pd.DataFrame (wh_bldgs_list)
    hvac_df = pd.DataFrame (hvac_bldgs_list)
    wh_df.to_csv (wh_manifest_filename, index=False)
    hvac_df.to_csv (hvac_manifest_filename, index=False)


if __name__ == "__main__":
    manifest_filename = "./datasets_manifest.csv"

    if not check_valid_bldgs_manifest():
        write_manifest (manifest_filename=manifest_filename)

    df = pd.read_csv (manifest_filename)

    if not check_wh_bldgs_manifest () or not check_hvac_bldgs_manifest():

        write_wh_hvac_manifests (full_df=df)
