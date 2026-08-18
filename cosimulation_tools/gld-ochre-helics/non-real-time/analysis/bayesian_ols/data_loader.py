"""
Author: Midrar Adham
Created: Thu Apr 23 2026
"""

"""
For new PEG users, usage [wh dir example]:
- import the class --> from data_loader import DataLoader
- loader = DataLoader (results=wh_dir)
- loader = load_csv_files (threshold=5000.0)

"""
import os
import json
import pandas as pd
from pathlib import Path



class DataLoader:
    def __init__(
        self,
        results_dir: str,
        manifest_file: str,
        day_start: int = 1440,
        day_end: int = 2880,
        upgrade: str = "up00"
    ):

        self.all_dfs = {}
        self.up = upgrade
        self.day_end = day_end
        self.day_start = day_start
        self.results_dir = results_dir
        self.manifest_file = manifest_file

    def _collect_files_from_directories(self, files_dir: str, manifest_file : str) -> list:
        """
        append each csv file to a list and returns the path/filenames
        """
        from pprint import pprint as pp
        prefix_default_bldg_dir = Path ("/mnt/datasets/resstock-2024/cosimulation")
        suffix_default_bldg_dir = Path ("up02/simulation_results_august/ochre.parquet")
        xfmr_bldg_map = {}
        xfmr_filenames = [fname.split(".csv")[0] for fname in os.listdir(files_dir)]
        with open (manifest_file, "r") as f:
            manifest_file_data = json.load (f)
        for item in manifest_file_data:
            xfmr_name = item["transformer_name"]
            
            if xfmr_name not in xfmr_filenames:
                continue
            
            bldg_ids = item.get("building_ids", [item["building_ids"]])
            
            xfmr_bldg_map[xfmr_name] = [
                str(prefix_default_bldg_dir / bldg_id / suffix_default_bldg_dir)
                for bldg_id in bldg_ids
            ]
        pp(xfmr_bldg_map)
        quit()
        # print(xfmr_bldg_map)
 
    def _clean_dataframe(self, filename: str):
        # names=["time", "power_out.real"]
        df = pd.read_csv(filename, header=0, usecols=['# timestamp', 'power_out.real'], skiprows=8)
        df = df.iloc[self.day_start : self.day_end]
        df = df.rename(columns={"# timestamp": "time", "power_out.real": "power_out"})
        df.loc[:, "time"] = df["time"].apply(lambda x: x.strip("PST"))
        df.loc[:, "time"] = pd.to_datetime(df["time"])
        df["power_out"] = df["power_out"].map(lambda x: f'{x:.2f}')
        df["power_out"] = df["power_out"].map(lambda x: 0.00 if 0.00 >= float(x) > -1.00 else x)

        return df

    def _create_binary_states(self, df: pd.DataFrame, threshold: float):
        df = df.copy()
        df["state"] = (df[df.columns[1]] > threshold).astype("bool")

        return df

    def load_transformer_data(self):
        xfmr = f"{self.results_dir}residential_transformer.csv"
        df = pd.read_csv(xfmr, skiprows=8, usecols=["# timestamp", "power_out"])
        # print("\n\ndon't forget you're using the second day of the data\n\n")
        df = df.iloc[self.day_start : self.day_end]
        df.loc[:, "# timestamp"] = df["# timestamp"].apply(lambda x: x.strip("PST"))
        df.loc[:, "# timestamp"] = pd.to_datetime(df["# timestamp"])
        df.loc[:, "power_out"] = df["power_out"].apply(lambda x: complex(x))
        df.loc[:, "power_out"] = df["power_out"].apply(lambda x: x.real)
        df["# timestamp"] = pd.to_datetime(df["# timestamp"], errors="coerce")
        df = df.set_index("# timestamp")
        df = df.resample("10min").mean()
        df = df.reset_index()
        df["power_out"] = pd.to_numeric(df["power_out"], errors="coerce")
        df = df.rename(columns={"# timestamp": "Time"})

        return df

    def _get_buildings_from_transformers (self, manifest_file_dir : str, xfmr_files_dir : list) -> list[str]:
        pass
    
    def load_csv_files(self, threshold: float):
        """
        Returns the
        """

        cosim_files = self._collect_files_from_directories(
            files_dir=self.results_dir,
            manifest_file = self.manifest_file
            )

        for filename in cosim_files:
            xfmr_name = filename.split("/")[-1]
            print(xfmr_name)
            quit()
            df = self._clean_dataframe(filename=filename)
            df = self._create_binary_states(df=df, threshold=threshold)

            self.all_dfs[filename] = df

        return self.all_dfs