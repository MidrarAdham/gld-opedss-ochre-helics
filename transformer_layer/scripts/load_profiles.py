import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from config import load_config

class LoadProfiles:
    
    def __init__(self,
                 dataset_dir: str,
                 n_buildings: int,
                 upgrades: list[str] = ['up00'],
                 randomized : bool = False
                 ):
        
        self.upgrades = upgrades
        self.dataset_dir = dataset_dir
        self.n_buildings = n_buildings
        self.cached_bldg_ids_file = './cached_building_ids.csv'
        self.randomized = randomized

        # Data storing structure:

        self.customer_data: dict[str, pd.DataFrame] = {}
        self.customer_summaries: dict[str, dict] = {}
        self.load_profiles: list[str] = []
    
    def _check_file_exists (self) -> bool:
        '''
        look for self.cached_bldg_ids file, which contains all the correct building IDs.
        INPUTS: NONE
        OUTPUTS: Write a new cached_bldg_ids if no file exists in the current directory
        RETURNS: NONE
        '''

        file_path = Path(self.cached_bldg_ids_file)

        if file_path.is_file():
            # print('='*50)
            # print(f'file {self.cached_bldg_ids_file} is in directory.')
            # print('='*50)
            return True
        
        else:
            print('='*50)
            print(f'file {self.cached_bldg_ids_file}.csv is NOT in directory.')
            print(f'Writing a new {self.cached_bldg_ids_file}.csv')
            print('='*50)
            return False

    def _filter_datasets (self):
        '''
        Goes through the downloaded ResStock dataset, ensures every building has in.schedules.csv and it is not empty.

        INPUTS: NONE
        OUTPUTS: writes a csv file containing the correct building IDs
        RETURNS: NONE
        '''
        root = Path (self.dataset_dir)
        with open(self.cached_bldg_ids_file, 'w') as f:
            for upgrade in self.upgrades:
                for bldg_id in root.iterdir():
                    up00 = bldg_id / upgrade
                    target_file = up00 / 'in.schedules.csv'

                    if target_file.is_file() and target_file.stat().st_size > 0:
                        f.write(f"{bldg_id.name}\n")

    def _build_input_files_paths (self, building_ids) -> list[str]:
        '''
        Build paths to the correct building ID folders.
        INPUTS: building IDs folders
        OUTPUTS: NONE
        RETURNS: a list of input paths to the correct bldg IDs
        '''
        input_paths = []

        for upgrade in self.upgrades:
            for bldg in building_ids:
                input_paths.append(os.path.join(self.dataset_dir, bldg, upgrade))

        
        return input_paths

    def _read_the_cached_file (self) -> list:
        '''
        Reads the cached_bldg_ids file, adjust the buildings according to the existing ones
        INPUTS: NONE
        OUTPUTS: NONE
        RETURNS: The available build IDs
        '''
        with open (self.cached_bldg_ids_file, 'r') as input_paths:
            lines = [line.strip() for line in input_paths if line.strip()]
        
        available = len(lines)

        if self.n_buildings > available:
            print("="*50)
            print(
                f"[LoadProfileAnalyzer] Requested n_buildings={self.n_buildings} "
                f"but only {available} are available in {self.cached_bldg_ids_file}. "
                f"Using {available}."
            )
            print("="*50)
            num_buildings = available
        
        else:
            num_buildings = self.n_buildings
        
        if self.randomized:
            # This picks a random set of num_buildings. 
            return random.sample(lines, num_buildings)
        else:
            return lines[:num_buildings]
        
    def _read_simulation_output_files (self, input_path: str):
        '''
        After running OCHRE simulation, this method reads the output file.

        The input files we are reading are the ochre.json files.

        INTPUTS: Path to the ochre simulation output file
        OUTPUTS: NONE
        RETURNS: The building name [str], upgrade name [str], and the data within the simulation output file
        '''
        # keep_cols = ['Time', 'Total Electric Power (kW)',
        #              'Total Reactive Power (kVAR)',
        #             'Total Electric Energy (kWh)',
        #             'Total Reactive Energy (kVARh)'
        #             ]
        keep_cols = ['Time', 'Total Electric Power (kW)',
                     'Total Reactive Power (kVAR)',
                    ]
        
        try:

            # Get the building file name
            bldg = os.path.basename(os.path.dirname(input_path))

            # Get the upgrade name
            upgrade = os.path.basename(input_path)

            # build the output file name
            # csv_filename = f"out_{bldg}_{upgrade}.csv"
            csv_filename = "ochre.csv"

            # Create the path that leads to the output file name
            target_file = os.path.join(input_path, csv_filename)
            # Read the output file name
            df = pd.read_csv(target_file, usecols=keep_cols)
            
            df['Time'] = pd.to_datetime(df['Time'])

            df['day'] = df['Time'].dt.floor('D')

        
        except FileNotFoundError:
            print("="*50)
            print(f"CSV file {target_file} not found.")
            print(f"Check building {bldg}, upgrade {upgrade} to see if {target_file} exists!")
            print("Quitting ... ")
            print("="*50)
            quit()
        
        return df, bldg, upgrade


    def input_files_handler (self) -> list:
        '''
        A container method. It contains the methods needed to:
            - Ensures a build ID file exists in the current directory, and write a new one if none exists
            - Constructing the input files paths
        See the methods for more details
        
        INPUTS: NONE
        RETURNS: [list] Input paths for bldg files

        '''
        if not self._check_file_exists():

            self._filter_datasets()
            building_ids = self._read_the_cached_file()
        else:
            building_ids = self._read_the_cached_file()
        
        input_paths = self._build_input_files_paths (building_ids=building_ids)
        
        return input_paths

    def _calculate_demand (self, data: pd.DataFrame):
        """
        Introducing the 'demand'. As defined by Kersting, the demand is the power consumption over a period of time. In
        this work, the demand is chosen over a five-minute interval.

        This method aggregates the dataset into five-minute intervals.

        INPUTS: Dataframe of a single-day simulation.
        RETURNS: A 5-minute interval dataframe with one row per interval.
        """
        data['Time'] = pd.to_datetime(data['Time'])

        data['interval_start'] = data['Time'].dt.floor('5min')

        demand = data.groupby('interval_start', as_index=False).agg(
            {
                'Total Electric Power (kW)': 'mean',
                'Total Reactive Power (kVAR)': 'mean',
            }
        )

        demand = demand.rename(
            columns={
                'Total Electric Power (kW)': 'Total Electric Power Average Demand (kW)',
                'Total Reactive Power (kVAR)': 'Total Reactive Power Average Demand (kVAR)',
            }
        )

        demand['Time'] = demand['interval_start']
        demand['day'] = demand['Time'].dt.floor('D')

        return demand

    def _calculate_maximum_demand (self, data: pd.DataFrame):
        """
        Docstring for _calculate_maximum_demand
        
        :param data: A single-day OCHRE simulation. Resolution does not matter.
        :type data: pd.DataFrame
        :Returns: Dataframe with a new column representing the maximum demand.
        """


        max_demand = data['Total Electric Power Average Demand (kW)'].max()

        data['Total Electric Power Max Demand (kW)'] = max_demand

        return data
    
    def _calculate_energy_and_average_demand (self, data: pd.DataFrame):
        """
        Method to _calculate_energy_and_average_demand
        
        :param data: A single-day OCHRE simulation. Resolution does not matter.
        :type data: pd.DataFrame

        Return: original data with two new columns:
            - The energy consumed over time.
            - The average demand, that is: Energy / total simulation time
        """

        if data.empty:
            data['Energy Interval (kWh)'] = np.nan
            data['Average Demand (kW)'] = np.nan
            return data

        interval_hours = 5 / 60

        data['Energy Interval (kWh)'] = data['Total Electric Power Average Demand (kW)'] * interval_hours

        total_energy = data['Energy Interval (kWh)'].sum()
        total_hours = data['interval_start'].nunique() * interval_hours

        data['Average Demand (kW)'] = total_energy / total_hours if total_hours > 0 else np.nan

        return data

    def _summarize_individual_customers (self, data: pd.DataFrame):
        
        max_d = data['Total Electric Power Max Demand (kW)'].iloc[0]
        avg_d = data['Average Demand (kW)'].iloc[0]
        total_kwh = data['Energy Interval (kWh)'].sum()
        load_factor = avg_d / max_d if max_d > 0 else np.nan

        return {
            'max_demand_kw': max_d,
            'avg_demand_kw': avg_d,
            'total_energy_kwh': total_kwh,
            'load_factor': load_factor,
        }
    
    # ============================================================
    # ============== Aggregate Customers Methods =================
    # ============================================================

    def _calculate_diversified_demand (self, data: pd.DataFrame):
        '''
        Diversified demand is defined as the sum of all customers demand for each instant of time
        '''
        # data_unique = data.drop_duplicates(subset=['interval_start', 'Total Electric Power Average Demand (kW)'])

        diversified_demand = (
            data.groupby('interval_start')['Total Electric Power Average Demand (kW)']
            .sum()
            .reset_index(name='Diversified Demand (kW)')
            .sort_values('interval_start')
            .reset_index(drop=True)
        )


        return diversified_demand
    
    def _calculate_max_diverisifed_demand (self, data):

        return data['Diversified Demand (kW)'].max()
    
    def _calculate_noncoincident_demand (self, data: pd.DataFrame):
        
        max_noncoincident_demand = 0

        for df in data:
            max_noncoincident_demand += df['Total Electric Power Max Demand (kW)'].max()

        return max_noncoincident_demand
    
    def _calculate_diversity_factor (self, max_noncoincident_demand: float,
                                     max_diversified_demand: float):
        
        return max_noncoincident_demand / max_diversified_demand
    
    def _calculate_utilization_factor (self, max_diversified_demand: float, transformer_rating: int, pf: float = 0.9):
        max_diversified_demand = max_diversified_demand / pf

        return max_diversified_demand / transformer_rating
    
    def _calculate_load_diversity (self, max_noncoincident_demand: float, max_diversified_demand: float):
        
        return max_noncoincident_demand - max_diversified_demand
    
    def _calculate_load_duration_curve (self, diversified_demand: pd.DataFrame):

        diversified_demand_for_ldc = diversified_demand.copy()
        
        diversified_demand_for_ldc = diversified_demand.sort_values(
            'Diversified Demand (kW)', ascending=False).reset_index(drop=True)
        
        n = len(diversified_demand_for_ldc)

        diversified_demand_for_ldc['Percent of Time'] = (np.arange(1, 1 + n) / n) * 100

        return diversified_demand_for_ldc

    def _build_data_and_summary_dictionary (self, 
                                            data: pd.DataFrame,
                                            summary: dict,
                                            bldg: str,
                                            upgrade: str,
                                            day : int):
        
        """
        Docstring for _build_data_and_summary_dictionary
        
        :param data: A single day load profile of each house.
        :type data: pd.DataFrame
        :param summary: Description
        :type summary: dict
        :param bldg: Description
        :type bldg: str
        :param upgrade: Description
        :type upgrade: str
        """

        customer_id = f"bldg_{bldg}_{upgrade}"

        # customer_data: nested by day

        if customer_id not in self.customer_data:
            self.customer_data[customer_id] = {}

        self.customer_data[customer_id][day] = data

        # customer_summaries: nested by day
        if customer_id not in self.customer_summaries:
            self.customer_summaries[customer_id] = {}

        self.customer_summaries[customer_id][day] = summary


        # Now load_profiles only has customer IDs, which can then be referenced later to get the data for each customer.
        # Try this in the api_test.py script. Now this script can be used as a module!
        # You jsut need to remember that load_profiles is a list of customer IDs!
        if customer_id not in self.load_profiles:
            self.load_profiles.append(customer_id)
        

    def individual_customer_load_calculations (self,
                                               data: pd.DataFrame,
                                               bldg: str,
                                               up: str
                                               ):
        
        """
        Method encapsulates all the individual customer calculations
        :param data: A single day power consumption profile
        :type data: pd.DataFrame
        :param bldg: a string representing the simulated building in OCHRE
        :type bldg: str
        :param up: A string representing the OCHRE upgrade being simulated
        :type up: str
        """

        data = data.set_index ('Time')

        groups = data.groupby ('day')

        for day, df_group in groups:
            
            df_group = df_group.reset_index()
        
            demand = self._calculate_demand (data=df_group)

            demand = self._calculate_maximum_demand (data= demand)

            demand = self._calculate_energy_and_average_demand (data=demand)

            summary = self._summarize_individual_customers (data=demand)

            self._build_data_and_summary_dictionary (data=demand,
                                                     summary=summary,
                                                     bldg=bldg,
                                                     upgrade=up,
                                                     day=day
                                                     )

    def aggregate_customers_load_calculations (self,
                                               customer_ids: list[str] | None = None,
                                               transformer_kva: float = 50.0,
                                               power_factor: float = 0.9
                                               ):
        """
        A container method. It contains all methods needed to calculate aggregate metrics using Kersting's Book, Chapter 2.

        INPUTS: 
        - dict {Timestamp (day) : dataframe} -> self.load_profiles
        - list[float] -> transformer capacity in kVA
        - float -> power factor. Default is 0.9

        OUTPUTS:
        """
        if customer_ids is None:
            customer_ids = self.load_profiles

        # get every dataframe from the self.load_profiles dictionary
        list_of_dfs = [
            day_df
            for cid in customer_ids
            for day_df in self.customer_data[cid].values()
            ]

        # get every dataframe from the self.load_profiles dictionary
        # list_of_dfs = [list(self.load_profiles[dfs].values())[0] for dfs in range(len(self.load_profiles))]

        # concatente the dataframes such that they included in a single dataframe.

        concat_df = pd.concat(list_of_dfs, axis=0)


        # calculate the diversified demand
        diversified_demand = self._calculate_diversified_demand(data=concat_df)

        # calculate the max. diversified demand
        max_diversified_demand = self._calculate_max_diverisifed_demand (data=diversified_demand)

        # calculate the max. noncoincident demand
        max_noncoincident_demand = self._calculate_noncoincident_demand (data=list_of_dfs)
        # print(max_diversified_demand, '\t -- ', max_noncoincident_demand)
        # quit()

        # calculate diversity factor
        diversity_factor = self._calculate_diversity_factor (max_noncoincident_demand = max_noncoincident_demand,
                                                                max_diversified_demand = max_diversified_demand)
        
        # calculate utilization factor: (this won't make sense if the n_buildings is high and kva is low!)
        utilization_factor = self._calculate_utilization_factor (max_diversified_demand = max_diversified_demand,
                                                                    transformer_rating=transformer_kva,
                                                                    pf= power_factor)

        load_diversity = self._calculate_load_diversity (max_noncoincident_demand = max_noncoincident_demand,
                                                            max_diversified_demand = max_diversified_demand)
        
        # only returning the data, no plots here
        load_duration_curve_data = self._calculate_load_duration_curve (diversified_demand = diversified_demand)

        return {
            'load_profiles_data' : concat_df,
            'diversified_demand': diversified_demand,
            'load_duration_curve': load_duration_curve_data,
            'max_diversified_kw': max_diversified_demand,
            'max_noncoincident_kw': max_noncoincident_demand,
            'diversity_factor': diversity_factor,
            'load_diversity_kw': load_diversity,
            'utilization_factor': utilization_factor,
            'n_customers': len(customer_ids),
            'transformer_kva_rating': transformer_kva,
            'power_factor_assumed': power_factor,
            }


    def run(self):
        
        self.customer_data = {}
        self.customer_summaries = {}
        self.load_profiles = []

        input_paths = self.input_files_handler()

        for input_path in input_paths:

            df, bldg, upgrade = self._read_simulation_output_files(input_path=input_path)

            self.individual_customer_load_calculations (data= df, bldg= bldg, up=upgrade)
        
        return self.load_profiles
    


if __name__ == "__main__":

    cfg = load_config ()
    
    dataset_dir = cfg["data"]["dataset_dir"]
    
    analyzer = LoadProfiles (dataset_dir = dataset_dir,
                              n_buildings = 50,
                              upgrades = ['up00']
                              )
    
    analyzer.run()
