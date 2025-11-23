#%%
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from pprint import pprint as pp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%%
def calculate_daily_metrics (df: pd.DataFrame, p_max_summary: list, bldg: str, up: str):
    '''
    This function calculates the load factor for every customer. [Add more context. Why?].
    
    Load factor: It is a term used to describle a load. It is the ratio of the average demand
    to the max demand.
    
    Steps (Cite William Kersting):

    1) Identify the peak power and the time where the power is at its peak
    2) Calculate the average power
    3) Get the energy, or calculate it if needed.
    4) Calculate the load factor = avergage power / peak power
    '''
    idx_peak = df['Total Electric Power (kW)'].idxmax()
    p_peak = df.loc[idx_peak, 'Total Electric Power (kW)']
    t_peak = df.loc[idx_peak, 'Time']
    t_sim = (df['Time'].iloc[-1] - df['Time'].iloc[0]).total_seconds() / 3600
    p_avg = df['Total Electric Energy (kWh)'].sum() / t_sim
    load_factor = p_avg / p_peak

    p_max_summary.append({
        'upgrade': up,
        'bldg_id': bldg,
        'daily_avg [kW]': p_avg,
        'peak_time': t_peak,
        'daily_peak_power [kW]': p_peak,
        'daily_load_factor': load_factor,
        'daily_energy_energy [kWh]': df['Total Electric Energy (kWh)'].sum()
    })
    return p_max_summary

# %%

def calculate_diversified_demand(df: pd.DataFrame, group_p_sum: None,group_time_index: None):
    '''
    Diversified demand is the sum of all loads at each time step. For isntance:
    Time    |   Sum of 30 houses [kW]
    15:00   |   20
    15:15   |   60
    15:30   |   35

    and so forth.
    '''
    p = df['Total Electric Power (kW)']
    
    if group_p_sum is None:
        group_p_sum = p.copy()
        group_time_index = df['Time']
    else:
        group_p_sum += p
    
    return group_p_sum, group_time_index
# %%
def calculate_load_duration_curve (p_diversified, t_diversified):
    '''
    Defintion:
    The LDC is the fraction of time wherein the load is above some X kW value.

    Steps:
    1) Calculate the diversified demand (already calculated)
    2) Sort it in a descending order.
    3) Then we can see the times wherein the demand is at max and descending for all houses.

    INPUTS:
    p_diversified and t_diversified are the diversified demand as per Kerstings. 
        - See calculate_diversified_demand() for more info
    
    The LDC typically shows the peak demand vs the percentage of time that that peak happened.
    '''

    p_div_sorted = np.sort(p_diversified)[::-1]
    if len(t_diversified) > 1:
        dt_hours = (t_diversified.iloc[1] - t_diversified.iloc[0]).total_seconds()/3600
    else:
        dt_hours = 0.0
    
    n = len(p_div_sorted)
    total_time_hours = n*dt_hours

    # let's compute the fraction of time:

    duration_fraction = np.arange(1, n+1)/n

    duration_hours = duration_fraction * total_time_hours

    return pd.DataFrame({
        "P_sorted_kW": p_div_sorted,
        'duration_fraction': duration_fraction,
        'duration_hours': duration_hours
    })

# %%
def calculate_diversity_factor (df, max_div_demand):
    '''
    Each customer has a different peak than the others. We want to know where would the peak of all
    customers fall within the load duration curve.

    STEPS:

    1) Calculate the non-coincidental demand:
        The sum of all the peak powers from houses, known as the Non-coincident Demand
    2) Cacluate the diversity factor: 
        This is simply the ratio of the maximum non-coincident demand to the maximum diversified demand
    '''
    # calculate the non-coincidental demand:
    noncoincidental_demand = df['daily_peak_power [kW]'].sum()

    diversity_factor = noncoincidental_demand / max_div_demand
    coincidence_factor = max_div_demand / noncoincidental_demand

    return diversity_factor, coincidence_factor

# %%

def accumulated_diversity_factor (load_profiles: list, summary_df: pd.DataFrame):
    '''
    Similar to calculate_diversity_factor(). However, it calculates the accumulated DF for plotting purposes.
    '''
    accum_diversity_factor = []
    for n in range(1, len(load_profiles)+1):

        # calculate the non-coincidental demand:
        p_sum_n = np.sum(load_profiles[:n], axis=0)

        # Get the peak of the non-coincidental demand:
        p_diversified_n = p_sum_n.max()

        # Maximum non-coincidental demand is the sum of the daily peak kW:
        sum_peaks_n = summary_df['daily_peak_power [kW]'].iloc[:n].sum()

        # 4) calculate diversity factor:
        diversity_factor_n = sum_peaks_n /p_diversified_n

        accum_diversity_factor.append ({
            'n_customers': n,
            'diversity_factor': diversity_factor_n
        })

    return pd.DataFrame(accum_diversity_factor)

# %%
def calculate_utilization_factor (max_diversified_demand, pf = 0.9, transformer_rating = 15):
    '''
    The utilization factor is an indication of how well the capacity of a given electrical device is being utilized. Consider a transformer rated for 15 kVA and assuming
    a power factor of 0.9, the maximum kVA demand can be calculated using the maximum diversified demand/pf. As such, the utilization factor becomes:

    Utilization factor = Maximum kVA demand / Transformer rating, where:

        Maximum kVA demand = Maximum diversified demand / Power factor
    
    Returns the utilization factor
    '''
    max_kva_demand = max_diversified_demand / pf
    utilization_factor = max_kva_demand / transformer_rating

    return utilization_factor
    
# %%
class LoadProfileAnalyzer:

    def __init__(self,
                 dataset_dir: str,
                 n_buildings: int = 12,
                 upgrades: list[str] = ['up00']
                 ):
        """
        Parameters:

        1) dataset_dir : str
            Path to where the ResStock dataset lives
        
        2) n_buildings: int
            The number of buildings to include after filtering

        3) upgrades: list[str] | None
            Which upgrades to process. Default is ["up00"]
        
        4) building_ids_file:
            The name of the file wherein the correct building IDs are saved.
            
        """
        self.upgrades = upgrades
        self.dataset_dir = dataset_dir
        self.n_buildings = n_buildings
        self.bldg_ids_file = './cached_building_ids.csv'

# %%
    def _filter_datasets(self):
        '''
        Parameters:

            Takes noting and returns nothing
        
        Function:

            Write a file called cached_building_ids.csv containing the building IDs
            for files that incorporate the appropriate in.schedules.csv files.

        Internal method: Learning info:
        
        The leading underscore says that this method cannot be called externally. 
        If we were to call any of the methods within the class from an outside script, 
        we can do that. However, the underscore says you should be used internally within the class.
        '''
        root = Path(self.dataset_dir)
        with open(self.bldg_ids_file, 'w') as f:
            for upgrade in self.upgrades:
                for bldg_id in root.iterdir():
                    up00 = bldg_id / upgrade
                    target_file = up00 / 'in.schedules.csv'
                    if target_file.is_file() and target_file.stat().st_size > 0:
                        f.write(f"{bldg_id.name}\n")

    def _load_profiles (self, input_path: str):
        """
        Internal method.

        Load a single building's csv file into a dataframe.

        Returns:
        
        1) df : pd.DataFrame
            The loaded dataframe with parsed time column
        
        2) bldg : str
            building ID extracted from the input_path

        3) up : str
            upgrade (i.e. up00) extracted from path
        """
        keep_cols = ['Time', 'Total Electric Power (kW)',
                     'Total Reactive Power (kVAR)',
                    'Total Electric Energy (kWh)',
                    'Total Reactive Energy (kVARh)'
                    ]
        try:
            bldg = os.path.basename(os.path.dirname(input_path))
            upgrade = os.path.basename(input_path)
            csv_filename = f"out_{bldg}_{upgrade}.csv"
            target_file = os.path.join(input_path, csv_filename)
            df = pd.read_csv(target_file, usecols=keep_cols)
            df['Time'] = pd.to_datetime(df['Time'])
        
        except FileNotFoundError:
            print("="*50)
            print(f"CSV file {target_file} not found.")
            print(f"Checking building {bldg}, upgrade {upgrade} to see if {target_file} exists!")
            print("Quitting ... ")
            print("="*50)
            quit()
        
        return df, bldg, upgrade

# %%
    def _compute_customer_metrics (self, load_summary: list[dict]) -> dict:

        """
        Computer metrics per customer.

        INPUTS:
        load_summary: list of dictionaries
            Each dict contains metrics for a single building and upgrade. These are produced by
            calculate_daily_metrics()
        
        RETURNS:
        dictionary with the following:

        - "summary_df": One row per building/upgrade with the following columns:

            bldg_id, upgrade, daily_peak_power [kW], daily_avg [kW], daily_energy [kWh], daily_load_factor
        """

        summary_df = pd.DataFrame(load_summary)
        return {"summary_df":summary_df}
    
    # %%
    def _compute_aggregation_metrics (self,
                                      customer_metrics: dict,
                                      load_profiles: list[np.ndarray],
                                      diversified_p,
                                      diversified_time) -> dict:
        
        """
        Compute aggregation-level metrics for all customers.

        INPUTS:
        customer_metrics: pd.DataFrame
            Results from _compute_customer_metrics()
        
        load_profiles:
            each array is a time series of 'Total Electric Power (kW)' for a single customer.

        diversified_p:
            Aggregated diverisifed demand time series (sum of all customers' power)

        diversified_time:
            Time index of the diversified_p

        RETURNS:
        dict with keys:
            diverisifed_demand : {
                                    "time": diverisifed_time,
                                    "power_kw": diverisifed_p,
                                    }
            ldf: pd.DataFrame, diversity_factor: float, coincidence_factor: float,
            diversity_vs_n: pd.DataFrame, p_max_diversified: float, t_max_diverisifed: Timestamp
        """
        summary_df = customer_metrics['summary_df']

        # find the max diversified demand and when it occurs:
        idx_group_peak = np.argmax(diversified_p)
        p_max_diversified = diversified_p[idx_group_peak]
        t_max_diversified = diversified_time.iloc[idx_group_peak]

        # Load Duration Curve (ldc)
        ldc = calculate_load_duration_curve(
            p_diversified=diversified_p,
            t_diversified=diversified_time,
        )
        
        # Diversity / coincidence factors (scalar)
        diversity_factor, coincidence_factor = calculate_diversity_factor(
            df=summary_df,
            max_div_demand=p_max_diversified,
        )

        # Diversity factor vs number of customers
        diversity_factor_n_df = accumulated_diversity_factor(
            load_profiles=load_profiles,
            summary_df=summary_df,
        )

        return {
            "diversified_demand": {
                "time": diversified_time,
                "power_kw": diversified_p,
            },
            "ldc": ldc,
            "diversity_factor": diversity_factor,
            "coincidence_factor": coincidence_factor,
            "diversity_vs_n": diversity_factor_n_df,
            "p_max_diversified": p_max_diversified,
            "t_max_diversified": t_max_diversified,
        }
    
    # %%

    def _compute_transformer_metrics (self,
                                      aggregation_metrics: dict,
                                      pf: float = 1,
                                      transformer_size_kva: float = 15.0,
                                      ) -> dict:
        """
        Compute transformer-level metrics based on aggregated demand.

        Inputs
        ------
        aggregation_metrics : dict
            Result from _compute_aggregation_metrics(), must contain "p_max_diversified".
        pf : float, optional
            Assumed power factor for converting kW to kVA (default 0.9).
        transformer_rating_kva : float, optional
            Transformer kVA rating (default 15 kVA).

        Returns
        -------
        dict with keys:
            - "max_diversified_kw": float
            - "max_kva_demand": float
            - "transformer_rating_kva": float
            - "pf": float
            - "utilization_factor": float
        """
        p_max_diversified = aggregation_metrics['p_max_diversified']

        utilization_factor = calculate_utilization_factor(
            max_diversified_demand=p_max_diversified,
            pf=pf,
            transformer_rating=transformer_size_kva,
        )

        max_kva_demand = p_max_diversified / pf

        return {
            "max_diversified_kw": p_max_diversified,
            "max_kva_demand": max_kva_demand,
            "transformer_rating_kva": transformer_size_kva,
            "pf": pf,
            "utilization_factor": utilization_factor,
        }
# %%
    def check_cache_exists(self):
        '''
        INPUTS:
            cached_building_ids.csv

        Looks for cached_building_ids.csv. If it exists, then it parses it. If not, it excutes
        _filter_datasets method to write the file. See _filter_datasets() to learn more about
        cached_building_ids.csv
        '''
        file_path = Path(self.bldg_ids_file)
        if file_path.is_file():
            print('\n\nfile cached_building_ids.csv is in directory!\n\n')
        else:
            self._filter_datasets()

    def read_input_paths_files(self):
        """
        Read the cached_building_ids.csv file.
        Returns the input paths in a list limited by n_buildings

        Parameters:
            None
        """
        with open (self.bldg_ids_file, 'r') as input_paths:
            lines = [line.strip() for line in input_paths if line.strip()]
        
        available = len(lines)
        if self.n_buildings > available:
            print("="*50)
            print(
                f"[LoadProfileAnalyzer] Requested n_buildings={self.n_buildings} "
                f"but only {available} are available in {self.bldg_ids_file}. "
                f"Using {available}."
            )
            print("="*50)
            n = available
        else:
            n = self.n_buildings

        return lines[:n]

    def _build_input_paths(self, building_ids):
        """
        convert building IDs into a full path input files

        INPUTS:
            building IDs
        
        Returns:
            full building path
        """
        input_paths = []
        for upgrade in self.upgrades:
            for bldg in building_ids:
                input_paths.append(os.path.join(self.dataset_dir, bldg, upgrade))
        
        return input_paths
    
    def run(self):
        """
        Parameters:

        None

        The main function. Here you can see the flow of methods excution.
        """
        self.check_cache_exists()
        building_ids = self.read_input_paths_files()
        input_paths = self._build_input_paths(building_ids=building_ids)
        
        sns.set_theme(style="whitegrid", context="talk")

        load_summary = []
        load_profiles = []
        diversified_demand_p = None
        diversified_demand_time = None
        
        for input_path in input_paths:
            df, bldg, up = self._load_profiles(input_path)

            load_profiles.append(df['Total Electric Power (kW)'].to_numpy())

            load_summary = calculate_daily_metrics (
                df=df,
                p_max_summary=load_summary,
                bldg=bldg,
                up=up
                )
            
            diversified_demand_p, diversified_demand_time = calculate_diversified_demand(df=df, 
                                                        group_p_sum=diversified_demand_p,
                                                        group_time_index=diversified_demand_time
                                                        )

        if diversified_demand_p is None:
            print("="*50)
            raise ValueError(
                "Unknown diversified_demand_p. Check the input files or the calculations." \
                "Sorry this message sucks. I'm working on a better detailed message"
                )
        

        customer_metrics = self._compute_customer_metrics(load_summary=load_summary)
        
        aggregation_metrics = self._compute_aggregation_metrics(
            load_profiles=load_profiles,
            customer_metrics=customer_metrics,
            diversified_p=diversified_demand_p,
            diversified_time=diversified_demand_time
        )

        transformer_metrics = self._compute_transformer_metrics (
            pf = 1.0,
            transformer_size_kva= 15,
            aggregation_metrics=aggregation_metrics,
        )

        results = {
            "customer_metrics": customer_metrics,
            "aggregation_metrics": aggregation_metrics,
            "transformer_metrics": transformer_metrics,
            }
        
        return results
#%%
if __name__ == '__main__':
    dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/cosimulation'
    analyzer = LoadProfileAnalyzer(dataset_dir=dataset_dir, n_buildings=500)
    results = analyzer.run()
    print(results)
    # from pprint import pprint as pp
    # pp(results)