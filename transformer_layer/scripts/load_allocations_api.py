# Standard Python Libraries that we'll use in this script:
import os
import math
import click
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from pprint import pprint as pp
from sklearn.cluster import AgglomerativeClustering

# Calling my methods (all of these methods are Python files inside this folder):
from config import load_config
from load_profiles import LoadProfiles

"""
- This file read the configuration of the methods, calls the methods from load_profiles.py, and write the results.
- There are no computations here. Please read the README file before running any scripts.
"""

# Writing results to a file for analysis later

def write_results (cfg, method : str, results : pd.DataFrame):

    print("\n\n------- EXECUTING THE write_results method -------\n\n")
    
    results_dir = cfg["project"]["results_dir"]
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    results_id = cfg["project"]["run_id"]
    
    results.to_csv(f"{results_dir}/{results_id}_{method}.csv", index=False)

def method1_diversity_factor (cfg):
    
    print("\n\n------- EXECUTING THE method1_diversity_factor -------\n\n")
    results = []
    max_buildings = 27
    upgrades = cfg["data"]["upgrades"]
    n_trials = cfg["method1"]["n_trials"]
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]
    transformer_sizes = cfg["method1"]["transformer_kva_list"]
    
    


    for kva in transformer_sizes:

        for n_buildings in range(1, max_buildings):
        # for i in range(max_iter):
            
            # n_buildings = random.randint(1, 15)

            UF = []
            DF = []

            for trial in range(n_trials):

                analyzer = LoadProfiles (n_buildings = n_buildings,
                            dataset_dir = dataset_dir,
                            upgrades = upgrades,
                            randomized = True
                            )
            
                analyzer.run()

                agg_results = analyzer.aggregate_customers_load_calculations (
                    customer_ids=analyzer.load_profiles,
                    transformer_kva = kva,
                    power_factor = pf)
                
                UF.append(agg_results['utilization_factor'])
                DF.append(agg_results['diversity_factor'])
                
            avg_UF = np.mean(UF)
            std_UF = np.std(UF)
            avg_DF = np.mean(DF)

            results.append ({
                'kva': kva,
                'n_customers': n_buildings,
                'avg_utilization': avg_UF,
                'std_utilization': std_UF,
                'avg_diversity_factor': avg_DF
            })

    df = pd.DataFrame(results)
    
    write_results (cfg=cfg, method="method1", results= df)

    return df


def method2_load_survey (cfg):
    dataset_dir = cfg["data"]["dataset_dir"]
    upgrades = cfg["data"]["upgrades"]

    n_buildings = 10

    kw_list = []
    kwh_list = []



    analyzer = LoadProfiles (n_buildings = n_buildings,
                            dataset_dir=dataset_dir,
                            upgrades=upgrades,
                            randomized = False
                            )
            
    load_profiles = analyzer.run()

    for cid in load_profiles:
        for key, value in analyzer.customer_summaries[cid].items():
            kw_list.append(analyzer.customer_summaries[cid][key]['max_demand_kw'])
            kwh_list.append(analyzer.customer_summaries[cid][key]['total_energy_kwh'])
    
    return kw_list, kwh_list

def linear_regr (cfg, kwh : list, kw : list):
    slope, intercept, r_value, p_value, std_err = stats.linregress (x=kwh, y=kw)
    r_squared = r_value ** 2

    results = {
        'intercept_a': intercept,
        'slope_b': slope,
        'r_squared':r_squared,
        'equation': f"kW_peak = {intercept:.4f} + {slope:.6f} x kWh"
    }
    
    data_df = pd.DataFrame ({
        'kwh': kwh,
        'kw': kw
    })
    regression_df = pd.DataFrame ([results])

    write_results (cfg=cfg, method="method2_data", results=data_df)
    write_results (cfg=cfg, method="method2_regression", results=regression_df)

    return regression_df

def method3_transformer_load_management (cfg):
    
    upgrades = cfg["data"]["upgrades"]
    n_trials = 20
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]
    
    # These transformer configs. are obtained from method 1:
    transformer_config = [
        {'kva' : 25.0, 'n_customers': 3},
        {'kva' : 50.0, 'n_customers': 9},
        {'kva' : 75.0, 'n_customers': 17}
    ]

    all_results = {}

    for config in transformer_config:
        kva = config['kva']
        n_customers = config['n_customers']

        transformer_kwh_list = []
        max_diverisifed_kw_list = []

        for trial in range(n_trials):

            analyzer = LoadProfiles (
                n_buildings = n_customers,
                dataset_dir=dataset_dir,
                upgrades=upgrades,
                randomized = True
                )
            analyzer.run()
            
            agg_results = analyzer.aggregate_customers_load_calculations (
                customer_ids=analyzer.load_profiles,
                transformer_kva = kva,
                power_factor = pf
                )
            # from pprint import pprint as pp
            # print("\n====\n")
            # # pp(agg_results)
            # print(agg_results['load_profiles_data'].columns)
            # print("="*50)
            # transformer_kwh_list.append(agg_results['load_profiles_data']['Total Electric Energy (kWh)'].sum())
            transformer_kwh_list.append(agg_results['load_profiles_data']['Energy Interval (kWh)'].sum())
            max_diverisifed_kw_list.append(agg_results['max_diversified_kw'])
        

        raw_data = pd.DataFrame ({
            'trial': range(n_trials),
            'transformer_kwh': transformer_kwh_list,
            'max_diversified_kw': max_diverisifed_kw_list,
            'kva_rating': kva,
            'n_customers': n_customers
        })
        
        write_results (cfg=cfg, method = f"method3_kva_{int(kva)}", results = raw_data)

        all_results[kva] = {
            'transformer_kwh' : transformer_kwh_list,
            'max_diversified_kw' : max_diverisifed_kw_list,
            'n_customers' : n_customers
        }
    
    return all_results

def method3_regr (results):

    regression_results = {}
    regression_list = []

    for kva, data in results.items():
        kwh_list = data['transformer_kwh']
        kw_list = data['max_diversified_kw']

        slope, intercept, r_value, p_value, std_err = stats.linregress (kwh_list, kw_list)
        
        kw_predicted = [intercept + slope * kwh for kwh in kwh_list]
        residuals = [actual - pred for actual, pred in zip(kw_list, kw_predicted)]

        kwh_mean = np.mean (kwh_list)
        kwh_std = np.std (kwh_list)
        kw_mean = np.mean (kw_list)
        kw_std = np.mean (kw_list)
        residual_std = np.std (residuals)

        regression_results[kva] = {
            'intercept' : intercept,
            'slope' : slope,
            'r_squared' : r_value ** 2,
            'equation' : f"kw_max_div = {intercept:.4f} + {slope:.6f} x kWh_transformer",
            'n_customers': data['n_customers'],
            'kwh_mean' : kwh_mean,
            'kwh_std': kwh_std,
            'kw_mean': kw_mean,
            'kw_std': kw_std,
            'residual_std': residual_std,  # Prediction uncertainty
            'n_trials': len(kwh_list),
            'kva' : kva
            }
        
        regression_list.append (regression_results[kva])
    
    write_results (cfg=cfg, method= "method3_regression", results=pd.DataFrame (regression_list))

    return regression_results

def _sample_cluster_sizes (n_customers : int, rng: np.random.Generator,
                           min_size : int = 1, max_size : int = 6) -> list[int]:
    
    """
    A cluster size is the cluster of houses per location that sums up to the n_customers.
    
    :param n_customers: Description
    :type n_customers: int
    :param rng: Description
    :type rng: np.random.Generator
    :param min_size: Description
    :type min_size: int
    :param max_size: Description
    :type max_size: int
    :return: Description
    :rtype: list[int]
    """

    if min_size < 1 or max_size < min_size:
        raise ValueError (f"Invalid cluster size bounds: min_size = {min_size}, max_size={max_size}")
    
    sizes = []
    remaining = n_customers

    while remaining > 0:
        s = int (rng.integers(min_size, max_size + 1))
        s = min (s, remaining)
        sizes.append(s)
        remaining -= s
    
    return sizes
def _pick_smallest_transformer (required_kva : float, available_kva = (25.0, 50.0, 75.0)) -> float:
    """
    Pick the smallest transformer rating that can supply the required kVA.
    Returns: 
        - None if none of the transformer capacities fit
        - Otherwise, the transformer size
    :param required_kva: Description
    :type required_kva: float
    :param available_kva: Description
    :return: Description
    :rtype: float
    """

    for s in sorted (available_kva):
        if required_kva <= s:
            return float (s)
    return float("nan")


def method4_metered_feeder_max_demand_for_specific_27_customers (cfg):
    """
    Cluster-based sizing (location-aware proxy)
    Methodology:
        - Sample N customers per trial
        - partition into clusters (houses per transformer location)
        - for each cluster, compute the peak diversified demand and convert it to apparent power using pf & UF
        - choose the smallest transformer that meets the apparent power.

    Outputs:
    Two csv files:
        - method4_cluster_trials.csv (per trial)
        - method4_cluster_assignments/.csv (cluster-level details)
    
    :param cfg: read the configuration file, config.toml
    """
    upgrades = cfg["data"]["upgrades"]
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]


    UF = 0.8
    available_kva = [25.0, 50.0, 75.0]
    n_trials = cfg["method4"]["n_trials"]
    n_total_customers = cfg["method4"]["n_total_customers"]

    cluster_min = 1
    cluster_max = 12

    seed = 123
    rng = np.random.default_rng(seed=seed)

    trial_rows = []
    cluster_rows = []

    analyzer = LoadProfiles(
            n_buildings=n_total_customers,
            dataset_dir=dataset_dir,
            upgrades=upgrades,
            randomized=False
        )
    analyzer.run()

    for trial in range(n_trials):


        # 1- simulated the metered feeder peak (max diversified demand)

        customer_ids = analyzer.load_profiles
        rng.shuffle (customer_ids)

        cluster_sizes = _sample_cluster_sizes (n_customers=len(customer_ids),
                                               rng=rng, min_size=cluster_min,
                                               max_size=cluster_max
                                               )
        
        idx = 0
        n25 = n50 = n75 = 0
        # A cluster needs to be oversized, so it is larger than the max diversified demand
        n_oversize = 0 
        installed_kva_total = 0
        # The sum of clusters peak. This is not the feeder peak. This is useful for debugging
        peak_kw_total = 0

        # Iterate over the cluster index and size:
        for c_idx, c_size in enumerate (cluster_sizes, start=1):
            cluster_ids = customer_ids[idx: idx + c_size]
            idx += c_size
    
            agg_results = analyzer.aggregate_customers_load_calculations(
                customer_ids=cluster_ids,
                transformer_kva=75.0,
                power_factor=pf
            )

            cluster_peak_kw = float (agg_results['max_diversified_kw']) # Max diversified demand

            cluster_req_kva = cluster_peak_kw / pf # max diversified demand in kVA

            chosen = _pick_smallest_transformer (required_kva=cluster_req_kva,
                                                 available_kva=available_kva)

            # Now we need to deal with the fact that a nan might be returned in chosen:

            if math.isnan(chosen):
                n_oversize += 1
            else:
                installed_kva_total += chosen
                if chosen == 25.0:
                    n25 += 1
                elif chosen == 50.0:
                    n50 += 1
                elif chosen == 75.0:
                    n75 += 1
            
            peak_kw_total += cluster_peak_kw

            cluster_rows.append ({
                "trial": trial,
                "cluster_index":c_idx,
                "cluster_size_houses" : c_size,
                "cluster_peak_kw":cluster_peak_kw,
                "cluster_required_kva":cluster_req_kva,
                "chosen_transformer_kva":chosen,
            })
        
        trial_rows.append ({
            "trial":trial,
            "n_total_customers": n_total_customers,
            "uf_target":UF,
            "pf":pf,
            "cluster_min": cluster_min,
            "cluster_max": cluster_max,
            "n_clusters": len(cluster_sizes),
            "n_25kva": n25,
            "n_50kva": n50,
            "n_75kva": n75,
            "n_oversize_clusters": n_oversize,
            "installed_kva_total": installed_kva_total,
            "sum_cluster_peak_kw": peak_kw_total,
        })

    df_trials = pd.DataFrame (trial_rows)
    df_clusters = pd.DataFrame (cluster_rows)

    write_results(cfg=cfg, method="method4_cluster_trials", results=df_trials)
    write_results(cfg=cfg, method="method4_cluster_assignments", results=df_clusters)

def method4_metered_feeder_max_demand (cfg):
    """
    Cluster-based sizing (location-aware proxy)
    Methodology:
        - Sample N customers per trial
        - partition into clusters (houses per transformer location)
        - for each cluster, compute the peak diversified demand and convert it to apparent power using pf & UF
        - choose the smallest transformer that meets the apparent power.

    Outputs:
    Two csv files:
        - method4_cluster_trials.csv (per trial)
        - method4_cluster_assignments/.csv (cluster-level details)
    
    :param cfg: read the configuration file, config.toml
    """
    upgrades = cfg["data"]["upgrades"]
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]


    UF = 0.8
    available_kva = [25.0, 50.0, 75.0]
    n_trials = cfg["method4"]["n_trials"]
    n_total_customers = cfg["method4"]["n_total_customers"]

    cluster_min = 1
    cluster_max = 18

    seed = 123
    rng = np.random.default_rng(seed=seed)

    trial_rows = []
    cluster_rows = []

    for trial in range(n_trials):


        # 1- simulated the metered feeder peak (max diversified demand)
        analyzer = LoadProfiles(
            n_buildings=n_total_customers,
            dataset_dir=dataset_dir,
            upgrades=upgrades,
            randomized=True
        )
        analyzer.run()

        customer_ids = analyzer.load_profiles
        rng.shuffle (customer_ids)

        cluster_sizes = _sample_cluster_sizes (n_customers=len(customer_ids),
                                               rng=rng, min_size=cluster_min,
                                               max_size=cluster_max
                                               )
        
        idx = 0
        n25 = n50 = n75 = 0
        # A cluster needs to be oversized, so it is larger than the max diversified demand
        n_oversize = 0 
        installed_kva_total = 0
        # The sum of clusters peak. This is not the feeder peak. This is useful for debugging
        peak_kw_total = 0

        # Iterate over the cluster index and size:
        for c_idx, c_size in enumerate (cluster_sizes, start=1):
            cluster_ids = customer_ids[idx: idx + c_size]
            idx += c_size
    
            agg_results = analyzer.aggregate_customers_load_calculations(
                customer_ids=cluster_ids,
                transformer_kva=75.0,
                power_factor=pf
            )

            cluster_peak_kw = float (agg_results['max_diversified_kw']) # Max diversified demand

            cluster_req_kva = cluster_peak_kw / pf # max diversified demand in kVA

            chosen = _pick_smallest_transformer (required_kva=cluster_req_kva, available_kva=available_kva)

            # Now we need to deal with the fact that a nan might be returned in chosen:

            if math.isnan(chosen):
                n_oversize += 1
            else:
                installed_kva_total += chosen
                if chosen == 25.0:
                    n25 += 1
                elif chosen == 50.0:
                    n50 += 1
                elif chosen == 75.0:
                    n75 += 1
            
            peak_kw_total += cluster_peak_kw

            cluster_rows.append ({
                "trial": trial,
                "cluster_index":c_idx,
                "cluster_size_houses" : c_size,
                "cluster_peak_kw":cluster_peak_kw,
                "cluster_required_kva":cluster_req_kva,
                "chosen_transformer_kva":chosen,
            })
        
        trial_rows.append ({
            "trial":trial,
            "n_total_customers": n_total_customers,
            "uf_target":UF,
            "pf":pf,
            "cluster_min": cluster_min,
            "cluster_max": cluster_max,
            "n_clusters": len(cluster_sizes),
            "n_25kva": n25,
            "n_50kva": n50,
            "n_75kva": n75,
            "n_oversize_clusters": n_oversize,
            "installed_kva_total": installed_kva_total,
            "sum_cluster_peak_kw": peak_kw_total,
        })

    df_trials = pd.DataFrame (trial_rows)
    df_clusters = pd.DataFrame (cluster_rows)

    write_results(cfg=cfg, method="method4_cluster_trials", results=df_trials)
    write_results(cfg=cfg, method="method4_cluster_assignments", results=df_clusters)


# Second step is to add decorators. Above any function, add a an argument (if needed) and a command
# First step: create this group wherein other commands will be called and added here.
@click.group()
def cli ():
    """
    CLI for load allocation methods
    """
    pass

cfg = load_config ()
# method1_cfg = cfg["method1"]

@cli.command("method1")
def method1_command ():
    """
    Run Method 1 - Diversity Factor
    """
    method1_df = method1_diversity_factor (cfg=cfg)



@cli.command("method2")
def method2_command():
    """
    Run Method 2 - Load Survey.
    """
    kw_list, kwh_list = method2_load_survey(cfg=cfg)
    method2_regr_results = linear_regr(cfg, kwh_list, kw_list)


@cli.command("method3")
def method3_command ():
    """
    Run method 3 - transformer load management (TLM)
    
    dataset_dir: ResStock dataset directory
    """

    results = method3_transformer_load_management (cfg=cfg)
    method3_regr_results = method3_regr (results=results)


@cli.command("method4")
def method4_command ():
    """
    Run Method 4 - metered_feeder max demand
    """
    all_results = method4_metered_feeder_max_demand (cfg=cfg)

@cli.command ("method4_specific")
def method4_specific_command():
    """
    Run Method 4b - this is dedicated for a subset of load profiles
    """
    results = method4_metered_feeder_max_demand_for_specific_27_customers (cfg=cfg)

if __name__ == '__main__':
    cli()