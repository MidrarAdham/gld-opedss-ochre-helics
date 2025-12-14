# Standard Python Libraries that we'll use in this script:
import os
import click
import numpy as np
import pandas as pd
from scipy import stats
from pprint import pprint as pp
from pathlib import Path

# Calling my methods (all of these methods are Python files inside this folder):
from config import load_config
from load_profiles import LoadProfiles

"""
- This file read the configuration of the methods, calls the methods from load_profiles.py, and write the results.
- There are no computations here. Please read the README file before running any scripts.
"""

# Writing results to a file for analysis later

def write_results (cfg, method : str, results : pd.DataFrame):
    
    results_dir = cfg["project"]["results_dir"]
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    results_id = cfg["project"]["run_id"]
    
    results.to_csv(f"{results_dir}/{results_id}_{method}.csv", index=False)

def method1_diversity_factor (cfg):
    
    results = []
    max_buildings = 10
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
        kw_list.append(analyzer.customer_summaries[cid]['max_demand_kw'])
        kwh_list.append(analyzer.customer_summaries[cid]['total_energy_kwh'])
    
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
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]
    
    # These transformer configs. are obtained from method 1:
    transformer_config = [
        {'kva' : 25.0, 'n_customers': 6},
        {'kva' : 50.0, 'n_customers': 11},
        {'kva' : 75.0, 'n_customers': 18}
    ]

    all_results = {}

    for config in transformer_config:
        kva = config['kva']
        n_customers = config['n_customers']

        # A monte Carlo simulation - run myltiple trials
        n_trials = 50

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
            transformer_kwh_list.append(agg_results['load_profiles_data']['Total Electric Energy (kWh)'].sum())
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
            'n_trials': len(kwh_list)
            }
        
        regression_list.append (regression_results[kva])
    
    write_results (cfg=cfg, method= "method3_regr", results=pd.DataFrame (regression_list))

    return regression_results
            
def method4_metered_feeder_max_demand(cfg):
    """
    Method 4: Allocate load based on metered feeder demand. Ch. 2.4.1.4
    
    Steps:
    1. Simulate metered demand (aggregate all customers)
    2. Determine transformer configuration needed
    3. Apply allocation factor to distribute load
    """
    dataset_dir = cfg["data"]["dataset_dir"]
    upgrades = cfg["data"]["upgrades"]
    pf = cfg["electrical"]["power_factor"]
    n_total_customers = 300
    # Transformer capacity from Method 1
    
    transformer_capacity = {
        25.0: 6,   # 25 kVA can handle 6 customers
        50.0: 11,  # 50 kVA can handle 11 customers
        75.0: 18   # 75 kVA can handle 18 customers
    }
    
    # simulate substation meter:
    
    analyzer = LoadProfiles(
        n_buildings = n_total_customers,
        dataset_dir = dataset_dir,
        upgrades = upgrades,
        randomized=True
    )
    analyzer.run()
    
    agg_results = analyzer.aggregate_customers_load_calculations(
        customer_ids=analyzer.load_profiles,
        transformer_kva=1000,  # Dummy value, we just need max diversified demand
        power_factor = pf
    )
    
    metered_demand_kw = agg_results['max_diversified_kw']
    
    
    # Determine how many transformers needed
    
    # Interesting, Kersting says use as many 75 kVA as possible, then 50 kVA, then 25 kVA
    remaining_customers = n_total_customers
    transformer_list = []
    
    # Add 75 kVA transformers
    while remaining_customers >= transformer_capacity[75.0]:
        transformer_list.append({'kva': 75.0, 'id': f'T{len(transformer_list)+1}_75kVA'})
        remaining_customers -= transformer_capacity[75.0]
    
    # Add 50 kVA transformers
    while remaining_customers >= transformer_capacity[50.0]:
        transformer_list.append({'kva': 50.0, 'id': f'T{len(transformer_list)+1}_50kVA'})
        remaining_customers -= transformer_capacity[50.0]
    
    # Add 25 kVA transformers
    while remaining_customers > 0:
        transformer_list.append({'kva': 25.0, 'id': f'T{len(transformer_list)+1}_25kVA'})
        remaining_customers -= min(remaining_customers, transformer_capacity[25.0])
    
    
    # Calculate total transformer kVA
    total_transformer_kva = sum(t['kva'] for t in transformer_list)
    
    # Calculate allocation factor (Kersting Eq 2.11)
    allocation_factor = metered_demand_kw / total_transformer_kva
    
    # Allocate load to each transformer (Kersting Eq 2.12)
    allocation_results = []
    for transformer in transformer_list:
        allocated_kw = allocation_factor * transformer['kva']
        utilization = allocated_kw / (transformer['kva'] * pf)  # Convert to kVA and get utilization
        
        allocation_results.append({
            'transformer_id': transformer['id'],
            'kva_rating': transformer['kva'],
            'allocated_kw': allocated_kw,
            'allocated_kva': allocated_kw / pf,
            'utilization_factor': utilization
        })
    
    summary_data = {
        'metered_demand_kw': metered_demand_kw,
        'total_customers': n_total_customers,
        'allocation_factor': allocation_factor,
        'total_transformer_kva': total_transformer_kva,
        'n_transformers_25kva': sum(1 for t in transformer_list if t['kva'] == 25.0),
        'n_transformers_50kva': sum(1 for t in transformer_list if t['kva'] == 50.0),
        'n_transformers_75kva': sum(1 for t in transformer_list if t['kva'] == 75.0)
    }
    write_results (cfg=cfg, method= "method4_allocation_results", results = pd.DataFrame(allocation_results))
    write_results (cfg=cfg, method= "method4_summary", results = pd.DataFrame ([summary_data]))


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

if __name__ == '__main__':
    cli()