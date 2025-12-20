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
    max_buildings = cfg["method1"]["max_buildings"]
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

    n_buildings = cfg["method2"]["max_buildings"]

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
    n_trials = cfg["method3"]["n_trials"]
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

def _choose_transformer_mix (required_kva:float, sizes=(25.0, 50.0, 75.0)):
    """
    Find an integer count for each transformer size (n25, n50, n75) that meet or exceed
    the required kVA. 
    
    :param required_kva: the max. diversified demand
    :type required_kva: float
    :param sizes: chosen transformer sizes
    """

    s25, s50, s75 = sizes # rated kva for each transformer
    best = None

    # The maximum allowable number of customers for each transformer (the worst case)
    max_n25 = int(np.ceil (required_kva/s25)) + 2
    max_n50 = int(np.ceil (required_kva/s50)) + 2
    max_n75 = int(np.ceil (required_kva/s75)) + 2
    print(max_n25)
    print(max_n50)
    print(max_n75)

    for n75 in range (max_n75 + 1):
        for n50 in range(max_n50 + 1):
            for n25 in range (max_n25 + 1):
                installed = n25*s25 + n50*s50 + n75*s75

                if installed < required_kva:
                    continue
                overbuild = installed - required_kva
                # print('installed: ', installed, 'required kVA: ', required_kva, 'overbuild? ', overbuild)
                n_total = n25 + n50 + n75
                candidate = (overbuild, n_total, n75, n50, n25, installed)
                if best is None or candidate < best:
                    best = candidate
    
    overbuild, n_total, n75, n50, n25, installed = best

    return {
        "n_25kva": int(n25),
        "n_50kva": int(n50),
        "n_75kva": int(n75),
        "installed_kva": float(installed),
        "overbuild_kva": float(overbuild),
        "n_total_transformers": int(n_total),
    }


def method4_metered_feeder_max_demand (cfg):
    """
    determine how many 25, 50, 75 kVA transfromers are needed based on the simulated data from OCHRE
    and planning utilization factor
    
    :param cfg: read the configuration file, config.toml
    """
    upgrades = cfg["data"]["upgrades"]
    pf = cfg["electrical"]["power_factor"]
    dataset_dir = cfg["data"]["dataset_dir"]
    n_trials = cfg["method4"]["n_trials"]
    n_total_customers = cfg["method4"]["n_total_customers"]
    UF = 0.8
    sizes = (25.0, 50.0, 75.0)

    # 1- simulated the metered feeder peak (max diversified demand)
    analyzer = LoadProfiles(
        n_buildings=n_total_customers,
        dataset_dir=dataset_dir,
        upgrades=upgrades,
        randomized=True
    )
    analyzer.run()

    agg_results = analyzer.aggregate_customers_load_calculations(
        customer_ids=analyzer.load_profiles,
        transformer_kva=75.0,
        power_factor=pf
    )

    metered_demand_kw = float (agg_results['max_diversified_kw']) # Max diversified demand

    # 2- Required installed kVA from the planning UF target (UF). The UF is assumed to be 0.8

    feeder_peak_kva = metered_demand_kw / pf # max diversified demand in kVA
    required_installed_kva = feeder_peak_kva / UF

    # 3- Choose a transformer count:
    plan = _choose_transformer_mix (required_kva=required_installed_kva, sizes=sizes)

    transformer_list = []
    tid = 1 # transformer ID

    for kva, n in [(75.0, plan["n_75kva"]), (50.0, plan["n_50kva"]), (25.0, plan["n_25kva"])]:
        for _ in range(n):
            transformer_list.append({"id": f"T{tid}_{int(kva)}kVA", "kva": kva})
            tid += 1
    
    total_transformer_kva = sum(t["kva"] for t in transformer_list)

    allocation_factor_kw_per_kva = metered_demand_kw / total_transformer_kva

    allocation_results = []

    for t in transformer_list:
        allocated_kw = allocation_factor_kw_per_kva * t['kva']
        allocated_kva = allocated_kw / pf
        utilization = allocated_kva / t['kva'] # this is the allocation factor/ pf

        allocation_results.append ({
            "transformer_id": t["id"],
            "kva_rating": t["kva"],
            "allocated_kw": allocated_kw,
            "allocated_kva": allocated_kva,
            "utilization_factor": utilization,
        })

    summary = {
        "metered_demand_kw": metered_demand_kw,
        "feeder_peak_kva": feeder_peak_kva,
        "pf": pf,
        "uf_target": UF,
        "required_installed_kva": required_installed_kva,
        "installed_kva": plan["installed_kva"],
        "overbuild_kva": plan["overbuild_kva"],
        "total_transformer_kva": total_transformer_kva,
        "allocation_factor_kw_per_kva": allocation_factor_kw_per_kva,
        "n_transformers_25kva": plan["n_25kva"],
        "n_transformers_50kva": plan["n_50kva"],
        "n_transformers_75kva": plan["n_75kva"],
        "n_total_transformers": plan["n_total_transformers"],
        "total_customers_simulated": n_total_customers,
    }

    write_results (cfg=cfg, method="method4_transformer_plan", results=pd.DataFrame([plan | {
        "metered_demand_kw": metered_demand_kw,
        "required_installed_kva": required_installed_kva,
        "uf_target": UF,
        "pf": pf,
    }]))

    write_results(cfg=cfg, method="method4_allocation_results", results=pd.DataFrame(allocation_results))
    write_results(cfg=cfg, method="method4_summary", results=pd.DataFrame([summary]))


# def _build_transformer_list(n_total_customers: int, transformer_capacity: dict[float, int]) -> list:
#     """
#     Greedy sizing as per kersting: as many 75 kvas as possible, then 50s, then 25s.
#     transformer_capacity maps kVA -> customers_supported (from your Method 1 results).
#     """
#     remaining = n_total_customers
#     t_list = []
#     idx = 1

#     for kva in (75.0, 50.0, 25.0):
#         cap = transformer_capacity[kva]
#         while remaining >= cap:
#             t_list.append({"kva": kva, "id": f"T{idx}_{int(kva)}kVA"})
#             idx += 1
#             remaining -= cap

#     # if anything remains (because capacities are coarse), add one 25kVA to cover the remainder
#     if remaining > 0:
#         t_list.append({"kva": 25.0, "id": f"T{idx}_25kVA"})
    
#     print(t_list)

#     return t_list


# def method4_metered_feeder_max_demand(cfg):
#     upgrades = cfg["data"]["upgrades"]
#     pf = cfg["electrical"]["power_factor"]
#     dataset_dir = cfg["data"]["dataset_dir"]
#     n_trials = cfg["method4"]["n_trials"]
#     n_total_customers = cfg["method4"]["n_total_customers"]

#     transformer_capacity = {25.0: 3, 50.0: 9, 75.0: 17}

#     transformer_list = _build_transformer_list(n_total_customers, transformer_capacity)
#     total_transformer_kva = sum(t["kva"] for t in transformer_list)

#     trial_rows = []
#     for trial in range(n_trials):
#         analyzer = LoadProfiles(
#             n_buildings=n_total_customers,
#             dataset_dir=dataset_dir,
#             upgrades=upgrades,
#             randomized=True
#         )
#         analyzer.run()

#         agg_results = analyzer.aggregate_customers_load_calculations(
#             customer_ids=analyzer.load_profiles,
#             transformer_kva=total_transformer_kva,
#             power_factor=pf
#         )
        
#         # max diversified demand = metered demand; just going of Kersting terminology
#         metered_demand_kw = agg_results["max_diversified_kw"]
#         allocation_factor = metered_demand_kw / total_transformer_kva
#         utilization_system = allocation_factor / pf

#         trial_rows.append({
#             "trial": trial,
#             "n_total_customers": n_total_customers,
#             "metered_demand_kw": metered_demand_kw,
#             "total_transformer_kva": total_transformer_kva,
#             "allocation_factor_kw_per_kva": allocation_factor,
#             "utilization_factor": utilization_system,
#             "utilization_factor_check": agg_results["utilization_factor"],  # should match
#             "n_75kva": sum(1 for t in transformer_list if t["kva"] == 75.0),
#             "n_50kva": sum(1 for t in transformer_list if t["kva"] == 50.0),
#             "n_25kva": sum(1 for t in transformer_list if t["kva"] == 25.0),
#         })

#     df_trials = pd.DataFrame(trial_rows)
#     write_results(cfg=cfg, method="method4_trials", results=df_trials)

#     # Summary stats
#     s = df_trials["metered_demand_kw"]
#     u = df_trials["utilization_factor"]
#     df_summary = pd.DataFrame([{
#         "n_trials": n_trials,
#         "n_total_customers": n_total_customers,
#         "total_transformer_kva": total_transformer_kva,
#         "metered_kw_mean": s.mean(),
#         "metered_kw_std": s.std(ddof=1),
#         "metered_kw_p50": s.quantile(0.50),
#         "metered_kw_p90": s.quantile(0.90),
#         "metered_kw_p95": s.quantile(0.95),
#         "metered_kw_p99": s.quantile(0.99),
#         "util_mean": u.mean(),
#         "util_std": u.std(ddof=1),
#         "util_p95": u.quantile(0.95),
#         "util_p99": u.quantile(0.99),
#     }])

#     write_results(cfg=cfg, method="method4_summary", results=df_summary)
#     return df_trials, df_summary




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