import os
import click
import numpy as np
import pandas as pd
from scipy import stats
# Calling my new method, hehe :D
from load_profiles import LoadProfiles

# Configurations: 

DEFAULT_DATASET_DIR = f"{os.getcwd()}/datasets/cosimulation/"

# Engine: 
def check_file (filename: str):

    for files in os.listdir ('./api_methods'):
        if files == filename:
            return pd.read_csv ('./api_methods/'+filename)

def method1_diversity_factor (dataset_dir : str):

    transformer_sizes = [25.0, 50.0, 75.0]

    pf = 0.9
    
    n_trials = 10

    results = []

    max_buildings = 50

    for kva in transformer_sizes:

        for n_buildings in range(1, max_buildings):
        # for i in range(max_iter):
            
            # n_buildings = random.randint(1, 15)

            UF = []
            DF = []

            for trial in range(n_trials):

                analyzer = LoadProfiles (n_buildings = n_buildings,
                            dataset_dir=dataset_dir,
                            upgrades=['up00'],
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


    return results, max_buildings


def method2_load_survey (dataset_dir : str):
    n_buildings = 1e6

    kw_list = []
    kwh_list = []



    analyzer = LoadProfiles (n_buildings = n_buildings,
                            dataset_dir=dataset_dir,
                            upgrades=['up00'],
                            randomized = False
                            )
            
    load_profiles = analyzer.run()

    for cid in load_profiles:
        kw_list.append(analyzer.customer_summaries[cid]['max_demand_kw'])
        kwh_list.append(analyzer.customer_summaries[cid]['total_energy_kwh'])
    
    return kw_list, kwh_list

def linear_regr (kwh : list, kw : list):
    slope, intercept, r_value, p_value, std_err = stats.linregress (x=kwh, y=kw)
    r_squared = r_value ** 2

    results = {
        'intercept_a': intercept,
        'slope_b': slope,
        'r_squared':r_squared,
        'equation': f"kW_peak = {intercept:.4f} + {slope:.6f} x kWh"
    }

    return results

def method3_transformer_load_management (dataset_dir : str):
    pf = 0.9
    
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
                upgrades=['up00'],
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
        

        
        all_results[kva] = {
            'transformer_kwh' : transformer_kwh_list,
            'max_diversified_kw' : max_diverisifed_kw_list,
            'n_customers' : n_customers
        }
    
    return all_results

def method3_regr (results):
    regression_results = {}

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

            'statistics' : {
                'kwh_mean' : kwh_mean,
                'kwh_std': kwh_std,
                'kw_mean': kw_mean,
                'kw_std': kw_std,
                'residual_std': residual_std,  # Prediction uncertainty
                'n_trials': len(kwh_list)
            }
        }

    return regression_results
            
def method4_metered_feeder_max_demand(dataset_dir: str, n_total_customers: int = 300):
    """
    Method 4: Allocate load based on metered feeder demand. Ch. 2.4.1.4
    
    Steps:
    1. Simulate metered demand (aggregate all customers)
    2. Determine transformer configuration needed
    3. Apply allocation factor to distribute load
    """
    pf = 0.9
    
    # Transformer capacity from Method 1
    transformer_capacity = {
        25.0: 6,   # 25 kVA can handle 6 customers
        50.0: 11,  # 50 kVA can handle 11 customers
        75.0: 18   # 75 kVA can handle 18 customers
    }
    
    # simulate substation meter:

    print(f"Step 1: Simulating metered demand for {n_total_customers} customers...")
    
    analyzer = LoadProfiles(
        n_buildings=n_total_customers,
        dataset_dir=dataset_dir,
        upgrades=['up00'],
        randomized=True
    )
    analyzer.run()
    
    agg_results = analyzer.aggregate_customers_load_calculations(
        customer_ids=analyzer.load_profiles,
        transformer_kva=1000,  # Dummy value, we just need max diversified demand
        power_factor=pf
    )
    
    metered_demand_kw = agg_results['max_diversified_kw']
    print(f"  Metered Demand: {metered_demand_kw:.2f} kW")
    
    
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
    
    return {
        'metered_demand_kw': metered_demand_kw,
        'total_customers': n_total_customers,
        'allocation_factor': allocation_factor,
        'total_transformer_kva': total_transformer_kva,
        'transformer_allocations': allocation_results,
        'n_transformers': len(transformer_list)
    }


# Second step is to add decorators. Above any function, add a an argument (if needed) and a command
# First step: create this group wherein other commands will be called and added here.
@click.group()
def cli ():
    """
    CLI for load allocation methods
    """
    pass

@cli.command("method1")
@click.option (
    "--dataset-dir",
    default = DEFAULT_DATASET_DIR,
    show_default = True,
    help = "Path to dataset directory"
)

def method1_command (dataset_dir):
    """
    Run Method 1 - Diversity Factor
    """
    results, ax_buildings = method1_diversity_factor(dataset_dir=dataset_dir)


@cli.command("method2")
@click.option(
    "--dataset-dir",
    default=DEFAULT_DATASET_DIR,
    show_default=True,
    help="Path to dataset directory"
)
def method2_command(dataset_dir):
    """
    Run Method 2 - Load Survey.
    """
    kw_list, kwh_list = method2_load_survey(dataset_dir=dataset_dir)
    method2_regr_results = linear_regr(kwh_list, kw_list)


@cli.command("method3")
@click.option (
    "--dataset-dir",
    default = DEFAULT_DATASET_DIR,
    show_default = True,
    help = "Path to dataset directory"
)

def method3_command (dataset_dir):
    """
    Run method 3 - transformer load management (TLM)
    
    dataset_dir: ResStock dataset directory
    """

    results = method3_transformer_load_management (dataset_dir=dataset_dir)
    method3_regr_results = method3_regr (results=results)


@cli.command("method4")
@click.option(
    "--dataset-dir",
    default=DEFAULT_DATASET_DIR,
    show_default=True,
    help="Path to dataset directory"
)
def method4_command(dataset_dir):
    """
    Run Method 4 - metered_feeder max demand
    """
    all_results = method4_metered_feeder_max_demand (dataset_dir=dataset_dir)
    



def main ():
    data = check_file (filename = 'method1_50.csv')

    if isinstance(data, pd.DataFrame):
        # ================= Diversified Peak Method ==================
        # data, max_buildings = method1_diversity_factor (dataset_dir=dataset_dir)
        cli()
        # df = pd.DataFrame(data)
        # df.to_csv(f'./api_methods/method1_{max_buildings}.csv', index=False)
        # ================= Diversified Peak Method ==================
        # ------------------------------------------------------------
        # ==================== Load Survey Method ====================
        # kw_list, kwh_list = method2_load_survey (dataset_dir=dataset_dir)
        # results = linear_regr (kwh=kwh_list, kw=kw_list)
        # print(pd.DataFrame(results))
        # ==================== Load Survey Method ====================
        # ------------------------------------------------------------
        # ================ Transformer Load Management ===============
        # results = method3_transformer_load_management (dataset_dir=dataset_dir)
        # regr_results = method3_regr (results=results)

        # for kva, results in regr_results.items():
        #     print(f"\n{kva} kVA Transformer ({results['n_customers']} customers):")
        #     print(f"  {results['equation']}")
        #     print(f"  R² = {results['r_squared']:.4f}")
        # ================ Transformer Load Management ===============
        # ------------------------------------------------------------
        # ================ Metered Feeder Max. Demand =================
        # method4_metered_feeder_max_demand (dataset_dir=dataset_dir)
        pass
    else:
        data = pd.read_csv ('./api_methods/method3_50.csv')

if __name__ == '__main__':
    cli()