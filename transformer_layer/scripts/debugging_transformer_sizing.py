import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
from config import load_config
import matplotlib.pyplot as plt
from pprint import pprint as pp
from pathlib import Path as Path
import matplotlib.ticker as ticker

def build_dataset_dir (bldg_ids : list, prefix : str, suffix : str):
    """
    Given a list of building IDs passed, this function creates a full path for each building ID
    
    :param bldg_ids: a list of building IDs
    :type bldg_ids: list
    :param prefix: the prefix of the path where the bldg ids are located
    :type prefix: str
    :param suffix: the suffix of the path wherein the bldg ids are located
    :type suffix: str
    """
    finalized_dataset_dir = []
    for bldg_id in bldg_ids:
        finalized_dataset_dir.append(f"{prefix}/{bldg_id}{suffix}")
    
    return finalized_dataset_dir

def get_list_full_dataset (dataset_dir : str) -> list:
    """
    Get the list of all the files in the given directory
    
    :param dataset_dir: directory where all the csv files exists
    :type dataset_dir: str
    """
    x = ['409590', '355669']
    # x = [filename for filename in os.listdir (dataset_dir)]
    # shuffeled_list = random.sample (x, k=len (x))
    # return shuffeled_list
    

def concatenate_load_profiles (dataset_dir : str):
    """
    To concatenate dataframes, we execute the following steps:
    1) append all load profiles in a single list
    2) concatenate that list into a single dataframe
    
    :param dataset_dir: the directory wherein the dataset is located
    :type dataset_dir: str
    """
    dfs = []

    
    for idx, filename in enumerate(dataset_dir):

        bldg_name = (filename.split('/'))[-3]
        keep_cols = ['Time', 'Total Electric Power (kW)','Total Reactive Power (kVAR)']

        df = pd.read_csv (filename, usecols=keep_cols)

        df = df.head(1440)

        if idx != 0:
            df = df.drop('Time', axis=1)
        
        df = df.rename (columns={
            'Total Electric Power (kW)' : f'{bldg_name}: Total Electric Power (kW)',
            'Total Reactive Power (kVAR)' : f'{bldg_name}: Total Reactive Power (kVAR)'
            })
        
        df[f'{bldg_name}: Apparent Power (kVA)'] = np.sqrt (
            (df[f'{bldg_name}: Total Electric Power (kW)']**2) + (df[f'{bldg_name}: Total Reactive Power (kVAR)'])**2
            )
        

        dfs.append(df)

    return pd.concat (dfs, axis=1)

def plotting (df : pd.DataFrame) -> None:

    df['Time'] = pd.to_datetime (df['Time']).dt.strftime ('%H: %M')

    sns.set_theme (context= 'notebook')

    fig, axes = plt.subplots (nrows=3, ncols=1, figsize = (16, 10))

    # colors = plt.cm.tab10(np.linspace(0, 1, 3))
    colors = sns.color_palette("deep", 3)

    ax = axes[0]

    ax.plot (df['Time'], df['409590: Apparent Power (kVA)'], label = 'bldg ID: 409590 kVA', linewidth = 2, color=colors[0])
    ax.set_xlabel ('Time (HH: MM)', fontweight = 'bold')
    ax.set_ylabel ('Apparent Power (kVA)', fontweight = 'bold')
    ax.set_title ('Building 409590 Demand (kVA)', fontweight = 'bold')
    ax.axhline (y=df['409590: Apparent Power (kVA)'].max(), color='r', linestyle='--',
                 label=f'Bldg 409590 Peak = {(df['409590: Apparent Power (kVA)'].max()).round(0)} (kVA)', alpha=0.3)
    ax.xaxis.set_major_locator (ticker.MaxNLocator (nbins=30))
    ax.tick_params (axis='x', rotation=45)
    ax.legend ()
    ax.grid (True, alpha=0.3)
    ax.set_xlim(df['Time'].min(), df['Time'].max())

    ax = axes[1]

    ax.plot (df['Time'], df['355669: Apparent Power (kVA)'], label = 'bldg ID: 355669 kVA', linewidth = 2, color=colors[1])
    ax.set_xlabel ('Time (HH: MM)', fontweight = 'bold')
    ax.set_ylabel ('Apparent Power (kVA)', fontweight = 'bold')
    ax.set_title ('Building 355669 Demand (kVA)', fontweight = 'bold')
    ax.axhline (y=df['355669: Apparent Power (kVA)'].max(), color='r', linestyle='--',
                 label=f'Bldg 355669 Peak = {(df['355669: Apparent Power (kVA)'].max()).round(0)} (kVA)', alpha=0.3)
    ax.xaxis.set_major_locator (ticker.MaxNLocator (nbins=20))
    ax.tick_params (axis='x', rotation=45)
    ax.legend ()
    ax.grid (True, alpha=0.3)
    ax.set_xlim(df['Time'].min(), df['Time'].max())

    ax = axes[2]

    ax.plot (df['Time'], df['diversified demand (kVA)'], label = 'Diversified Demand (kVA)', linewidth = 2, color=colors[2])
    ax.set_xlabel ('Time (HH: MM)', fontweight = 'bold')
    ax.set_ylabel ('Diversified Demand (kVA)', fontweight = 'bold')
    ax.set_title ('Bldgs 355669 & 409590 \nDiversified Demand (kVA)', fontweight = 'bold')
    ax.axhline (y=df['diversified demand (kVA)'].max(), color='r', linestyle='--',
                 label=f'Diversified Peak = {(df['diversified demand (kVA)'].max()).round(0)} kVA', alpha=0.3)
    ax.xaxis.set_major_locator (ticker.MaxNLocator (nbins=20))
    ax.tick_params (axis='x', rotation=45)
    ax.set_xlim(df['Time'].min(), df['Time'].max())

    ax.legend ()
    ax.grid (True, alpha=0.3)

    plt.tight_layout ()
    plt.savefig ('./probelamtic_buildings_short.png')
    plt.show()

def abnormal_buildings ():
    filename = "/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/scripts/OR_upgrade0.csv"
    valid_dataset = '/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/load_profiles/cosimulation/'
    valid_bldgs = [int(bldg) for bldg in os.listdir(valid_dataset)]
    metadata = pd.read_csv(filename, low_memory=False)
    filtered_bldg_id = []
    filtered_metadata = metadata[['bldg_id', 'out.electricity.total.energy_consumption..kwh']]
    metadata_bldgs = filtered_metadata['bldg_id'].to_list()
    metadata_bldgs = [int(bldg) for bldg in metadata_bldgs]

    for valid_bldg in valid_bldgs:
        for bldg in metadata_bldgs:
            # print(bldg,'<-->', valid_bldg)
            if bldg == valid_bldg:
                # Now we know folder exists, check if it is empty
                p = f'{valid_dataset}{bldg}/'

                if os.listdir (p):
                    filtered_bldg_id.append(bldg)
    
    filtered_metadata = filtered_metadata[filtered_metadata['bldg_id'].isin(filtered_bldg_id)]
    high_energy_bldgs = filtered_metadata[filtered_metadata['out.electricity.total.energy_consumption..kwh'] > 10000]
    high_energy_bldgs = high_energy_bldgs.sort_values(by='out.electricity.total.energy_consumption..kwh',ascending=False)
    high_energy_bldgs_id = high_energy_bldgs['bldg_id'].to_list()

    

    return high_energy_bldgs_id

    
def closer_look (dataset_dir : list):
    sns.set_theme (context= 'notebook')
    for filename in dataset_dir:
        bldg_name = (filename.split('/'))[-3]
        df = pd.read_csv (filename)
        interval_hours = 5/60

        df['Total Energy (kWh)'] = (((df['Total Electric Power (kW)'] * interval_hours).cumsum ())).round(2)
        df['Total Apparent Power (kVA)'] = np.sqrt (
            (df['Total Electric Power (kW)']**2) +
            (df['Total Reactive Power (kVAR)']**2)
            )
        cols = [col for col in df.columns if '(kW)' in col or '(kWh)' in col or '(kVA)' in col]
        cols.remove('Hot Water Unmet Demand (kW)')
        cols.remove ('HVAC Cooling Electric Power (kW)')
        cols.remove ('Lighting Electric Power (kW)')
        colors = sns.color_palette("deep", len(df.columns))
        df['Time'] = pd.to_datetime (df['Time']).dt.strftime ('%H:%M')
        df = df.head(1440)
        fig, ax = plt.subplots (1,1, figsize=(16,6))
        ax2 = ax.twinx()
        # ax3 = ax.twinx()
        # ax3.spines['right'].set_position(('outward', 60))
        
        for idx, col in enumerate(cols):
            if '(kWh)' in col:
                ax2.plot (df['Time'], df[col],color = colors[idx], label=col, linewidth=2, linestyle='--', alpha=0.7)
            elif '(kW)' in col:
                ax.plot (df['Time'], df[col], color = colors[idx], label=col, linewidth=2)

        ax.xaxis.set_major_locator (ticker.MaxNLocator (nbins=20))
        ax.yaxis.set_major_locator (ticker.MaxNLocator (nbins=12))
        ax.set_xlim (df['Time'].min(), df['Time'].max())
        ax.set_xlabel ('Time (HH:MM)', fontweight = 'bold')
        ax.set_ylabel ('Real Power (kW)', fontweight = 'bold')
        ax.set_title (f'Bldg {bldg_name}: Real Power (kW)', fontweight = 'bold', fontsize=16)
        ax.axhline (y=df['Total Apparent Power (kVA)'].max(),
                    label=f'Peak Demand = {(df['Total Apparent Power (kVA)'].max()).round(0)} (kVA)',
                    alpha=0)
        
        ax.grid (True, alpha=0.3)
        ax.legend (loc='upper left')

        ax2.xaxis.set_major_locator (ticker.MaxNLocator (nbins=20))
        ax2.yaxis.set_major_locator (ticker.MaxNLocator (nbins=12))
        ax2.set_xlim (df['Time'].min(), df['Time'].max())
        ax2.set_xlabel ('Time (HH:MM)', fontweight = 'bold')
        ax2.set_ylabel ('Total Energy (kWh)', fontweight = 'bold')
        ax2.legend (loc='upper right')
        ax2.grid (True, alpha=0.3)
        
        plt.tight_layout ()
        plt.savefig (f'demand_profile_bldg_{bldg_name}.png', dpi=300)

    plt.show()
def vis_all_excessive_profiles (dataset_dir : str):
    """
    Analyze power consumption for high-energy buildings
    Handles missing columns gracefully
    """
    results = []
    
    for idx, filename in enumerate(dataset_dir):
        bldg_id = (filename.split('/'))[-3]
        
        try:
            df = pd.read_csv(filename)
            df = df.head(1440)  # One day
            
            interval_hours = 5 / 60  # 5-minute intervals
            
            # Helper function to safely get column data
            def safe_get(col_name):
                if col_name in df.columns:
                    return df[col_name]
                else:
                    return pd.Series([0] * len(df))  # Return zeros if column missing
            
            # Calculate energy (kWh) and peak power (kW) for each component
            analysis = {
                'bldg_id': bldg_id,
                
                # Total (these should always exist)
                'total_energy_kwh': (safe_get('Total Electric Power (kW)') * interval_hours).sum(),
                'total_peak_kw': safe_get('Total Electric Power (kW)').max(),
                'total_avg_kw': safe_get('Total Electric Power (kW)').mean(),
                
                # HVAC Heating
                'heating_energy_kwh': (safe_get('HVAC Heating Electric Power (kW)') * interval_hours).sum(),
                'heating_peak_kw': safe_get('HVAC Heating Electric Power (kW)').max(),
                'heating_pct': 0,
                
                # HVAC Cooling
                'cooling_energy_kwh': (safe_get('HVAC Cooling Electric Power (kW)') * interval_hours).sum(),
                'cooling_peak_kw': safe_get('HVAC Cooling Electric Power (kW)').max(),
                'cooling_pct': 0,
                
                # Water Heating
                'water_heating_energy_kwh': (safe_get('Water Heating Electric Power (kW)') * interval_hours).sum(),
                'water_heating_peak_kw': safe_get('Water Heating Electric Power (kW)').max(),
                'water_heating_pct': 0,
                
                # Lighting
                'lighting_energy_kwh': (safe_get('Lighting Electric Power (kW)') * interval_hours).sum(),
                'lighting_peak_kw': safe_get('Lighting Electric Power (kW)').max(),
                'lighting_pct': 0,
                
                # Other (includes EV, appliances, etc.)
                'other_energy_kwh': (safe_get('Other Electric Power (kW)') * interval_hours).sum(),
                'other_peak_kw': safe_get('Other Electric Power (kW)').max(),
                'other_pct': 0,
            }
            
            # Calculate percentages
            total = analysis['total_energy_kwh']
            if total > 0:
                analysis['heating_pct'] = (analysis['heating_energy_kwh'] / total) * 100
                analysis['cooling_pct'] = (analysis['cooling_energy_kwh'] / total) * 100
                analysis['water_heating_pct'] = (analysis['water_heating_energy_kwh'] / total) * 100
                analysis['lighting_pct'] = (analysis['lighting_energy_kwh'] / total) * 100
                analysis['other_pct'] = (analysis['other_energy_kwh'] / total) * 100
            
            results.append(analysis)
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} / {len(dataset_dir)} buildings...")
                
        except Exception as e:
            print(f"Error processing {bldg_id}: {e}")
            continue
    
    power_analysis = pd.DataFrame(results)
    
    power_analysis = power_analysis.sort_values('total_peak_kw', ascending=False)
    
    power_analysis.to_csv('high_energy_buildings_power_analysis.csv', index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(power_analysis['total_peak_kw'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(power_analysis['total_peak_kw'].mean(), color='blue', 
                       linestyle='--', linewidth=2,
                       label=f'Mean: {power_analysis["total_peak_kw"].mean():.1f} kW')
    axes[0, 0].set_xlabel('Peak Power (kW)', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Buildings', fontweight='bold')
    axes[0, 0].set_title(f'Peak Power Distribution ({len(power_analysis)} buildings)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    
    top_20 = power_analysis.head(20).copy()
    top_20['bldg_id'] = top_20['bldg_id'].astype(str)
    
    x_pos = np.arange(len(top_20))
    bottom = np.zeros(len(top_20))
    
    components = ['heating_energy_kwh', 'cooling_energy_kwh', 'water_heating_energy_kwh', 'other_energy_kwh']
    colors_bars = ['orange', 'lightblue', 'pink', 'gray']
    labels = ['Heating', 'Cooling', 'Water Heating', 'Other']
    
    for component, color, label in zip(components, colors_bars, labels):
        axes[0, 1].bar(x_pos, top_20[component], bottom=bottom, 
                      color=color, label=label, edgecolor='black', linewidth=0.5)
        bottom += top_20[component].values
    
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(top_20['bldg_id'], rotation=45, ha='right')
    axes[0, 1].set_xlabel('Building ID', fontweight='bold')
    axes[0, 1].set_ylabel('Energy [kWh]', fontweight='bold')
    axes[0, 1].set_title('Energy of highest 20 Buildings Peaks', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)


    axes[1, 0].scatter(power_analysis['heating_pct'], power_analysis['total_peak_kw'], 
                      alpha=0.5, edgecolor='black', s=50)
    axes[1, 0].set_xlabel('Heating % of Total Energy', fontweight='bold')
    axes[1, 0].set_ylabel('Peak Power [kW]', fontweight='bold')
    axes[1, 0].set_title('Peak Power vs Heating Percentage', fontweight='bold')
    axes[1, 0].axhline(60, color='red', linestyle='--', alpha=0.5, linewidth=2, label='60 kW threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    

    axes[1, 1].scatter(power_analysis['other_peak_kw'], power_analysis['total_peak_kw'], 
                      alpha=0.5, edgecolor='black', s=50)
    axes[1, 1].set_xlabel('"Other" Peak (kW) label in OCHRE (EV?)', fontweight='bold')
    axes[1, 1].set_ylabel('Total Peak [kW]', fontweight='bold')
    axes[1, 1].set_title('Total Peak vs Other Load Peak', fontweight='bold')
    axes[1, 1].axhline(60, color='red', linestyle='--', alpha=0.5, linewidth=2, label='60 kW threshold')
    axes[1, 1].axvline(10, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='maybe EV? (>10 kW)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('high_energy_buildings_investigation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return power_analysis



if __name__ == "__main__":

    cfg = load_config ()
    dataset_dir = cfg["data"]["dataset_dir"]
    prefix = dataset_dir
    suffix = '/up00/ochre.csv'

        
    final_dataset_dir = build_dataset_dir (bldg_ids=['409590', '355669'], prefix=prefix, suffix=suffix)

    df = concatenate_load_profiles (dataset_dir=final_dataset_dir)

    df['diversified demand (kVA)'] = df['409590: Apparent Power (kVA)'] + df['355669: Apparent Power (kVA)']

    verified_high_energy_bldgs_ids = abnormal_buildings ()

    dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/resstock_2025/load_profiles/cosimulation'
    prefix = dataset_dir
    suffix = '/up00/ochre.csv'

    final_dataset_dir = build_dataset_dir (bldg_ids=verified_high_energy_bldgs_ids, prefix=prefix, suffix=suffix)
    
    vis_all_excessive_profiles (dataset_dir=final_dataset_dir)
        
        


    # closer_look (dataset_dir=final_dataset_dir)

    