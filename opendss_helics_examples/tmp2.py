import os
import json
import time
import helics as h
import pandas as pd
import datetime as dt
from pprint import pprint as pp
from opendss_wrapper import OpenDSS

def set_paths():
    '''
    Setting the path of the directories we're using for this setup. Note that we make new folders, such as the results. The profiles/one_week_wh_data/ must be configured manually.
    '''
    main_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_dir, "results/")
    profiles_dir = os.path.join(main_dir, "profiles/one_week_wh_data/")
    
    dss_file = os.path.join(main_dir, "network_model", "model_base.dss")
    return results_dir, profiles_dir, dss_file

def initialize_opendss(dss_file):
    '''
    Run the dss file so I can get the storage element names later

    The start_time and time_step have no significance here, they are just placeholders. However, we are going
    to use the start_time and time_step in the OpenDSS instance to run the simulation in a later function.
    '''
    
    dss = OpenDSS([dss_file], time_step=stepsize, start_time= start_time)
    return dss


def get_storage_names(dss):
    '''
    Obtain the storage element names from the OpenDSS file. The OpenDSS wrapper helps a lot here. Such a relief.
    '''
    try:
        storage_element_names = dss.get_all_elements('Storage')
        storage_element_names = storage_element_names.index.to_list()
        storage_element_names = [i.replace('Storage.','') for i in storage_element_names]
        print("\n==========================================\n")
        print(f"Storage element sample: {storage_element_names[:3]}")
        print(f"Storage element length: {len(storage_element_names)}\n\n")
        print("==========================================\n")
    except Exception as e:
        print(f"Error getting storage element names: {e}")
        print("Review the 'get_storage_names' function in 'federate1.py' debugging.")
        print('shutting down...')
        quit()
        
    return storage_element_names


def gather_csv_files(profiles_dir):
    profile_files = [f for f in os.listdir(profiles_dir) if f.endswith('.csv')]
    return profile_files

def map_profiles_names_to_storage_names(storage_names, profiles_files):
    '''
    This function is pre-processed before the simulation starts.
    
    Now we have the storage names and the profile files, we need to map the profiles to the storage names.
    The results will be a dictionary where the keys are the storage names and the values are the profile data.
    For instance (Order is not important, despite the example below):
    {
        "storage_element_name_1": csv_file_name_1.csv,
        "storage_element_name_2": csv_file_name_2.csv,
        ...
    }
    '''

    published_data = {}
    try:
        bus_to_storage_map = {}
        for i, j in enumerate(storage_names):
            bus_to_storage_map[j] = profiles_files[i] if i < len(profiles_files) else None
        
        print("Bus to Storage Map length:", len(bus_to_storage_map))
        print("==========================================\n")
        return bus_to_storage_map
    except Exception as e:
        print(f"Error: {e}")
        print("Check the 'map_profiles_to_storage' function in 'federate1.py' debugging.")
        print('shutting down...')
        quit()
    
    return published_data

def load_csv_files (bus_to_storage_map, profiles_dir):
    '''
    Load the CSV files (power values) and map them to the storage elements.
    The output of this function is a dictionary where th keys are the storage names and the values are the power values.
    For instance:
    {
        "storage_element_name_1": [power_value_1, power_value_2, ...],
        "storage_element_name_2": [power_value_1, power_value_2, ...],
        ...
    }
    Each value in the list of each key will be published to the OpenDSS storage element during the appropriate time step.
    
    '''
    profile_data = {}
    try:
        for storage, filename in bus_to_storage_map.items():
            df = pd.read_csv(profiles_dir + filename)
            profile_data[storage] = df.iloc[:, 1].values.tolist()
        
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        print("Check the 'load_csv_files' function in 'federate1.py' debugging.")
        print('shutting down...')
        quit()
    
    return profile_data

def initialize_federate():
    '''
    Initialize the HELICS federate and publication.
    This is where we create the federate and get the publication for the storage powers.
    The 'storage_powers' publication is defined in the federate1.json file.
    '''
    
    fed = h.helicsCreateCombinationFederateFromConfig("federate1.json")
    pub = h.helicsFederateGetPublication(fed, "storage_powers")
    
    print("Federate initialized and publication obtained.")
    return fed, pub
    
def publishing_values_to_opendss(profile_data, pub):
    '''
    This function publishes the values to the OpenDSS storage elements.
    The values are published in the order of the time steps.
    The time steps are defined by the start_time and stepsize variables.
    There is another script that runs the OpenDSS simulation, which is 'DSSfederate.py'.
    That federate will subscribe to the values published by this function.
    '''
    
    print("Starting publishing values to OpenDSS...")
    num_steps = len(next(iter(profile_data.values())))
    
    for t in range(num_steps):
        power_dict = {storage: profile_data[storage][t] for storage in profile_data}
        json_str = json.dumps(power_dict)
        h.helicsPublicationPublishString(pub, json_str)
        granted_time = h.helicsFederateRequestTime(fed, (t + 1) * 60)
        print(f"Published at t={granted_time}: {json_str}")
        time.sleep(0.1)  # Simulate a delay for each time step
    
    pp(power_dict)

if __name__ == '__main__':
    # Initializing the start time and stepsize for the OpenDSS simulation
    start_time = dt.datetime(2021, 12, 25)
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(minutes=30)  # Start with 3 hours for testing
    # --------------------------------------------------------
    # Setting the paths and initializing OpenDSS
    results_dir, profiles_dir, dss_file = set_paths()
    dss = initialize_opendss(dss_file=dss_file)
    storage_names = get_storage_names(dss)
    profile_files = gather_csv_files(profiles_dir)
    bus_to_storage_map = map_profiles_names_to_storage_names(storage_names, profile_files)
    profile_data = load_csv_files(bus_to_storage_map, profiles_dir)
    # --------------------------------------------------------
    # Entering the HELICS publication phase
    fed, pub = initialize_federate()
    publishing_values_to_opendss(profile_data, pub)
    