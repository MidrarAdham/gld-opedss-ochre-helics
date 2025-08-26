import os
import json
import time
import helics as h
import pandas as pd
import datetime as dt
from pprint import pprint as pp
from opendss_wrapper import OpenDSS

def initialize_federate():
    '''
    Initializing the federates and get the publications fromt the federate's configuration file
    '''
    fed = h.helicsCreateCombinationFederateFromConfig("federate1.json")
    pub = h.helicsFederateGetPublication(fed, "storage_powers")
    return fed, pub


def execute_federate(fed):
    '''
    Run helics in excution mode ... Not sure how to explain it. Refer to the helics docs.
    '''
    h.helicsFederateEnterExecutingMode(fed)

def set_paths():
    '''
    Setting the path of the directories we're using for this setup. Note that we make new folders, such as the results. The profiles/one_week_wh_data/ must be configured manually.
    '''
    main_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_dir, "results/")
    profiles_dir = os.path.join(main_dir, "profiles/one_week_wh_data/")
    
    dss_file = os.path.join(main_dir, "network_model", "model_base.dss")
    return results_dir, profiles_dir, dss_file

def initialize_opendss(dss_file, stepsize, start_time):
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
    print("\n\n\n\nLoading the csv file right now\n\n\n\n")
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
    
def publishing_values_to_opendss(profile_data, pub, fed):
    '''
    This function publishes the values to the OpenDSS storage elements.
    The values are published in the order of the time steps.
    The time steps are defined by the start_time and stepsize variables.
    There is another script that runs the OpenDSS simulation, which is 'DSSfederate.py'.
    That federate will subscribe to the values published by this function.
    '''
    print("Starting publishing values to OpenDSS...")
    simulation_time_seconds = 300
    time_step_seconds = 60
    num_steps = simulation_time_seconds // time_step_seconds # five steps or every minutes
    
    for step in range(num_steps):

        target_time = (step + 1) * time_step_seconds # 60, 120, 180, ..
        granted_time = h.helicsFederateRequestTime(fed, target_time)
        
        # Get the data for this freaking time step
        power_dict = {storage: profile_data[storage][step] for storage in profile_data}
        json_str = json.dumps(power_dict)
        h.helicsPublicationPublishString(pub, json_str)

        total_power = sum(power_dict.values())
        print(f"Step {step}: Published at t={granted_time}: {len(power_dict)} elements, Total: {total_power:.2f} kW")
        time.sleep(0.1)
    
        print(f"\ngranted time:\t {granted_time}\n")
        if granted_time >= 300:
            print("federate1 execution completed.")
            disconnect_federate(fed)

    print("federate1 excution completed.")
    disconnect_federate(fed)


def disconnect_federate(fed):
    '''
    At the end of the simulation, we close the helids processes, including the broker.
    '''
    h.helicsFederateDisconnect(fed)
    print("federate1 done.")
    quit()



def main():
    # if __name__ == '__main__':
    '''
    If you are seeing this code for the first time, start from here. Here is where everything is coordinated, excuted, and simulation concluded.
    '''
    print("\n\nSetting Paths\n\n\n")
    # Initializing the start time and stepsize for the OpenDSS simulation
    start_time = dt.datetime(2021, 12, 25)
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(minutes=30)  # Start with 3 hours for testing
    # --------------------------------------------------------
    # Setting the paths and initializing OpenDSS
    results_dir, profiles_dir, dss_file = set_paths()
    dss = initialize_opendss(dss_file=dss_file, stepsize=stepsize, start_time=start_time)
    storage_names = get_storage_names(dss)
    profile_files = gather_csv_files(profiles_dir)
    bus_to_storage_map = map_profiles_names_to_storage_names(storage_names, profile_files)
    profile_data = load_csv_files(bus_to_storage_map, profiles_dir)
    # --------------------------------------------------------
    # Entering the HELICS publication phase
    fed, pub = initialize_federate()
    # Entering execution mode
    execute_federate(fed)
    # Publishing values to OpenDSS
    publishing_values_to_opendss(profile_data, pub, fed)
    # Disconnecting the federate
    # disconnect_federate(fed)
    # print("\n\nprocessing profiles-----9\n\n\n")
    # print("federate1 execution completed.")

main()