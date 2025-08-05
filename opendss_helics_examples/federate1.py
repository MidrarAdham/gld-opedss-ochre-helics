import os
import pandas as pd
import helics as h
import json
import time

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
    Set the path of the directories we're using for this setup. Note that we make new folders, such as the results. The profiles/one_week_wh_data/ must be configure manually
    '''
    main_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_dir, "results/")
    profiles_dir = os.path.join(main_dir, "profiles/one_week_wh_profile/")
    
    return results_dir, profiles_dir

def gather_csv_files(profiles_dir):
    '''
    This function gathers all the csv files that defines the power profiles for each DER. 
    '''
    profile_files = [f for f in os.listdir(profiles_dir) if f.endswith('.csv')]
    return profile_files

def publish_values(pub, fed):
    '''
    Publish the values obtained from the DER profiles. These values will be then subscribed to by the other federate. The other federate in this case is OpenDSS simulation, namely DSSfederate.py
    '''
    print("Publishing simple JSON dict...")
    for t in range(0, 5):
        storage_data = {"bat_1": 10.0 + t, "bat_2": 20.0 + t}
        json_str = json.dumps(storage_data)
        h.helicsPublicationPublishString(pub, json_str)
        granted_time = h.helicsFederateRequestTime(fed, (t + 1) * 60)
        print(f"Published at t={granted_time}: {json_str}")
        time.sleep(0.1)

def disconnect_federate(fed):
    '''
    At the end of the simulation, we close the helids processes, including the broker.
    '''
    h.helicsFederateDisconnect(fed)
    print("federate1 done.")

if __name__ == "__main__":
    '''
    If you are seeing this code for the first time, start from here. Here is where everything is coordinated, excuted, and simulation concluded.
    '''
    fed, pub = initialize_federate()
    execute_federate(fed)
    # Now let's read our values from the CSV files
    main_dir, results_dir, results_files = set_paths()
    profile_files = gather_csv_files()
    publish_values(pub, fed)
    disconnect_federate(fed)
    print("Federate1 execution completed.")