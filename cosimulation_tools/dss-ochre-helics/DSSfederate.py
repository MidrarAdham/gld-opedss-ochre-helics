import os
import json
import time
import helics as h
import datetime as dt
from pprint import pprint as pp
from opendss_wrapper import OpenDSS

def initialize_federate():
    fed = h.helicsCreateCombinationFederateFromConfig("DSSfederate.json")
    sub = h.helicsFederateGetInputByIndex(fed, 0)
    return fed, sub

def execute_federate(fed):
    print("Entering executing mode...")
    h.helicsFederateEnterExecutingMode(fed)

def subscribe_values(sub, fed):
    '''
    subscribe to values from federate1.py. See federate1.py, function with the name helicsPublicationPublishString.
    
    Right now:
    - it is a simple JSON dict with two keys: bat_1 and bat_2. It is not reading from files or
    any other source, just a simple dict.

    - It does not send the values to OpenDSS, it just prints them to the console to show values were received.
    
    In the DSSfederate.log file, you'll notice that the subscribed values are printed out five times, once for each time step.
    This is because the loop in the main function of federate1.py runs five times,
    publishing a new JSON string each time.
    
    The time steps are 60 seconds apart, so the values are published at t=60, t=120, t=180, t=240, and t=300 seconds.
    '''

    # Initialize OpenDSS:
    results_dir, profiles_dir, dss_file = set_paths()
    start_time = dt.datetime(2021, 12, 25)
    stepsize = dt.timedelta(minutes=1)
    dss = initialize_opendss(dss_file=dss_file, stepsize=stepsize, start_time=start_time)

    # Get storage names:

    storage_names = get_storage_names(dss=dss)
    print(f"Initialized OpenDSS with {len(storage_names)} storage elements")

    simulation_time_seconds = 3*3600 # three hours
    time_step_seconds = 60
    num_steps = simulation_time_seconds // time_step_seconds # five steps or every minutes

    for step in range(num_steps):

        target_time = (step + 1) * time_step_seconds # 60, 120, 180, ..
        granted_time = h.helicsFederateRequestTime(fed, target_time)
        
        raw = h.helicsInputGetString(sub)
        print(f"Step {step}: t={granted_time}: Raw data length: {len(raw) if isinstance(raw, str) else 'Not string'}")

        try:
            power_setpoints = json.loads(raw)
            total_power = sum(power_setpoints.values())
            print(f"Step {step}: t={granted_time}: Received {len(power_setpoints)} storage elements, Total: {total_power:.2f} kW")

            # Apply power setpoints to dss file
            successful_updates = 0
            for storage_name, power_value in power_setpoints.items():
                try:
                    dss.set_power(name=storage_name, element='Storage', p=-power_value, q=0, size=20.0) # The size is the kwrated value.
                    successful_updates +=1
                except Exception as e:
                    print("Something is happening, error setting power: ",e)
            print(f"Successfully set power for {successful_updates}/{len(power_setpoints)} storage elements")
            dss.run_dss()

        except Exception as e:
            print(f"Step {step}: t={granted_time}: JSON decode failed: {raw} → {e}")
        
        time.sleep(0.1)

        if granted_time >= 300:
            print("federate1 execution completed.")
            disconnect_federate(fed)
    
    print("DSSfederate subscription completed.")
    disconnect_federate(fed)

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
    dss.run_command('set controlmode=time')

    return dss

def get_storage_names(dss):
    '''
    Obtain the storage element names from the OpenDSS file. The OpenDSS wrapper helps a lot here. Such a relief.
    '''
    try:
        storage_element_names = dss.get_all_elements('Storage')
        storage_element_names = storage_element_names.index.to_list()
        storage_element_names = [i.replace('Storage.','') for i in storage_element_names]
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print(storage_element_names[0])
        # print(dss.get_all_properties(name='der_645_b_27', element='Storage'))
        # print(dss.get_property(name='der_645_b_27', property_name='kWrated', element='Storage'))
        print('\n\n')
        print('\n\n')
        print('\n\n')
        print('\n\n')
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

def disconnect_federate(fed):
    h.helicsFederateDisconnect(fed)
    print("DSSfederate done.")
    quit()

if __name__ == "__main__":
    fed, sub = initialize_federate()
    execute_federate(fed)
    subscribe_values(sub, fed)
    disconnect_federate(fed)
    print("DSSfederate execution completed.")
