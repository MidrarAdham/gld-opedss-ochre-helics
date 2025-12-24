import os
import sys
import json
import helics as h
import pandas as pd
import datetime as dt
import time
from opendss_wrapper import OpenDSS

def get_storage_element_names(dss):
    """Extract storage element names from OpenDSS"""
    print("Getting storage element names from DSS file")
    try:
        storage_elements = dss.get_all_elements('Storage')
        batt_list = []
        for index_name in storage_elements.index.tolist():
            if isinstance(index_name, str) and 'Storage.' in index_name:
                clean_name = index_name.split('.')[1]
            else:
                clean_name = str(index_name)
            batt_list.append(clean_name)
        
        print(f"✓ Found {len(batt_list)} storage elements")
        return batt_list
        
    except Exception as e:
        print(f"✗ FAILED to get storage element names: {e}")
        sys.exit(1)

def read_csv_files(profiles_dir, storage_names):
    """Read CSV files and map them to storage elements"""
    print("Loading the load profiles from CSV files")
    profile_data = {}
    
    try:
        csv_files = [f for f in os.listdir(profiles_dir) if f.endswith('.csv')]
        csv_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"Found {len(csv_files)} CSV files")
        print(f"Need data for {len(storage_names)} storage elements")
        
        for i, storage_name in enumerate(storage_names):
            if i < len(csv_files):
                csv_file = csv_files[i]
                file_path = os.path.join(profiles_dir, csv_file)
                
                try:
                    df = pd.read_csv(file_path)
                    if len(df.columns) >= 2:
                        power_values = df.iloc[:, 1].values
                        # Convert to float and handle NaN values
                        power_values = [float(x) if pd.notna(x) else 0.0 for x in power_values]
                        profile_data[storage_name] = power_values
                        
                        if i < 3:  # Show first 3 for debugging
                            print(f"  ✓ Loaded {csv_file} for {storage_name}")
                            print(f"    Data points: {len(power_values)}")
                            nonzero_count = sum(1 for v in power_values if v != 0)
                            max_value = max(power_values)
                            min_value = min(power_values)
                            print(f"    Non-zero: {nonzero_count}, Range: {min_value:.2f} to {max_value:.2f}")
                    else:
                        profile_data[storage_name] = [0.0] * 1440
                        
                except Exception as e:
                    print(f"  ✗ Error reading {csv_file}: {e}")
                    profile_data[storage_name] = [0.0] * 1440
            else:
                profile_data[storage_name] = [0.0] * 1440
                
        print(f"✓ Successfully loaded profiles for {len(profile_data)} storage elements")
        return profile_data
        
    except Exception as e:
        print(f"✗ FAILED to read CSV files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== STARTING HELICS RUNNER FEDERATE1 ===")
    print(f"Process start time: {time.ctime()}")
    
    # Simulation parameters - Match your CSV timestamps
    start_time = dt.datetime(2021, 12, 25)
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(minutes=30)  # Start with 3 hours for testing
    
    # Set up paths
    MainDir = os.path.abspath(os.path.dirname(__file__))
    ProfilesDir = os.path.join(MainDir, 'profiles/one_week_wh_data')
    ModelDir = os.path.join(MainDir, 'network_model')
    
    print(f"Main Directory: {MainDir}")
    print(f"Profiles Directory: {ProfilesDir}")
    print(f"Profiles dir exists: {os.path.exists(ProfilesDir)}")
    
    # Create OpenDSS instance
    try:
        dss_file = os.path.join(ModelDir, 'model_base.dss')
        print(f"DSS file: {dss_file}")
        dss = OpenDSS([dss_file], time_step=stepsize, start_time=start_time)
        print("✓ OpenDSS instance created successfully")
    except Exception as e:
        print(f"✗ FAILED to create OpenDSS instance: {e}")
        sys.exit(1)
    
    # Create HELICS federate (no coreInit needed with runner)
    try:
        print("\n=== CREATING HELICS FEDERATE ===")
        federate_config = os.path.join(os.path.dirname(__file__), "federate1.json")
        print(f"Config file: {federate_config}")
        
        fed = h.helicsCreateCombinationFederateFromConfig(federate_config)
        print("✓ HELICS federate created successfully")
        
        # Get federate info
        federate_name = h.helicsFederateGetName(fed)
        pub_count = h.helicsFederateGetPublicationCount(fed)
        print(f"Federate name: {federate_name}")
        print(f"Publications count: {pub_count}")
        
    except Exception as e:
        print(f"✗ FAILED to create HELICS federate: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get publications
    try:
        print("\n=== GETTING PUBLICATIONS ===")
        pub_storage_powers = h.helicsFederateGetPublication(fed, "storage_powers")
        print("✓ Publications obtained successfully")
    except Exception as e:
        print(f"✗ FAILED to get publications: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get storage element names and load data
    print("\n=== LOADING DATA ===")
    storage_element_names = get_storage_element_names(dss)
    print(f"Storage element sample: {storage_element_names[:3]}")
    
    profile_data = read_csv_files(ProfilesDir, storage_element_names)

    # Enter HELICS execution mode
    try:
        print(f"\n=== ENTERING HELICS EXECUTION MODE ===")
        print(f"Waiting for synchronization... Time: {time.ctime()}")
        
        h.helicsFederateEnterExecutingMode(fed)
        print("✓ Entered HELICS execution mode")
        print(f"Synchronized! Time: {time.ctime()}")
    except Exception as e:
        print(f"✗ FAILED to enter HELICS execution mode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Simulation loop
    print(f"\n=== STARTING SIMULATION LOOP ===")
    simulation_start_time = time.time()
    
    times = pd.date_range(start_time, freq=stepsize, end=start_time + duration)
    total_steps = len(times) - 1
    print(f"Total simulation steps: {total_steps}")
    
    for step in range(total_steps):
        current_time = times[step]
        step_start_time = time.time()
        
        # Print progress every 30 steps (30 minutes)
        if step % 30 == 0:
            elapsed = time.time() - simulation_start_time
            print(f"\nSTEP {step}/{total_steps} at {current_time}")
            print(f"Elapsed real time: {elapsed:.1f}s, Real time: {time.ctime()}")
            
        try:
            # Update time in co-simulation
            present_step = (current_time - start_time).total_seconds()
            
            granted_time = h.helicsFederateRequestTime(fed, present_step)
            
            if step % 30 == 0:
                print(f"HELICS time - Requested: {present_step}, Granted: {granted_time}")
            
        except Exception as e:
            print(f"✗ FAILED to request time at step {step}: {e}")
            break

        try:
            # Get power setpoints for current timestep
            storage_powers = {}
            
            for storage_name in storage_element_names:
                current_profile = profile_data[storage_name]
                
                if step < len(current_profile):
                    power_value = float(current_profile[step])
                    storage_powers[storage_name] = power_value
                else:
                    storage_powers[storage_name] = 0.0
            
            # Publish storage powers to HELICS
            json_string = json.dumps(storage_powers)
            h.helicsPublicationPublishString(pub_storage_powers, json_string)
            
            # Show statistics every 30 steps
            if step % 30 == 0:
                total_power = sum(storage_powers.values())
                nonzero_count = sum(1 for v in storage_powers.values() if v != 0)
                max_power = max(storage_powers.values()) if storage_powers else 0
                
                print(f"✓ Published powers - Total: {total_power:.2f} kW")
                print(f"  Non-zero elements: {nonzero_count}/{len(storage_powers)}")
                print(f"  Max power: {max_power:.2f} kW")
                
                # Show sample of first few non-zero values
                sample_nonzero = {k: v for k, v in storage_powers.items() if v != 0}
                if sample_nonzero:
                    sample_items = list(sample_nonzero.items())[:3]
                    print(f"  Sample non-zero: {sample_items}")
                else:
                    print("  ⚠ All power values are zero!")
                
        except Exception as e:
            print(f"✗ FAILED to publish storage powers at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Finalize the federate
    print(f"\n=== FINALIZING FEDERATE ===")
    total_elapsed = time.time() - simulation_start_time
    print(f"Total simulation time: {total_elapsed:.1f}s")
    
    try:
        h.helicsFederateFinalize(fed)
        h.helicsFederateFree(fed)
        h.helicsCloseLibrary()
        print("✓ HELICS federate finalized successfully")
    except Exception as e:
        print(f"✗ FAILED to finalize HELICS federate: {e}")

    print(f"=== FEDERATE1 COMPLETED at {time.ctime()} ===")