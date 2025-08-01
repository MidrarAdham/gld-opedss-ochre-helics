import os
import datetime as dt
import pandas as pd
import helics as h
import json
import time
from opendss_wrapper import OpenDSS

print("=== STARTING HELICS RUNNER DSSfederate ===")
print(f"Process start time: {time.ctime()}")

# Folder and File locations
MainDir = os.path.abspath(os.path.dirname(__file__))
ModelDir = os.path.join(MainDir, 'network_model')
ResultsDir = os.path.join(MainDir, 'results')
os.makedirs(ResultsDir, exist_ok=True)

print(f"Main Directory: {MainDir}")
print(f"Model Directory: {ModelDir}")
print(f"Results Directory: {ResultsDir}")

# Output files
main_results_file = os.path.join(ResultsDir, 'main_results.csv')
voltage_file = os.path.join(ResultsDir, 'voltage_results.csv')
storage_powers_results_file = os.path.join(ResultsDir, 'storage_powers_results.csv')
storage_soc_results_file = os.path.join(ResultsDir, 'storage_soc_results.csv')
storage_summary_file = os.path.join(ResultsDir, 'storage_summary.csv')

print("\n=== CREATING HELICS FEDERATE ===")
try:
    federate_config = os.path.join(os.path.dirname(__file__), "DSSfederate.json")
    fed = h.helicsCreateCombinationFederateFromConfig(federate_config)
    print("✓ HELICS federate created successfully")
    
    # Get federate info
    federate_name = h.helicsFederateGetName(fed)
    sub_count = h.helicsFederateGetInputCount(fed)
    print(f"Federate name: {federate_name}")
    print(f"Subscriptions count: {sub_count}")
    
except Exception as e:
    print(f"✗ FAILED to create HELICS federate: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== GETTING SUBSCRIPTIONS ===")
try:
    sub_storage_powers = h.helicsFederateGetInputByTarget(fed, "storage_powers")
    print("✓ Subscriptions obtained successfully")
except Exception as e:
    print(f"✗ FAILED to get subscriptions: {e}")
    exit(1)

print("\n=== CHECKING DSS FILES ===")
MasterFile = os.path.join(ModelDir, 'model_base.dss')
print(f"Master file exists: {os.path.exists(MasterFile)} - {MasterFile}")

print("\n=== CREATING OPENDSS INSTANCE ===")
try:
    start_time = dt.datetime(2021, 12, 25)  # Match federate1
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(minutes=30)  # Match federate1
    dss = OpenDSS([MasterFile], stepsize, start_time)
    print("✓ OpenDSS created successfully")
except Exception as e:
    print(f"✗ FAILED to create OpenDSS: {e}")
    exit(1)

print("\n=== RUNNING OPENDSS COMMANDS ===")
try:
    dss.run_command('set controlmode=time')
    print("✓ OpenDSS control mode set successfully")
except Exception as e:
    print(f"✗ FAILED to set OpenDSS control mode: {e}")
    exit(1)

print("\n=== GETTING DSS STORAGE ELEMENTS ===")
try:
    storage_elements_df = dss.get_all_elements('Storage')
    storage_element_names = list(storage_elements_df.index)
    
    clean_storage_names = []
    for name in storage_element_names:
        if name.startswith('Storage.'):
            clean_name = name.replace('Storage.', '')
        else:
            clean_name = name
        clean_storage_names.append(clean_name)
    
    print(f"✓ Found {len(clean_storage_names)} storage elements in OpenDSS")
    print(f"Sample storage elements: {clean_storage_names[:3]}")
    
    # Save storage info
    storage_elements_df.to_csv(os.path.join(ResultsDir, 'storage_info.csv'))
    print("✓ Storage element info saved")
    
except Exception as e:
    print(f"✗ FAILED to get DSS storage elements: {e}")
    exit(1)

print(f"\n=== ENTERING EXECUTION MODE ===")
try:
    print(f"Waiting for synchronization... Time: {time.ctime()}")
    h.helicsFederateEnterExecutingMode(fed)
    print("✓ Entered execution mode successfully")
    print(f"Synchronized! Time: {time.ctime()}")
except Exception as e:
    print(f"✗ FAILED to enter execution mode: {e}")
    exit(1)

print(f"\n=== STARTING SIMULATION LOOP ===")
simulation_start_time = time.time()

main_results = []
voltage_results = []
storage_powers_results = []
storage_soc_results = []
times = pd.date_range(start_time, freq=stepsize, end=start_time + duration)
total_steps = len(times) - 1

print(f"Total simulation steps: {total_steps}")

for step in range(total_steps):
    current_time = times[step]
    
    # Print progress every 30 steps (30 minutes)
    if step % 30 == 0:
        elapsed = time.time() - simulation_start_time
        print(f"\nSTEP {step}/{total_steps} at {current_time}")
        print(f"Elapsed real time: {elapsed:.1f}s")
    
    try:
        # Update time in co-simulation
        present_step = (current_time - start_time).total_seconds()
        granted_time = h.helicsFederateRequestTime(fed, present_step)
        
        if step % 30 == 0:
            print(f"HELICS time - Requested: {present_step}, Granted: {granted_time}")
            
    except Exception as e:
        print(f"✗ FAILED to request time at step {step}: {e}")
        break

    # =================== FIXED DATA RECEPTION SECTION ===================
    try:
        # Get storage power signals from other federate
        storage_powers = {}
        
        # ALWAYS try to get the data, ignore the update flag
        try:
            print(f"\n\n\n\nstorage data:{sub_storage_powers}\n\n\n\n")
            storage_powers_json = h.helicsInputGetString(sub_storage_powers)
            print(f"\n\n\n\nstorage data:{storage_powers_json}\n\n\n\n")
            
            if step % 30 == 0:
                print(f"DEBUG: Raw JSON type = {type(storage_powers_json)}")
                if isinstance(storage_powers_json, str):
                    print(f"DEBUG: Raw JSON length = {len(storage_powers_json)}")
            else:
                print(f"DEBUG: Raw JSON value = {storage_powers_json}")
            
            if storage_powers_json and len(storage_powers_json) > 0:
                storage_powers = json.loads(storage_powers_json)
                
                if step % 30 == 0:
                    print(f"✓ SUCCESS: Parsed {len(storage_powers)} storage power setpoints")
                    total_received = sum(storage_powers.values())
                    nonzero_received = sum(1 for v in storage_powers.values() if v != 0)
                    print(f"  Total power received: {total_received:.2f} kW")
                    print(f"  Non-zero setpoints: {nonzero_received}")
                    
                    # Show sample of received powers
                    sample_powers = {k: v for i, (k, v) in enumerate(storage_powers.items()) if i < 3}
                    print(f"  Sample setpoints: {sample_powers}")
            else:
                if step % 30 == 0:
                    print("○ PROBLEM: Empty or null JSON string received")
                    
                    # Additional debugging
                    isupdated = h.helicsInputIsUpdated(sub_storage_powers)
                    print(f"  Update flag: {isupdated}")
                    
        except Exception as read_error:
            if step % 30 == 0:
                print(f"✗ ERROR reading data: {read_error}")
                print(" stopping the simulation due to read error")
                quit()
            storage_powers = {}
                    
                # Diagnostic: Check input properties
            try:
                last_update_time = h.helicsInputLastUpdateTime(sub_storage_powers)
                input_type = h.helicsInputGetType(sub_storage_powers)
                print(f"  Last update time: {last_update_time}")
                print(f"  Input type: {input_type}")
                print(f"  Current simulation time: {present_step}")
            except Exception as diag_e:
                print(f"  Diagnostic error: {diag_e}")
            
            storage_powers = {}
                
        except Exception as e:
            print(f"✗ FAILED to get storage powers at step {step}: {e}")
            storage_powers = {}

    except Exception as e:
        print(f"✗ FAILED to get storage powers at step {step}: {e}")
        storage_powers = {}
    # =================== END FIXED DATA RECEPTION SECTION ===================

    try:
        # Set storage powers in OpenDSS
        successful_updates = 0
        total_power_set = 0.0
        update_errors = []
        
        for storage_name, power_setpoint in storage_powers.items():
            try:
                # Set power for storage element (note: using negative for charging)
                dss.set_power(storage_name, element='Storage', p=-power_setpoint)
                successful_updates += 1
                total_power_set += power_setpoint
            except Exception as e:
                update_errors.append(f"{storage_name}: {str(e)}")
                if len(update_errors) <= 3:  # Only show first 3 errors
                    if step % 30 == 0:
                        print(f"  ⚠ Failed to set power for {storage_name}: {e}")
        
        if step % 30 == 0:
            print(f"✓ Successfully updated {successful_updates}/{len(storage_powers)} storage elements")
            print(f"  Total power setpoint: {total_power_set:.2f} kW")
            if update_errors:
                print(f"  Update errors: {len(update_errors)}")
        
    except Exception as e:
        print(f"✗ FAILED to set DSS storage powers at step {step}: {e}")

    try:
        # Solve OpenDSS network model
        dss.run_dss()
        if step % 30 == 0:
            print("✓ DSS solved successfully")
    except Exception as e:
        print(f"✗ FAILED to solve DSS at step {step}: {e}")
        break

    try:
        # Get main circuit results
        circuit_info = dss.get_circuit_info()
        main_results.append(circuit_info)
        
        # Get voltage results (averaged)
        voltage_info = dss.get_all_bus_voltages(average=True)
        voltage_results.append(voltage_info)
        
        if step % 30 == 0:
            print("✓ Retrieved circuit info and voltages")
            total_power_kw = circuit_info.get('TotalPower(kW)', circuit_info.get('Total Power (kW)', 'N/A'))
            print(f"  Circuit Total Power: {total_power_kw} kW")
            
    except Exception as e:
        print(f"✗ FAILED to get circuit info at step {step}: {e}")

    try:
        # Get storage power and SOC results
        storage_data = dss.get_all_elements(element='Storage')
        
        # Extract actual powers
        storage_powers_actual = {}
        storage_soc_data = {}
        
        for idx, row in storage_data.iterrows():
            clean_name = idx.replace('Storage.', '') if idx.startswith('Storage.') else idx
            
            # Get actual power (check multiple possible column names)
            power_value = 0.0
            for power_col in ['kw', 'P(kW)', 'power_kw', 'Power', 'kW']:
                if power_col in storage_data.columns:
                    power_value = float(row[power_col])
                    break
            storage_powers_actual[clean_name] = power_value
            
            # Get SOC data (check multiple possible column names)
            soc_value = 0.0
            for soc_col in ['kwhstored', '%stored', 'SOC', 'StateOfCharge', 'stored_kwh']:
                if soc_col in storage_data.columns:
                    soc_value = float(row[soc_col])
                    break
            storage_soc_data[clean_name] = soc_value
        
        storage_powers_results.append(storage_powers_actual)
        storage_soc_results.append(storage_soc_data)
        
        if step % 30 == 0:
            total_actual_power = sum(storage_powers_actual.values())
            nonzero_actual = sum(1 for v in storage_powers_actual.values() if v != 0)
            avg_soc = sum(storage_soc_data.values()) / len(storage_soc_data) if storage_soc_data else 0
            
            print(f"✓ Storage results - Actual total power: {total_actual_power:.2f} kW")
            print(f"                  - Non-zero actual powers: {nonzero_actual}")
            print(f"                  - Average SOC: {avg_soc:.2f}")
            
            # Show available columns for debugging
            if step == 0:
                print(f"Available storage data columns: {list(storage_data.columns)}")
        
    except Exception as e:
        print(f"✗ FAILED to get storage data at step {step}: {e}")
        storage_powers_results.append({})
        storage_soc_results.append({})

print(f"\n=== SAVING RESULTS ===")
total_elapsed = time.time() - simulation_start_time
print(f"Total simulation time: {total_elapsed:.1f}s")

try:
    # Save main results
    main_df = pd.DataFrame(main_results)
    main_df.to_csv(main_results_file, index=False)
    print(f"✓ Main results saved: {len(main_df)} records")
    
    # Save voltage results
    voltage_df = pd.DataFrame(voltage_results)
    voltage_df.to_csv(voltage_file, index=False)
    print(f"✓ Voltage results saved: {len(voltage_df)} records")
    
    # Save storage power results
    storage_powers_df = pd.DataFrame(storage_powers_results)
    storage_powers_df.to_csv(storage_powers_results_file, index=False)
    print(f"✓ Storage power results saved: {storage_powers_df.shape}")
    
    # Save storage SOC results
    storage_soc_df = pd.DataFrame(storage_soc_results)
    storage_soc_df.to_csv(storage_soc_results_file, index=False)
    print(f"✓ Storage SOC results saved: {storage_soc_df.shape}")
    
    # Create storage summary statistics
    if not storage_powers_df.empty:
        summary_stats = {
            'total_elements': len(clean_storage_names),
            'simulation_steps': len(storage_powers_df),
            'total_energy_charged': storage_powers_df.sum().sum(),
            'avg_power_per_element': storage_powers_df.mean().mean(),
            'max_power_per_element': storage_powers_df.max().max(),
            'min_power_per_element': storage_powers_df.min().min(),
            'nonzero_values': (storage_powers_df != 0).sum().sum()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(storage_summary_file, index=False)
        print(f"✓ Storage summary saved")
        print(f"  Total energy charged: {summary_stats['total_energy_charged']:.2f} kWh")
        print(f"  Average power per element: {summary_stats['avg_power_per_element']:.4f} kW")
        print(f"  Non-zero values: {summary_stats['nonzero_values']}")
    
except Exception as e:
    print(f"✗ FAILED to save results: {e}")

print(f"\n=== FINALIZING FEDERATE ===")
try:
    h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
    print("✓ Federate finalized successfully")
except Exception as e:
    print(f"✗ FAILED to finalize federate: {e}")

print(f"=== DSSfederate COMPLETED at {time.ctime()} ===")