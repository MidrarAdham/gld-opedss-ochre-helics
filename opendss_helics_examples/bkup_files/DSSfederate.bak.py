import os
import datetime as dt
import pandas as pd
import helics as h
import json
from opendss_wrapper import OpenDSS

print("=== STARTING DSSfederate ===")

# Folder and File locations
MainDir = os.path.abspath(os.path.dirname(__file__))
ModelDir = os.path.join(MainDir, 'network_model')
ResultsDir = os.path.join(MainDir, 'results')
os.makedirs(ResultsDir, exist_ok=True)

print(f"Main Directory: {MainDir}")
print(f"Model Directory: {ModelDir}")
print(f"Results Directory: {ResultsDir}")

# Output files
load_info_file = os.path.join(ResultsDir, 'load_info.csv')
pv_info_file = os.path.join(ResultsDir, 'pv_info.csv')
main_results_file = os.path.join(ResultsDir, 'main_results.csv')
voltage_file = os.path.join(ResultsDir, 'voltage_results.csv')
elements_file = os.path.join(ResultsDir, 'element_results.csv')
pv_powers_results_file = os.path.join(ResultsDir, 'pv_powers_results.csv')
soc_results_file = os.path.join(ResultsDir, 'soc_results.csv')
test_file = os.path.join(ResultsDir, 'test_file.csv')

print("=== CREATING HELICS FEDERATE ===")
try:
    # Create federate
    fed = h.helicsCreateCombinationFederateFromConfig(
        os.path.join(os.path.dirname(__file__), "DSSfederate.json")
    )
    print("✓ HELICS federate created successfully")
except Exception as e:
    print(f"✗ FAILED to create HELICS federate: {e}")
    exit(1)

print("=== GETTING SUBSCRIPTIONS ===")
try:
    # register subscriptions
    sub_pv_powers = h.helicsFederateGetInputByTarget(fed, "pv_powers")
    sub_storage_powers = h.helicsFederateGetInputByTarget(fed, "storage_powers")
    print("✓ Subscriptions obtained successfully")
except Exception as e:
    print(f"✗ FAILED to get subscriptions: {e}")
    exit(1)

print("=== CHECKING DSS FILES ===")
MasterFile = os.path.join(ModelDir, 'Master.dss')
pv_dssfile = os.path.join(ModelDir, 'PVsystems.dss')
storage_dssfile = os.path.join(ModelDir, 'BatteryStorage.dss')

print(f"Master file exists: {os.path.exists(MasterFile)} - {MasterFile}")
print(f"PV file exists: {os.path.exists(pv_dssfile)} - {pv_dssfile}")
print(f"Storage file exists: {os.path.exists(storage_dssfile)} - {storage_dssfile}")

print("=== CREATING OPENDSS INSTANCE ===")
try:
    start_time = dt.datetime(2021, 1, 1)
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(days=1)
    dss = OpenDSS([MasterFile, pv_dssfile, storage_dssfile], stepsize, start_time)
    print("✓ OpenDSS created successfully")
except Exception as e:
    print(f"✗ FAILED to create OpenDSS: {e}")
    exit(1)

print("=== RUNNING OPENDSS COMMANDS ===")
try:
    # Run additional OpenDSS commands
    dss.run_command('set controlmode=time')
    print("✓ OpenDSS control mode set successfully")
except Exception as e:
    print(f"✗ FAILED to set OpenDSS control mode: {e}")
    exit(1)

print("=== GETTING DSS ELEMENTS ===")
try:
    # Get info on all properties of a class
    df = dss.get_all_elements('Load')
    df.to_csv(load_info_file)
    print(f"✓ Load elements: {len(df)} found")
    
    df = dss.get_all_elements(element='Generator')
    df.to_csv(pv_info_file)
    print(f"✓ Generator elements: {len(df)} found")
except Exception as e:
    print(f"✗ FAILED to get DSS elements: {e}")
    exit(1)

print("=== ENTERING EXECUTION MODE ===")
try:
    h.helicsFederateEnterExecutingMode(fed)
    print("✓ Entered execution mode successfully")
except Exception as e:
    print(f"✗ FAILED to enter execution mode: {e}")
    exit(1)

print("=== STARTING SIMULATION LOOP ===")
main_results = []
voltage_results = []
element_results = []
pv_powers_results = []
soc_results = []
times = pd.date_range(start_time, freq=stepsize, end=start_time + duration)

print(f"Total simulation steps: {len(times)}")

for step, current_time in enumerate(times):
    print(f"\n--- STEP {step}: {current_time} ---")
    
    try:
        # Update time in co-simulation
        present_step = (current_time - start_time).total_seconds()
        present_step += 1  # Ensures other federates update before DSS federate
        h.helicsFederateRequestTime(fed, present_step)
        print(f"✓ Time requested: {present_step}")
    except Exception as e:
        print(f"✗ FAILED to request time: {e}")
        break

    try:
        # get signals from other federate
        isupdated = h.helicsInputIsUpdated(sub_pv_powers)
        if isupdated == 1:
            pv_powers = h.helicsInputGetString(sub_pv_powers)
            pv_powers = json.loads(pv_powers)
            print(f"✓ Received pv_powers: {pv_powers}")
        else:
            pv_powers = {}
            print("○ No pv_powers update")
    except Exception as e:
        print(f"✗ FAILED to get pv_powers: {e}")
        pv_powers = {}

    try:
        isupdated = h.helicsInputIsUpdated(sub_storage_powers)
        if isupdated == 1:
            storage_powers = h.helicsInputGetString(sub_storage_powers)
            storage_powers = json.loads(storage_powers)
            print(f"✓ Received storage_powers: {storage_powers}")
        else:
            storage_powers = {}
            print("○ No storage_powers update")
    except Exception as e:
        print(f"✗ FAILED to get storage_powers: {e}")
        storage_powers = {}

    try:
        # set pv, storage powers
        for pv_name, set_point in pv_powers.items():
            dss.set_power(pv_name, element='Generator', p=set_point)
            print(f"✓ Set {pv_name} power to {set_point}")

        for storage_name, set_point in storage_powers.items():
            dss.set_power(storage_name, element='Storage', p=-set_point)
            print(f"✓ Set {storage_name} power to {-set_point}")
    except Exception as e:
        print(f"✗ FAILED to set DSS powers: {e}")
        break

    try:
        # solve OpenDSS network model
        dss.run_dss()
        print("✓ DSS solved successfully")
    except Exception as e:
        print(f"✗ FAILED to solve DSS: {e}")
        break
      
    try:
        # Get outputs for the feeder, all voltages, and individual element voltage and power
        main_results.append(dss.get_circuit_info())
        voltage_results.append(dss.get_all_bus_voltages(average=True))
        print("✓ Got circuit info and voltages")
    except Exception as e:
        print(f"✗ FAILED to get circuit info: {e}")
        break

    try:
        element_results.append({
            'PV Power (kW)': dss.get_power('pv2', element='Generator', total=True)[0],
            'Storage Power (kW)': dss.get_power('battery2', element='Storage', total=True)[0],
            'Line Power (kW)': dss.get_power('L15', element='Line', line_bus=1)[0],
            'Load Power (kW)': dss.get_power('S10a', element='Load', total=True)[0],
            'Load Power (kVAR)': dss.get_power('S10a', element='Load', total=True)[1],
            'PV Voltage (p.u.)': dss.get_voltage('pv2', element='Generator', average=True),
            'Storage Voltage (p.u.)': dss.get_voltage('battery2', element='Storage', average=True),
            'Line Voltage (p.u.)': dss.get_voltage('L15', element='Line', average=True),
            'Load Voltage (p.u.)': dss.get_voltage('S10a', element='Load', average=True),
        })
        print("✓ Got element powers and voltages")
    except Exception as e:
        print(f"✗ FAILED to get element data: {e}")
        break
    
    try:
        # get pv data
        pv_powers_data = {}
        for pv_name in pv_powers:
            pv_powers_data.update({pv_name: dss.get_power(pv_name, element='Generator', total=True)[0]})    
        pv_powers_results.append(pv_powers_data)
        print(f"✓ Got PV data: {pv_powers_data}")
    except Exception as e:
        print(f"✗ FAILED to get PV data: {e}")
        pv_powers_results.append({})

    try:
        # get storage data
        storage_data = dss.get_all_elements(element='Storage')
        print(f"Available storage columns: {list(storage_data.columns)}")
        
        storage_soc = {}
        for idx, row in storage_data.iterrows():
            # Check what columns are actually available
            # Common storage properties might be: 'kwhstored', '%stored', 'state', etc.
            if 'kwhstored' in storage_data.columns:
                storage_soc.update({idx.replace('Storage.',''): row['kwhstored']})
            elif '%stored' in storage_data.columns:
                storage_soc.update({idx.replace('Storage.',''): row['%stored']})
            else:
                # Use the first numeric column as fallback
                numeric_cols = storage_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    storage_soc.update({idx.replace('Storage.',''): row[numeric_cols[0]]})
                else:
                    storage_soc.update({idx.replace('Storage.',''): 0.0})
        
        soc_results.append(storage_soc)
        print(f"✓ Got storage SOC: {storage_soc}")
    except Exception as e:
        print(f"✗ FAILED to get storage data: {e}")
        soc_results.append({})

    print(f"✓ STEP {step} COMPLETED SUCCESSFULLY")
    
    # Only run a few steps for debugging
    if step >= 2:
        print("=== STOPPING AFTER 3 STEPS FOR DEBUGGING ===")
        break

print("=== SAVING RESULTS ===")
try:
    pd.DataFrame(main_results).to_csv(main_results_file)
    pd.DataFrame(voltage_results).to_csv(voltage_file)
    pd.DataFrame(element_results).to_csv(elements_file)
    pd.DataFrame(pv_powers_results).to_csv(pv_powers_results_file)
    pd.DataFrame(soc_results).to_csv(soc_results_file)
    print("✓ Results saved successfully")
except Exception as e:
    print(f"✗ FAILED to save results: {e}")

print("=== FINALIZING FEDERATE ===")
try:
    # finalize and close the federate
    h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
    print("✓ Federate finalized successfully")
except Exception as e:
    print(f"✗ FAILED to finalize federate: {e}")

print("=== DSSfederate COMPLETED ===")