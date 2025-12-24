# Simple OCHRE-HELICS Co-simulation Script
# This script runs ONE house in OCHRE as a HELICS federate
# The house publishes its power demand (no controls)
# Ready to connect with GridLAB-D federate

import os
import sys
import json
import click
import helics
import pandas as pd
import datetime as dt
from helics.cli import run
from ochre import Dwelling, Analysis
from ochre.utils import default_input_path


def create_dir():
    """Create a folder called 'cosimulation' to store all files"""
    main_path = os.path.join(os.getcwd(), "cosimulation")
    os.makedirs(main_path, exist_ok=True)
    return main_path

# Create the main folder
main_path = create_dir()

# Define which building to simulate
# You can change these to simulate different buildings
building_ids = ["bldg0112631"]  # Just ONE building
upgrades = ["up00", "up01"]     # Two upgrade scenario

# Create a dictionary to store the path for this house
house_paths = {}
i = 1
for building in building_ids:
    for upgrade in upgrades:
        house_paths[f"House_{i}"] = os.path.join(main_path, building, upgrade)
        i += 1

# Time settings
start_time = dt.datetime(2025, 1, 1)           # Start date
time_res = dt.timedelta(minutes=1)            # Time step = 10 minutes
duration = dt.timedelta(days=30)                # Simulate 1 day
sim_times = pd.date_range(
    start_time,
    start_time + duration,
    freq=time_res,
    inclusive="left",
)

initialization_time = dt.timedelta(days=1)

equipment_args = {
    # "PV": {"capacity": 5},                    # 5 kW solar panels
    # "Battery": {"capacity": 5, "capacity_kwh": 10},  # 5 kW, 10 kWh battery
}

# What data to publish from the house
# This is what GridLAB-D will receive
status_keys = [
    "Total Electric Power (kW)",  # Main thing GridLAB-D needs
]

# Weather file location
default_weather_file = os.path.join(
    default_input_path, "Weather", "USA_OR_Portland.Intl.AP.726980_TMY3.epw"
)

def make_helics_federate(name, config_file="ochre_helics_config.json"):
    """
    Create a HELICS federate from a JSON configuration file
    This sets up the connection to the HELICS broker
    """
    # Load the federate from the JSON config file
    fed = helics.helicsCreateValueFederateFromConfig(config_file)

    # Enter initialization mode and wait for other federates
    fed.enter_initializing_mode()
    return fed


def register_publication(name, fed, pub_type="string"):
    """
    Register a publication - this is how the house sends data to others
    """
    # return helics.helicsFederateRegisterGlobalTypePublication(fed, name, pub_type, "")
    return fed.get_publication_by_name(name)


def step_to(time, fed, offset=0):
    """
    Request the next time step in the co-simulation
    All federates must sync up at each time step
    """
    t_requested = (time - start_time).total_seconds() + offset
    while True:
        t_new = helics.helicsFederateRequestTime(fed, t_requested)
        if t_new >= t_requested:
            return

@click.group()
def cli():
    """OCHRE commands for co-simulation"""
    pass


@cli.command()
def setup():
    """
    COMMAND: setup
    Downloads the building data files from ResStock
    Run this FIRST, before running main
    Usage: python3 script.py setup
    """
    print("Downloading building data from ResStock...")
    for building in building_ids:
        for upgrade in upgrades:
            input_path = os.path.join(main_path, building, upgrade)
            os.makedirs(input_path, exist_ok=True)
            Analysis.download_resstock_model(building, upgrade, input_path, overwrite=False)
    print("Setup complete! Files saved to:", main_path)


@cli.command()
@click.argument("name", type=str)
@click.argument("input_path", type=click.Path(exists=True))
def house(name, input_path):
    """
    COMMAND: house
    Runs the OCHRE house simulation as a HELICS federate
    This is called automatically by 'main' - you don't run this directly
    """
    
    # Create HELICS federate
    fed = make_helics_federate(name)

    # Setup publication - house will publish its power demand
    # pub = register_publication(f"ochre_house_load.constant_power_12", fed, pub_type="complex")
    pub1 = register_publication(f"ochre_house_load_1.constant_power_12", fed, pub_type="complex")
    pub2 = register_publication(f"ochre_house_load_2.constant_power_12", fed, pub_type="complex")
    
    # NOTE: No subscription! House doesn't receive controls
    # It just runs naturally and publish power

    # Initialize OCHRE dwelling (the house simulation)
    print(f"Initializing OCHRE dwelling...")
    dwelling = Dwelling (
        name=name,
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        initialization_time=initialization_time,
        hpxml_file=os.path.join(input_path, "home.xml"),
        hpxml_schedule_file=os.path.join(input_path, "in.schedules.csv"),
        weather_file=default_weather_file,
        output_path=input_path,
        Equipment=equipment_args,
        save_args_to_json=True,
        # metrics_verbosity=5
    )
    print(f"{name} initialized successfully!")

    # Enter execution mode - simulation is ready to start
    fed.enter_executing_mode()
    
    pub1.publish(complex(0, 0))
    pub2.publish(complex(0, 0))
    print(f"{name} entering simulation loop...")

    for t in sim_times:
        # Sync with HELICS broker - wait for this time step - nice func
        step_to(t, fed)

        # Run OCHRE for one time step
        # Empty dictionary {} means no controls - house runs naturally unlike the ex in nrel repo
        status = dwelling.update({})

        # Get the house's real power demand (in kW)
        power_kw_1 = status.get("Total Electric Power (kW)", 0)

        # Get the house's real power demand (in kW)
        power_kw_2 = status.get("Total Electric Power (kW)", 0)



        
        # GridLAB-D uses Watts, OCHRE uses kW easpy peasy conversion
        power_w_1 = power_kw_1 * 1000
        power_w_2 = power_kw_2 * 1000
        
        power_complex_1 = complex(power_w_1, 0)
        power_complex_2 = complex(power_w_2, 0)
        pub1.publish(power_complex_1)
        pub2.publish(power_complex_2)
        
        # Print status (optional - helpful for debugging)
        print(f"{t}: Power = {power_kw_1:.2f} kW ({power_w_1:.0f} W)")
        print(f"{t}: Power = {power_kw_2:.2f} kW ({power_w_2:.0f} W)")

    # Simulation complete - save results and close
    print(f"{name} simulation complete!")
    dwelling.finalize()
    fed.finalize()


def get_house_fed_config(name, input_path):
    """
    Creates the configuration for the house federate
    This tells HELICS how to launch the house
    """
    cmd = f"{sys.executable} -u {__file__} house {name} {input_path}"
    cmd = cmd.replace("\\", "/")  # Fix for Windows paths
    return {
        "name": name,
        "host": "localhost",
        "directory": ".",
        "exec": cmd,
    }


@cli.command()
def main():
    """
    COMMAND: main
    Runs the complete co-simulation
    Usage: python3 script.py main
    """
    print("="*60)
    print("OCHRE-HELICS Co-simulation")
    print("="*60)
    
    # Write the config configuration for the house federate
    house_feds = [get_house_fed_config(name, path) for name, path in house_paths.items()]
    
    # This include:
    # - broker: True (HELICS will start a broker automatically)
    # - federates: list of all federates to run
    config = {
        "name": "ochre_cosimulation",
        "broker": True,
        "federates": house_feds  # Just the house - no aggregator!
    }

    # Save configuration to a JSON file
    config_file = os.path.join(main_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_file}")

    # Run the co-simulation using HELICS
    print("Starting co-simulation...")
    print("Output files will be saved to:", main_path)
    print("="*60)
    run(["--path", config_file])
    print("="*60)
    print("Co-simulation complete!")

cli.add_command(setup)
cli.add_command(house)
cli.add_command(main)

if __name__ == "__main__":
    cli()