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
from ochre import Analysis


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
@click.argument("input_path_1", type=click.Path(exists=True))
@click.argument("input_path_2", type=click.Path(exists=True))
def house(name, input_path_1, input_path_2):
    """
    COMMAND: house
    Runs the OCHRE house simulation as a HELICS federate
    This is called automatically by 'main' - you don't run this directly
    """

    # Load pre-computed OCHRE results for each triplex_load
    def load_csv(path):
        csv_path = os.path.join(path, "ochre.csv")
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Time"])
        return df.set_index("Time")

    df1 = load_csv(input_path_1)
    df2 = load_csv(input_path_2)
    print(f"{name} data loaded: {len(df1)} timesteps (load_1), {len(df2)} timesteps (load_2)")

    # Create HELICS federate
    fed = make_helics_federate(name)

    pub1 = register_publication(f"ochre_house_load_1.constant_power_12", fed, pub_type="complex")
    pub2 = register_publication(f"ochre_house_load_2.constant_power_12", fed, pub_type="complex")

    # Enter execution mode - simulation is ready to start
    fed.enter_executing_mode()

    pub1.publish(complex(0, 0))
    pub2.publish(complex(0, 0))
    print(f"{name} entering simulation loop...")

    for t in sim_times:
        # Sync with HELICS broker - wait for this time step
        step_to(t, fed)

        # Get power independently from each building's CSV (kW -> W)
        power_kw_1 = df1["Total Electric Power (kW)"].get(t, 0)
        power_kw_2 = df2["Total Electric Power (kW)"].get(t, 0)

        pub1.publish(complex(power_kw_1 * 1000, 0))
        pub2.publish(complex(power_kw_2 * 1000, 0))

        print(f"{t}: load_1 = {power_kw_1:.2f} kW | load_2 = {power_kw_2:.2f} kW")

    print(f"{name} simulation complete!")
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