"""
Master script to run OCHRE + GridLAB-D co-simulation via HELICS

This script creates a HELICS configuration that runs both:
1. OCHRE house federate (publishes power demand)
2. GridLAB-D federate (subscribes to power demand and simulates distribution system)

Usage:
    python3 run_ochre_gridlabd_cosim.py
"""

import json
import os
import sys
from helics.cli import run

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your OCHRE script (the one we created earlier)
OCHRE_SCRIPT = "cosim_from_scratch.py"

# Path to your GridLAB-D model file
GRIDLABD_MODEL = "powerflow_4node.glm"

# Where to save the master config file
OUTPUT_DIR = os.getcwd()
MASTER_CONFIG_FILE = os.path.join(OUTPUT_DIR, "master_cosim_config.json")

# Path where OCHRE stores its building data
OCHRE_MAIN_PATH = os.path.join(OUTPUT_DIR, "cosimulation")
OCHRE_HOUSE_PATH = os.path.join(OCHRE_MAIN_PATH, "bldg0112631", "up00")

# =============================================================================
# CREATE HELICS CONFIGURATION
# =============================================================================

def create_master_config():
    """
    Creates a HELICS configuration file that launches both federates
    """
    
    # OCHRE federate configuration
    ochre_cmd = f"{sys.executable} -u {OCHRE_SCRIPT} house House_1 {OCHRE_HOUSE_PATH}"
    ochre_cmd = ochre_cmd.replace("\\", "/")  # Fix Windows paths
    
    ochre_federate = {
        "name": "House_1",
        "host": "localhost",
        "directory": ".",
        "exec": ochre_cmd
    }
    
    # GridLAB-D federate configuration
    # NOTE: Adjust the gridlabd path if it's not in your system PATH
    gridlabd_cmd = f"gridlabd {GRIDLABD_MODEL}"

    
    
    gridlabd_federate = {
        "name": "GridLABD_federate",
        "host": "localhost", 
        "directory": ".",
        "exec": gridlabd_cmd
    }
    
    # Master configuration
    config = {
        "name": "ochre_gridlabd_cosimulation",
        "broker": True,  # Automatically start a HELICS broker
        "federates": [
            ochre_federate,
            gridlabd_federate
        ]
    }
    
    return config


def main():
    """
    Main function to run the co-simulation
    """
    print("="*70)
    print("OCHRE + GridLAB-D Co-simulation via HELICS")
    print("="*70)
    
    # Check if required files exist
    if not os.path.exists(OCHRE_SCRIPT):
        print(f"ERROR: OCHRE script not found: {OCHRE_SCRIPT}")
        print("Please make sure the OCHRE script is in the current directory")
        return
    
    if not os.path.exists(GRIDLABD_MODEL):
        print(f"ERROR: GridLAB-D model not found: {GRIDLABD_MODEL}")
        print("Please make sure the .glm file is in the current directory")
        return
    
    if not os.path.exists(OCHRE_HOUSE_PATH):
        print(f"ERROR: OCHRE building data not found: {OCHRE_HOUSE_PATH}")
        print("Please run setup first:")
        print(f"  python3 {OCHRE_SCRIPT} setup")
        return
    
    # Create master configuration
    print("\nCreating master HELICS configuration...")
    config = create_master_config()
    
    # Save to file
    with open(MASTER_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {MASTER_CONFIG_FILE}")
    
    # Display what will run
    print("\nFederates to be launched:")
    print("  1. OCHRE House (House_1)")
    print("  2. GridLAB-D (GridLABD_federate)")
    
    print("\nData flow:")
    print("  OCHRE → [house power] → GridLAB-D")
    
    print("\nStarting co-simulation...")
    print("="*70)
    
    # Run the co-simulation
    try:
        run(["--path", MASTER_CONFIG_FILE])
        print("="*70)
        print("Co-simulation completed successfully!")
        print("\nOutput files:")
        print(f"  - OCHRE results: {OCHRE_HOUSE_PATH}")
        print("  - GridLAB-D results: substation_power.csv, house_meter.csv, etc.")
    except Exception as e:
        print("="*70)
        print(f"ERROR during co-simulation: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure GridLAB-D is installed and in your PATH")
        print("  2. Check that all files exist in the correct locations")
        print("  3. Verify OCHRE setup was run successfully")


if __name__ == "__main__":
    main()
