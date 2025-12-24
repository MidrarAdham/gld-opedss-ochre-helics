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

class OchreGridlabdMaster:
    '''
    Establishing the path needed for the files that we need to run
    '''

    def __init__(self):
        
        # OCHRE equipment file
        self.OCHRE_SCRIPT = "cosim_from_scratch.py"

        # GridLAB-D four node feeder
        self.GRIDLABD_MODEL = "powerflow_4node.glm"

        # We'll create a master config file, containing the path for the ochre and gridlab-d files and
        # how to run them.
        OUTPUT_DIR = os.getcwd()
        self.MASTER_CONFIG_FILE = os.path.join(OUTPUT_DIR, "master_cosim_config.json")

        # OCHRE creates a folder to store the building data and it saves it. This folder is called cosimulation
        OCHRE_MAIN_PATH = os.path.join(OUTPUT_DIR, "cosimulation")
        self.OCHRE_HOUSE_PATH = os.path.join(OCHRE_MAIN_PATH, "bldg0112631", "up00")


    def create_master_config(self):
        """
        - Creates a HELICS configuration file that launche both federates
        - This file is not called from main. It is called from the run_cosimulation() method
        """
        
        # OCHRE federate configuration
        ochre_cmd = f"{sys.executable} -u {self.OCHRE_SCRIPT} house House_1 {self.OCHRE_HOUSE_PATH}"
        ochre_cmd = ochre_cmd.replace("\\", "/")  # Fix Windows paths
        
        ochre_federate = {
            "name": "House_1",
            "host": "localhost",
            "directory": ".",
            "exec": ochre_cmd
        }
        
        # GridLAB-D federate configuration
        gridlabd_cmd = f"gridlabd {self.GRIDLABD_MODEL}"

        gridlabd_federate = {
            "name": "GridLABD_federate",
            "host": "localhost", 
            "directory": ".",
            "exec": gridlabd_cmd
        }
        
        # Master configuration - Avoiding the three terminal operation, we can have the broker
        # starting automatically!
        config = {
            "name": "ochre_gridlabd_cosimulation",
            "broker": True,  # Automatically start the HELICS broker
            "federates": [
                ochre_federate,
                gridlabd_federate
            ]
        }
        return config

    def run_cosimulation (self):

    # def main():
        """
        Main function to run the co-simulation
        """
        print("="*70)
        print("OCHRE + GridLAB-D Co-simulation via HELICS")
        print("="*70)
        
        # Debugging statements - Check the path of the required scripts:
        # Checking for ochre files
        if not os.path.exists(self.OCHRE_SCRIPT):
            print(f"ERROR: OCHRE script not found: {self.OCHRE_SCRIPT}")
            print("Please make sure the OCHRE script is in the current directory")
            return
        
        # Checking for the feeder files
        if not os.path.exists(self.GRIDLABD_MODEL):
            print(f"ERROR: GridLAB-D model not found: {self.GRIDLABD_MODEL}")
            print("Please make sure the .glm file is in the current directory")
            return
        
        # Checking for the buildign files:
        if not os.path.exists(self.OCHRE_HOUSE_PATH):
            print(f"ERROR: OCHRE building data not found: {self.OCHRE_HOUSE_PATH}")
            print("Run setup first:")
            print(f"  python3 {self.OCHRE_SCRIPT} setup")
            return
        
        # Once all the files existed, then let's create the configuration files:
        print("\nCreating master HELICS configuration...")
        config = self.create_master_config()
        
        
        # Save the config file to directory:
        with open(self.MASTER_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved here: {self.MASTER_CONFIG_FILE}")
        
        # Check ochre notebook to understand the commands to run for each federate:
            
            # OCHRE has a nice setup using click module, worth checking!
        print("\nFederates to be launched:")
        print("  1. OCHRE House (House_1)")
        print("  2. GridLAB-D (GridLABD_federate)")
        
        print("\nData flow:")
        print("  OCHRE → [house power] → GridLAB-D")
        
        print("\nStarting co-simulation...")
        print("="*70)
        
        # Ok, let's run the co-simulation
        # helics run is great. What this command does:

        # helics run --path master_cosim_config.json
        try:
            run(["--path", self.MASTER_CONFIG_FILE])
            print("="*70)
            print("Co-simulation completed successfully!")
            print("\nOutput files:")
            print(f"  - OCHRE results: {self.OCHRE_HOUSE_PATH}")
            print("  - GridLAB-D results: substation_power.csv, house_meter.csv, etc.")
        except Exception as e:
            print("="*70)
            print(f"ERROR during co-simulation: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Make sure GridLAB-D is installed and in your PATH")
            print("  2. Check that all files exist in the correct locations")
            print("  3. Verify OCHRE setup was run successfully")


if __name__ == "__main__":
    # set the class:
    master = OchreGridlabdMaster()
    # Call the run_cosimulation() method
    master.run_cosimulation()


