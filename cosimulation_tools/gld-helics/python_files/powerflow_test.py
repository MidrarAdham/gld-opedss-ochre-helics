"""
Created on Thu June 26 09:28 a.m.
@author: MidrarAdham

References: 1abc_Transmission_simulator.py by Monish.Mukherjee
"""

import math
import numpy
import time
import random
import logging
import argparse
import helics as h
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from pypower.api import case118, ppoption, runpf, runopf

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)



def create_federate (config):
    """
    1) Create the federate from the JSON config file
    """
    fed = h.helicsCreateValueFederateFromConfig(config)
    return fed

def initialize_execute_federate (fed):
    """
    2) Enter the initialization and execution mode
    """
    h.helicsFederateEnterInitializingMode(fed) # Initialization
    status = h.helicsFederateEnterExecutingMode(fed)
    return status

def publications_handling (fed):
    pubkeys_count = h.helicsFederateGetPublicationCount(fed)    #Get the number of publication in each federate
    subkeys_count = h.helicsFederateGetInputCount(fed)  # Get the number of subscriptions in a federate.
    return pubkeys_count, subkeys_count

def get_power_info (filename):
    """
    Reads a csv file and returns a list of the needed values
    """
    df = pd.read_csv(filename)
    return df['time'].values ,df['watts'].values

def destroy_federate(fed):
    status = h.helicsFederateDisconnect(fed)
    h.helicsFederateDestroy(fed)
    logger.info("Federate finalized")


def federate_subscription():
    """
    TODO: subscription federate so you can get measurement data from GLD.
    """
    pass


if __name__ == "__main__":
    default_path = "/home/deras/gld-opedss-ochre-helics/gridlabd_helics_example"
    py_federate_config_file = f"{default_path}/python_files/powerflow_4node_config_py.json"
    gld_federate_config_file = f"{default_path}/gld_files/powerflow_4node_config_gld.json"
    filename = "../csv_profiles/der_profile_updated.csv"
    
    print('going into create federate function')
    fed = create_federate (py_federate_config_file)

    print('going into initialize_execute_federate')
    status = initialize_execute_federate (fed)
    print(f"Initialization status: {status}")
    
    print('going into publications_handling')
    pubkeys_count, subkey_count = publications_handling (fed)
    print(f"Publications: {pubkeys_count}, Subscriptions: {subkey_count}")
    
    print("going into helicsFederateGetPublicationByIndex")
    pubid = h.helicsFederateGetPublicationByIndex(fed, 0)
    
    # Get publication info for debugging
    pub_key = h.helicsPublicationGetKey(pubid)
    pub_type = h.helicsPublicationGetType(pubid)
    print(f"Publication key: {pub_key}")
    print(f"Publication type: {pub_type}")
    
    print("getting the power info from the csv file")
    sim_time, watts = get_power_info(filename)

    grantedtime = 0
    interval = 60
    total_steps = len(watts)

    start = time.time()    
    print("going into the loop")
    for t in range(total_steps):
        power_val = watts[t]
        time_val = sim_time[t]
        
        print(f"Step {t}: time={time_val}, power={power_val}")
        
        print(f"Publishing power value: {power_val}")
        try:
            h.helicsPublicationPublishDouble(pubid, float(power_val))
            print("Publication successful")
        except Exception as e:
            print(f"Publication failed: {e}")
        
        # Verify the publication was successful
        if h.helicsPublicationIsValid(pubid):
            print("Publication is valid")
        else:
            print("WARNING: Publication is not valid!")

        # Sync time
        requested_time = (t+1)*interval
        print(f"Requesting time: {requested_time}")
        grantedtime = h.helicsFederateRequestTime(fed, requested_time)
        print(f"Granted time: {grantedtime}")

        # Check if we're synchronized properly
        if grantedtime < requested_time:
            print(f"WARNING: Granted time {grantedtime} is less than requested time {requested_time}")

        end = time.time()
        elapsed = end - start
        print(f'Elapsed time: {elapsed:.2f} seconds')
        
        if elapsed >= 300:
            print("Stopping the simulation after 5 minutes")
            break

    print("destroying the federates")
    h.helicsFederateDisconnect(fed)
    destroy_federate(fed)