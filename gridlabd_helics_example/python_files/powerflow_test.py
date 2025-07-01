"""
Created on Thu June 26 09:28 a.m.
@author: MidrarAdham

References: 1abc_Transmission_simulator.py by Monish.Mukherjee
"""

import scipy.io as spio
from pypower.api import case118, ppoption, runpf, runopf
import math
import numpy
import matplotlib.pyplot as plt
import time
import helics as h
import random
import logging
import argparse

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
    pass

def publications_handling (fed):
    pubkeys_count = h.helicsFederateGetPublicationCount(fed)    #Get the number of publication in each federate
    subkeys_count = h.helicsFederateGetInputCount(fed)  # Get the number of subscriptions in a federate.
    return pubkeys_count, subkeys_count

def time_synchronization_federate ():
    """
    needed to sync with GLD every time step.
    """
    grantedtime = h.helicsFederateRequestTime(fed, h.HELICS_TIME_MAXTIME)
    pass

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
    
    print("I am here 0")
    fed = create_federate (py_federate_config_file)
    print(fed)
    initialize_execute_federate (fed)
    print("I am here 2")
    pubkeys_count, subkey_count = publications_handling (fed)
    print("I am here 4")
    print(f"fed {fed}\nfederate name: none for now\npublications in federate: {pubkeys_count}\nsubscriptions in federate: {subkey_count}")
    print("I am here 5")