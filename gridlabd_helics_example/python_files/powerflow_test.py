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
    fed = h.helicsCreateValueFederateFromConfig(config)
    pass

def initialize_execute_federate ():
    pass


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
    py_federate_config_file = "./powerflow_4node_config_py.json"
    gld_federate_config_file = "./powerflow_4node_config_gld.json"
    fed = create_federate(py_federate_config_file)
    print("\n\n-------------\n\n")
    print(fed)
    print("\n\n-------------\n\n")
    # pass