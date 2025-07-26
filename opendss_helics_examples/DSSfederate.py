# Author: Midrar
# Date: 2025-07-25
# reference: DSSfederate.py from dss-cosim repository, check the references.

import os
import datetime as dt
import pandas as pd
import helics as h
import json
from opendss_wrapper import OpenDSS

MainDir = os.path.abspath(os.path.dirname(__file__)) # Get the absolute path of the current directory
ResultsDir = os.path.join(MainDir, 'results')
os.makedirs(ResultsDir, exist_ok=True)  # Create results directory if it doesn't exist


# Create federate
fed = h.helicsCreateCombinationFederateFromConfig('./', "DSSfederate.json")

# register subscriptions

sub_storage_powers = h.helicsFederateRegisterSubscription(fed, "storage_powers", "")

MasterFile = os.path.join('./', 'model_base.dss')
# pv_dssfile = os.path.join('./', 'PVsystems.dss')
# storage_dssfile = os.path.join('./', 'BatteryStorage.dss')
start_time = dt.datetime(2021, 1, 1)
stepsize = dt.timedelta(minutes=1)
duration = dt.timedelta(days=1)
dss = OpenDSS([MasterFile], stepsize, start_time)

dss.run_command('set controlmode=time')
loads = dss.get_all_elements('Load')
storage = dss.get_all_elements('Storage')

