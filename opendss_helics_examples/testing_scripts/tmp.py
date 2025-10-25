import pandas as pd
import datetime as dt
from opendss_wrapper import OpenDSS
import re

start_time = dt.datetime(2021, 1, 1)
stepsize = dt.timedelta(minutes=1)

storage_dss_file = './network_model/model_base.dss'

dss = OpenDSS([storage_dss_file], stepsize, start_time)
batt = dss.get_all_elements('Storage')
batt_list = []

for i in batt.index.tolist():
    new_batt = i.split('.')[1]
    batt_list.append(new_batt)

print(batt_list)

