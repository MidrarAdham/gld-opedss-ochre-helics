import pandas as pd
import datetime as dt
from opendss_wrapper import OpenDSS

MasterFile =  './network_model/model_base.dss' 

start_time = dt.datetime(2021, 1, 1)
stepsize = dt.timedelta(minutes=1)
duration = dt.timedelta(days=1)
dss = OpenDSS([MasterFile], stepsize, start_time)
storage_elements = dss.get_all_elements('Storage')

for i in storage_elements.index.tolist():
    new_batt = i.split('.')[1].replace('der','DER')
    print("\n\n\n\n\n")
    print(new_batt)
    print("\n\n\n\n\n")
    batt_list.append(new_batt)
