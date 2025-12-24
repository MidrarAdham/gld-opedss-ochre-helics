import os
import datetime as dt
from ochre import Dwelling
from ochre.utils import default_input_path
from ochre import CreateFigures
import matplotlib.pyplot as plt

dwelling_args = {
    # Timing parameters
    "start_time": dt.datetime(2025, 1, 1, 0, 0),  # (year, month, day, hour, minute)
    "time_res": dt.timedelta(minutes=10),         # time resolution of the simulation
    "duration": dt.timedelta(days=3),             # duration of the simulation

    # Input files
    "hpxml_file": os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml"),
    "hpxml_schedule_file": os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
    "weather_file": os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
}

new_equipment = {
    'EV': {
        "vehicle_type": "BEV",
        "charging_level":"Level 1",
        "range": 200,
    },
    "PV": {
        "capacity": 5,
    },
}

new_equipment_args = {
    **dwelling_args,
    "Equipment": new_equipment

}

dwelling = Dwelling(**new_equipment_args)

df, metrics, hourly = dwelling.simulate()

print(df.columns)

fig = CreateFigures.plot_time_series_detailed((df['Total Electric Power (kW)'],))

plt.show()

print(df.groupby(df.index.date).agg({'Total Electric Power (kW)': 'mean'}))