import os
import datetime as dt
from ochre import Analysis
from ochre import CreateFigures
from ochre import ElectricVehicle
import matplotlib.pyplot as plt


equipment_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=15),
    "duration": dt.timedelta(days=10),
    "save_results": False,  # if True, must specify output_path
    "output_path": os.getcwd(),
    "seed": 1,  # setting random seed to create consistent charging events

    # Equipment-specific parameters
    "vehicle_type": "PHEV",
    "charging_level": "Level 1",
    "range": 200,
}

# Initialize equipment
equipment = ElectricVehicle(**equipment_args)

# Simulate equipment
df = equipment.simulate()

print(df.head())

# fig, ax = plt.subplots()

# for col in df.columns:
#     if not "Unmet" in col:
#         ax.plot(df.index, df[col], label=col)
#         ax.set_xlabel("Time")
#         ax.set_ylabel(col)
#         ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
#         ax.tick_params(axis='x', rotation=45)
# ax.legend()
# ax.grid(alpha=0.5)

fig = CreateFigures.plot_daily_profile(df_raw=df, column="EV Electric Power (kW)", plot_max=False, plot_min=False, plot_average=False)
plt.show()


# fig, ax = plt.subplots()
# ax.plot(df.index, df['EV Electric Power (kW)'])
# plt.show()
# plt.close()

# fig = CreateFigures.plot_time_series_detailed((df["EV SOC (-)"],))