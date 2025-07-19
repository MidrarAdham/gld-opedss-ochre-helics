import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("./der_profile.csv", skiprows=7,usecols=['# timestamp','tn_meter_5br_231:measured_real_power'])
df = df.rename(columns={'# timestamp':'time', 'tn_meter_5br_231:measured_real_power':'watts'})
df['time'] = pd.to_datetime(df['time'].str.replace(' UTC', '', regex=False))


def plotting (x,y, output_file):
    # Plot watts vs time
    plt.figure(figsize=(10, 5))
    plt.plot(x,y, label='Watts')
    plt.xlabel('Time')
    plt.ylabel('Watts')
    plt.title('Watts vs Time')
    plt.legend()
    plt.tight_layout()
    # Set a small number of x-axis ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.savefig(output_file, dpi=300)
    plt.close()


df2 = pd.read_csv("./powerflow_4node_logs.csv", skiprows=8)
print(df2)
# df2 = pd.to_datetime(df2['# timestamp'])

plotting(df['time'], df['watts'], 'quickie.png')
plotting(df2['# timestamp'], df2['rated_power'], 'quickie2.png')
