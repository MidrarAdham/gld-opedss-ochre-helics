import os
import pandas as pd
import seaborn as sns
from new import LoadProfiles
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_ldc_bar(ldc_df):
    df = ldc_df.copy()

    df["duration_bin"] = pd.cut(df["duration_fraction"], bins=96, labels=False)
    bin_means = df.groupby("duration_bin")["P_sorted_kW"].mean().reset_index()
    bin_means["P_sorted_kW"] = bin_means["P_sorted_kW"] / 1000

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=bin_means, x="duration_bin", y="P_sorted_kW")
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.set_xlabel("Percent of Time (%)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Load Duration Curve")
    plt.tight_layout()
    return fig, ax

def line_plots (df: pd.DataFrame, x: str, y:str, title: str):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x=x, y=y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


analyzer = LoadProfiles(
    dataset_dir=f"{os.getcwd()}/datasets/cosimulation/",
    n_buildings=10,
    upgrades=['up00']
    )

analyzer.run()
data = analyzer.load_profiles[0]

interval_df = (
    data[['Time','Total Electric Power (kW)', 'Total Reactive Power (kVAR)',
       'Total Electric Energy (kWh)', 'Total Reactive Energy (kVARh)',
       'interval_start', 'Total Electric Power Average Demand (kW)',
       'Total Electric Power Max Demand (kW)', 'Energy (kWh)']]
    .drop_duplicates(subset='interval_start')
    .sort_values('interval_start')
)

interval_df['interval_start'] = pd.to_datetime(interval_df['interval_start']).dt.strftime('%d %H:%M')
interval_df['Time'] = pd.to_datetime(interval_df['Time']).dt.strftime('%d %H:%M')


fig, ax = plt.subplots(figsize=(16, 6))

# 1) Bars = instantaneous power
sns.barplot(
    data=interval_df,
    x='interval_start',
    y='Total Electric Power (kW)',
    ax=ax,
    label='Instantaneous Power (kW)'
)

# 2) Line = average demand
sns.lineplot(
    data=interval_df,
    x='interval_start',
    y='Total Electric Power Average Demand (kW)',
    ax=ax,
    color='black',
    label='Average Demand (kW)'
)

sns.lineplot(
    data=interval_df,
    x='interval_start',
    y='Energy (kWh)',
    ax=ax,
    color='lightcoral',
    label='Energy (kWh)'
)

sns.scatterplot (
    data=interval_df,
    x='interval_start',
    y='Total Electric Power Max Demand (kW)',
    color = 'crimson'

)

# Labels and formatting
ax.set_title("Instantaneous Power (bars) vs Average Demand (line)")
ax.set_xlabel("Time (5-minute intervals)")
ax.set_ylabel("Power (kW)")

ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
ax.tick_params(axis='x', rotation=45)

ax.legend()
fig.tight_layout()
plt.show()


# fig, ax = plt.subplots(2,1, figsize=(16, 10))


# sns.barplot(
#     data=interval_df,
#     x='interval_start',
#     y='Total Electric Power (kW)',
#     color='steelblue',
#     ax=ax[0]
# )

# ax[0].set_xlabel("Time (Day hour:minute)")
# ax[0].set_ylabel("Real Power (kW)")
# ax[0].set_title("Instantaneous (kW)")
# ax[0].xaxis.set_major_locator(ticker.MaxNLocator(20))
# ax[0].tick_params(axis='x', rotation=45)


# sns.lineplot (
#     data=interval_df,
#     x='Time',
#     y='Total Electric Power Average Demand (kW)',
#     ax=ax[1]
# )


# # sns.barplot(
# #     data=interval_df,
# #     x='Time',
# #     y='Total Electric Power Average Demand (kW)',
# #     color='steelblue',
# #     ax=ax[1]
# # )

# ax[1].set_xlabel("Time (Day hour:minute)")
# ax[1].set_ylabel("Average Demand (kW)")
# ax[1].set_title("Average Demand per 5-Minute Interval")
# ax[1].xaxis.set_major_locator(ticker.MaxNLocator(20))
# ax[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.show()

