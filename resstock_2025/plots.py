import pandas as pd
import seaborn as sns
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