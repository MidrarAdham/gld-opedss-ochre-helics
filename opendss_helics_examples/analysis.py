import os
import matplotlib
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting

def set_paths():
    main_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_dir, "results/")
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    return main_dir, results_dir, results_files

def create_timestamps (results_dir, filename):
    
    print("creating timestamps for", filename)

    df = pd.read_csv(results_dir+filename)
    num_rows = len(df)
    timestamps = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-01 23:55:00', periods=num_rows)
    df['timestamp'] = timestamps.strftime('%Y-%m-%d %H:%M:%S')
    return df

if __name__ == "__main__":
    main_dir, results_dir, results_files = set_paths()
    for filename in results_files:
        print(f"Processing file: {filename}")

        # if filename == 'storage_powers_results.csv':
        if filename == 'main_results.csv':
            df = create_timestamps(results_dir=results_dir, filename=filename)
            cols = df.columns
            # Plot all columns except 'timestamp' vs timestamp in one plot
            plt.figure(figsize=(10, 6))
            for col in cols:
                if col != 'timestamp':
                    plt.plot(df['timestamp'], df[col], label=col)
            plt.xlabel('Timestamp')
            plt.ylabel('Value')
            plt.grid(True)
            plt.title('All Columns vs Timestamp')
            plt.legend()
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            break