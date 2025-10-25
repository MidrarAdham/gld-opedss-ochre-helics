import os
import pandas as pd

def set_paths():
    main_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_dir, "results/")
    profiles_dir = os.path.join(main_dir, "profiles/one_week_wh_data/")
    
    return results_dir, profiles_dir

def read_csv_files(profiles_dir):
    profile_files = [f for f in os.listdir(profiles_dir) if f.endswith('.csv')]
    print(profile_files)

if __name__ == '__main__':
    results_dir, profiles_dir = set_paths()
    read_csv_files(profiles_dir)

