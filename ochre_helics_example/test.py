import pandas as pd
import os
from ochre.utils.schedule import ALL_SCHEDULE_NAMES

# Path to the parent directory that contains all your folders
base_dir = '/home/deras/gld-opedss-ochre-helics/datasets/cosimulation/'
missing = []

for folder in os.listdir(base_dir):
    up00_path = os.path.join(base_dir, folder, "up00")
    if os.path.isdir(up00_path):
        target_file = os.path.join(up00_path, "in.schedules.csv")
        if os.path.isfile(target_file):
            try:
                df = pd.read_csv(target_file)
            except pd.errors.EmptyDataError:
                missing.append(up00_path)

        else:
        # if not os.path.isfile(target_file):
            missing.append(up00_path)
    else:
        print(f"Skipping: {folder} (no up00 folder found)")

for m in missing:
    bldg_id = m.split('/')[-2]
    # print(f"Building ID: {bldg_id}")
    if bldg_id == '28905':
        print(bldg_id)
    # print(m)
print(f"\nTotal folders missing in.schedules.csv: {len(missing)}\n")

