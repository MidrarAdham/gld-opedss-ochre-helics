import os
import shutil
import pandas as pd

def filter_datasets(dataset_path):
    '''
    This function looks into the ResStock datasets, checks buildings with no in.schedules.csv file,
    and ignore them. It also remove building IDs with empty in.schedules.csv

    The fuction returns a list of building IDs that have in.schedules.csv file

    TODO: Remove this function from here. It takes so much time. Also, this is bad implementation
    '''
    
    missing = []
    exists = []
    for folder in os.listdir(dataset_path):
        upgrade_path = os.path.join(dataset_path, folder, 'up00')
        input_path = upgrade_path
        if os.path.isdir(upgrade_path):
            target_file = os.path.join(upgrade_path, 'in.schedules.csv')
            if os.path.isfile(target_file):
                "Checking if in.schedules.csv is empty or not"
                try:
                    df = pd.read_csv(target_file)
                    bldg_id = upgrade_path.split('/')[-2]
                    exists.append(bldg_id)
                except pd.errors.EmptyDataError:
                    missing.append(bldg_id)
            else:
                "Checking if in.schedules.csv exists"
                # os.remove(target_file.split('/')[])
                missing.append(bldg_id)
    # exists is just a list of building IDs.
    return exists, missing

def delete_files (missing, input_path):
    for f in missing:
        target_dir = os.path.join(input_path,f,'up00')
        try:
            print(f'remove this directory: {target_dir}')
            shutil.rmtree(target_dir)
        except FileNotFoundError:
            continue
        except OSError:
            continue

dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/cosimulation'
existing, missing = filter_datasets(dataset_path=dataset_dir)

delete_files(missing=missing, input_path=dataset_dir)