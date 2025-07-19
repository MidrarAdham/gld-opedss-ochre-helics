import pandas as pd

def read_files (filename):
    df = pd.read_csv(filename)
    print(df)
    return df['time'].values ,df['watts'].values

filename = '../csv_profiles/der_profile_updated.csv'
timee, watts = read_files (filename)
print(timee)
print("\n\n--------------\n\n")
print(watts)
