# %%
# Import libs
import os
import numpy as np
import pandas as pd
from pathlib import Path
# %%
def read_datasets (input_path : str) -> pd.DataFrame:
    """
    Read a csv file using pandas
    
    :param input_path: A string indicating the csv file Path
    :type input_path: str
    :return: A dataframe
    :rtype: DataFrame
    """
    return pd.read_csv (input_path)

def slice_load_profile (df : pd.DataFrame) -> pd.DataFrame:
    """
    Take a dataframe and slice it in a two-hour window
    
    :param df: a one-week water heater load profile
    :type df: pd.DataFrame
    :return: a sliced dataframe (two-hour window)
    :rtype: DataFrame
    """
    return df.head (n = 121)

def initialize_water_heater_states (df : pd.DataFrame) -> pd.DataFrame:
    """
    Each water heater at a minute t is:
        math::x(t) in {0, 4.5~kW}
    Convert to state s(t) in {0,1} where:
        s(t) = 1 if ON
        s(t) = 0 if OFF
    
    :param df: a sliced dataframe (two-hour window) of a water heater load profile
    :type df: pd.DataFrame
    :return: a dataframe with states column {0,1}
    :rtype: DataFrame
    tn_meter_4br_46:measured_real_power
    """
    df['state'] = (df['tn_meter_4br_46:measured_real_power'] ==4500).astype(int)
    return df
# %%

def experiment_one (df_original : pd.DataFrame):
    """
    The "Unknown" we need to learn:
        - Define the unknown quantity:
            p = probability this water heater is ON at a random minute (within a chosen window = 2 hrs)
    Goal:
        We are learning the probability from our binary observations (states)
    
    Step A: Prior -> The initial belief:
        p ~ Beta (alpha, beta):
            
            - The Beta distribution answers the question how do I represent the
            uncertainty about a probability?

            - alpha & beta summarize our belief:
                - alpha: how much evidence for a success?
                - beta: how much evidence for a failure?
                - success = heads, failure = tails
                - alpha -1 successes and beta-1 failures
            - Example:
                - Beta (1,1): alpha-1 = 0, beta-1 = 0 --> A flat belief
                - Beta (20,20): alpha -1 = 19, beta-1 = 19 --> A strong belief that coin is fair
                    - The distribution is centered at 0.5
                - Beta (8,2): alpha-1 = 7, beta-1=1 --> Heads is more likely
                    - 7 heads and 1 tail, centered at 0.8, heads is more likely
    """
    # Prep the data
    # H: number of ON minutes, T: the number of OFF minutes

    df = df_original.head(60)
    H = (df['state'] == 1).sum()
    T = (df['state'] == 0).sum()
    print(f"H={H}, T={T}, H+T={H+T}, df len={len(df)}")
    
    # Step A: This is my prior belief --> Belief before I looked at the data (manual)
    alpha = 1 # water heater is ON
    beta = 1 # water heater is OFF

    # Step 3: Compute the posterior
    # H ~ Bionomial(n,p) can read as the random variable H is generated according to a bionomial ...
    print('The posterior is:')
    print('p | data ~ Beta(alpha+H, beta+T)')
    print(f'So p | data ~ Beta(1+5, 8+55) \nP | data ~ Beta ({alpha+H}, {beta+T})')

    # Step 4: Compute one summary from the posterior (the posterior mean)
    # Weighted average: E[p] = (alpha) / (alpha + beta)
    # This is not the truth, this a summary of my belief
    # Applying the weighted avg to the posterior: E[p|data] = (alpha) / (alpha + beta)
    E = round((alpha+H) /((alpha+H) + (beta+T)), 3)
    print(f'The expected value E = {E}')

    print(f'''
    Step 5: Use the posterior to make a prediction
    Now we have P | data ~ Beta ({alpha+H}, {beta+T})
    Let's predict the next 60 minutes:
        k = how many minutes will the water heater be on?
        k is a count between 0 to 60
    
    The bottom line is: let's reflect the uncertainty in P | data ~ Beta ({alpha+H}, {beta+T})
    We'll use E[k | data] = m . E[p | data], but we already calculated E = round((alpha+H) /((alpha+H) + (beta+T)), 3)
    m = is the next 60 minutes
    ''')
    m = 60
    E_expected = E * m
    print(E_expected)

    # Step 6::
    


# %%
if __name__ == '__main__':

    # 
    root = Path (__file__).resolve ().parents[3]
    
    dataset_dir = root / 'cosimulation_tools' / 'dss-ochre-helics' / 'profiles' / 'one_week_wh_data'

     # Get the dataset input files:
    input_files = [file for file in dataset_dir.iterdir()]

    # Read the input files
    df = read_datasets (input_path=input_files[0])

    print(df.head(20))

    quit()

    # Slice the input files to a two-hour window
    df = slice_load_profile (df=df)

    # initialize the state of the given profile
    df = initialize_water_heater_states (df=df)

    # Experiment 1
    experiment_one (df_original=df)
    