'''
Author: Midrar Adham
Created: Thu Apr 23 2026
'''
'''
For new PEG users, usage [wh dir example]:
- import the script.
- loader = DataLoader (results=wh_dir)
- loader = load_csv_files (threshold=5000.0)

'''
import os
import pandas as pd

class DataLoader:

    def __init__(self, results_dir : str):
        
        self.all_dfs = {}
        self.results_dir = results_dir
    
    def _collect_files_from_directories (self, files_dir : str) -> list:
        '''
        append each csv file to a list and returns the path/filenames
        '''
        return [f'{files_dir}{fname}' for fname in os.listdir (files_dir) if 'ochre' in fname]

    def _clean_dataframe (self, filename : str):
        df = pd.read_csv (filename, header=0, names=['time', 'power_out'] ,skiprows=8)
        df = df.iloc[1440:2880]
        df.loc[:, 'time'] = df['time'].apply (lambda x: x.strip ('PST'))
        df.loc[:, 'time'] = pd.to_datetime (df['time'])
        df.loc[:, 'power_out'] = df['power_out'].apply (lambda x: complex (x))
        df.loc[:, 'power_out'] = df['power_out'].apply(lambda x: x.real)

        return df
    
    def _create_binary_states (self, df : pd.DataFrame, threshold : float):
        df = df.copy ()
        df['state'] = (df[df.columns[1]] > threshold).astype('bool')

        return df


    
    def load_csv_files (self,threshold : float):

        cosim_files = self._collect_files_from_directories (files_dir=self.results_dir)

        for filename in cosim_files:
            
            df = self._clean_dataframe (filename=filename)
            df = self._create_binary_states (df=df, threshold=threshold)

            self.all_dfs[filename] = df