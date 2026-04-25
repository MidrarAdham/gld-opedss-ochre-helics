'''
Author: Midrar Adham
Created: Thu Apr 23 2026
'''
'''
For new PEG users, usage [wh dir example]:
- import the class --> from data_loader import DataLoader
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


    def load_transformer_data (self):
        xfmr = f'{self.results_dir}residential_transformer.csv'
        df = pd.read_csv (xfmr, skiprows=8, usecols=['# timestamp', 'power_out'])
        print("\n\ndon't forget you're using the second day of the data\n\n")
        df = df.iloc [1440:2880]
        df.loc[:, '# timestamp'] = df['# timestamp'].apply (lambda x: x.strip ('PST'))
        df.loc[:, '# timestamp'] = pd.to_datetime (df['# timestamp'])
        df.loc[:, 'power_out'] = df['power_out'].apply (lambda x: complex (x))
        df.loc[:, 'power_out'] = df['power_out'].apply(lambda x: x.real)
        df['# timestamp'] = pd.to_datetime(df['# timestamp'], errors='coerce')
        df = df.set_index ('# timestamp')
        df = df.resample ("10min").mean()
        df = df.reset_index ()
        df['power_out'] = pd.to_numeric(df['power_out'], errors='coerce')
        df = df.rename (columns={'# timestamp':'Time'})

        return df
    
    def load_csv_files (self,threshold : float):
        """
        Returns the 
        """

        cosim_files = self._collect_files_from_directories (files_dir=self.results_dir)

        for filename in cosim_files:
            
            df = self._clean_dataframe (filename=filename)
            df = self._create_binary_states (df=df, threshold=threshold)

            self.all_dfs[filename] = df
        
        return self.all_dfs
    