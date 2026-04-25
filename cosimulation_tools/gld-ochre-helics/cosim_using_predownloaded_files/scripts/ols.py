'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import numpy as np
import pandas as pd

class OrdinaryLeastSquare:
    
    def __init__(self,
             wh_histories: dict,
             hvac_histories: dict,
             wh_all_dfs: dict,
             hvac_all_dfs: dict,
             feeder_demand: pd.DataFrame
             ):
        """
        Store everything OLS needs to run its analysis.

        Parameters
        ----------
        wh_histories : dict
            The output of BayesianEstimator.fit_many() run on WH data.
            Keys are filenames, values are the history dicts containing
            'mean', 'ci_lower', 'ci_upper', etc. for each chunk.

        hvac_histories : dict
            Same as wh_histories but for HVAC data.

        wh_all_dfs : dict
            The output of DataLoader.all_dfs for WH data.
            Needed to build the WH state matrix.

        hvac_all_dfs : dict
            Same as wh_all_dfs but for HVAC data.
            Needed to build the HVAC state matrix.

        feeder_demand : pd.DataFrame
            The full house transformer data from DataLoader.load_transformer_data().
            This is the signal we are trying to decompose.
        """
        self.wh_histories   = wh_histories
        self.hvac_histories = hvac_histories
        self.wh_all_dfs     = wh_all_dfs
        self.hvac_all_dfs   = hvac_all_dfs
        self.feeder_demand  = feeder_demand

    def _build_state_matrix (self, all_dfs):
        '''
        returns the state matrix

        Parameters
        ----------
        all_dfs : dict
            The output of DataLoader.all_dfs for WH or HVAC data.
            Needed to build the WH or HVAC state matrix.
        '''
        state_matrix = {}
        for filename, df in all_dfs.items ():
            df['time'] = pd.to_datetime (df['time'])
            states = df.set_index('time')['state']
            states.index = pd.to_datetime(states.index)
            states = states.resample('10min').mean()  # if any minute in chunk is ON, chunk is ON
            state_matrix[filename] = states.values
        
        return pd.DataFrame(state_matrix)

    def _estimate_background_demand (self, state_matrix : pd.DataFrame):
        '''
        returns the background feeder information, that is background = original feeder signal - feeder signal where all wh=0

        Parameters 
        ----------
        full house feeder signal : pd.DataFrame
            The feeder demand of all houses simulated.
        
        state matrix  : dict
            The states of each DER formatted in a matrix.
        '''
        der_off_chunks = (state_matrix.sum (axis=1) == 0)
        der_off_indices = np.where(der_off_chunks.values)[0]
        
        background_demand = np.interp (
            np.arange (144),
            der_off_indices,
            self.feeder_demand['power_out'].values [der_off_indices]
            )
        
        estimated_combined_demand = np.clip (
            self.feeder_demand['power_out'].values - background_demand, 0, None)
        
        return estimated_combined_demand
    
    def _build_mean_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the mean values from each DER's Bayesian history and
        organize them into a single matrix.

        Each column represents one DER, each row represents one 10-minute
        chunk. The values are the posterior mean ON-probabilities estimated
        by the Bayesian updater.

        Parameters
        ----------
        histories : dict
            The output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'mean' list of
            144 values.

        Returns
        -------
        pd.DataFrame
            A (144 x n_ders) DataFrame where each column is one DER's
            mean ON-probability sequence across all chunks.
        """
        mean_matrix = {}

        for filename, history in histories.items():
            mean_matrix[filename] = history['mean']

        return pd.DataFrame(mean_matrix)
    
    def _run_simultaneous_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            wh_state_matrix : pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            ) -> dict:
        """
        Run OLS simultaneously for WH and HVAC to estimate the kW scaling
        factor for each DER type.

        The idea is to find the values kw_wh and kw_hvac such that:

            kw_wh * x_wh + kw_hvac * x_hvac ≈ feeder_demand - background

        where x_wh and x_hvac are the total expected number of ON units
        at each chunk, summed across all DERs of that type.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
            Shape: (144 x n_wh_ders).

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
            Shape: (144 x n_hvac_ders).

        Returns
        -------
        dict
            Contains:
            - 'kw_wh'        : float, estimated watts per WH unit
            - 'kw_hvac'      : float, estimated watts per HVAC unit
            - 'wh_predicted' : np.ndarray, predicted WH demand across 144 chunks
            - 'x_wh'         : np.ndarray, total WH ON-probability per chunk
            - 'x_hvac'       : np.ndarray, total HVAC ON-probability per chunk
        """
        x_wh = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        # Estimate and subtract the background demand
        background_constant = self.feeder_demand['power_out'].values[
            np.where((pd.DataFrame(wh_state_matrix).sum(axis=1) == 0).values)[0]
        ].mean()

        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # Stack x_wh and x_hvac into matrix A and solve A * [kw_wh, kw_hvac] = feeder_minus_background
        A = np.column_stack([x_wh, x_hvac])
        kw_estimate, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)
        kw_wh, kw_hvac = kw_estimate

        # Scale each vector by its estimated kW factor
        wh_predicted = kw_wh * x_wh

        return {
            'kw_wh':        kw_wh,
            'kw_hvac':      kw_hvac,
            'wh_predicted': wh_predicted,
            'x_wh':         x_wh,
            'x_hvac':       x_hvac,
        }

    def _run_sequential_ols(self, x_hvac: np.ndarray, wh_predicted: np.ndarray) -> dict:
        """
        Run a second OLS pass using only the HVAC vector to refine the
        HVAC demand estimate.

        After Stage 1 (simultaneous OLS), we subtract the WH prediction
        from the raw feeder signal. What remains should mostly contain
        HVAC demand plus a background constant. We then solve:

            kw_hvac * x_hvac + baseline ≈ feeder_demand - wh_predicted

        The reason we run OLS a second time is that the simultaneous OLS
        in Stage 1 can mix up the WH and HVAC contributions. By isolating
        HVAC in its own regression, we get a cleaner estimate.

        Parameters
        ----------
        x_hvac : np.ndarray
            Total HVAC ON-probability per chunk. Comes directly from
            the 'x_hvac' key in the output of _run_simultaneous_ols().

        wh_predicted : np.ndarray
            Predicted WH demand across 144 chunks. Comes directly from
            the 'wh_predicted' key in the output of _run_simultaneous_ols().

        Returns
        -------
        dict
            Contains:
            - 'kw_hvac'          : float, refined estimated watts per HVAC unit
            - 'baseline'         : float, refined background constant
            - 'hvac_predicted'   : np.ndarray, predicted HVAC demand across 144 chunks
        """
        # Subtract WH prediction from raw feeder to isolate HVAC + background
        y_hvac = self.feeder_demand['power_out'].values - wh_predicted

        # Build A matrix with HVAC vector and a constant background column
        A_hvac = np.column_stack([x_hvac, np.ones(len(x_hvac))])

        # Solve for kw_hvac and the refined baseline constant
        hvac_estimate, _, _, _ = np.linalg.lstsq(A_hvac, y_hvac, rcond=None)
        kw_hvac, baseline = hvac_estimate

        hvac_predicted = kw_hvac * x_hvac

        return {
            'kw_hvac':        kw_hvac,
            'baseline':       baseline,
            'hvac_predicted': hvac_predicted,
        }
    

    def run (self):
        # Step 1 - build state matrices
        wh_state_matrix   = self._build_state_matrix(all_dfs=self.wh_all_dfs)

        # Step 2 - build mean matrices
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # Step 3 - simultaneous OLS
        simultaneous_results = self._run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            wh_state_matrix=wh_state_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        # Step 4 - sequential OLS to refine HVAC estimate
        sequential_results = self._run_sequential_ols(
            x_hvac=simultaneous_results['x_hvac'],
            wh_predicted=simultaneous_results['wh_predicted']
        )

        # Step 5 - return everything the caller needs
        return {
            'wh_predicted':   simultaneous_results['wh_predicted'],
            'hvac_predicted': sequential_results['hvac_predicted'],
            'kw_wh':          simultaneous_results['kw_wh'],
            'kw_hvac':        sequential_results['kw_hvac'],
            'baseline':       sequential_results['baseline'],
        }