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
    
    def _build_combined_mean_matrix(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        estimated_background: np.ndarray
        ) -> pd.DataFrame:
        """
        Combine WH and HVAC mean matrices into one, normalize each DER's
        contribution by the total across all DERs, then scale by the
        background-subtracted feeder demand.

        This transforms raw ON-probabilities into demand-weighted values
        (in Watts), so each value represents that DER's estimated share
        of the feeder signal at that chunk.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
            Shape: (144 x n_wh_ders).

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
            Shape: (144 x n_hvac_ders).

        estimated_background : np.ndarray
            Output of _estimate_background_demand().
            The feeder signal with background demand subtracted.

        Returns
        -------
        pd.DataFrame
            A (144 x n_total_ders) DataFrame where each value is that
            DER's estimated Watts contribution at that chunk.
        """
        # Step 1 - combine WH and HVAC into one matrix
        combined = pd.concat([wh_mean_matrix, hvac_mean_matrix], axis=1)

        # Step 2 - compute total ON-probability across all DERs at each chunk
        total = combined.sum(axis=1)

        # Step 3 - divide each DER by the total to get its fractional share
        normalized = combined.div(total, axis=0)

        # Step 4 - scale by background-subtracted feeder demand to convert to Watts
        combined_mean_matrix = normalized.multiply(estimated_background, axis=0)

        return combined_mean_matrix


    def _run_simultaneous_ols(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        ) -> dict:
        """
        Run standard OLS simultaneously for WH and HVAC to estimate
        the kW scaling factor for each DER type.

        Solves: kw_wh * x_wh + kw_hvac * x_hvac ≈ feeder - background

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.

        wh_state_matrix : pd.DataFrame
            Output of _build_state_matrix() for WH data.
            Used to estimate the background constant.

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.

        Returns
        -------
        dict
            Contains kw_wh, kw_hvac, wh_predicted, x_wh, x_hvac,
            feeder_minus_background.
        """
        x_wh   = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        A = np.column_stack([x_wh, x_hvac])
        kw_estimate, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)
        kw_wh, kw_hvac = kw_estimate

        wh_predicted = kw_wh * x_wh

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac':                 kw_hvac,
            'wh_predicted':            wh_predicted,
            'x_wh':                    x_wh,
            'x_hvac':                  x_hvac,
            'feeder_minus_background': feeder_minus_background,
        }
    
    def _run_sequential_ols(
        self,
        x_hvac: np.ndarray,
        wh_predicted: np.ndarray,
        feeder_minus_background: np.ndarray,
        ) -> dict:
        """
        Run a second OLS pass using only the HVAC vector to refine
        the HVAC demand estimate.

        Solves: kw_hvac * x_hvac + baseline ≈ feeder_minus_background - wh_predicted

        Parameters
        ----------
        x_hvac : np.ndarray
            Total HVAC ON-probability per chunk.

        wh_predicted : np.ndarray
            Predicted WH demand across 144 chunks.

        feeder_minus_background : np.ndarray
            Feeder signal with background removed.

        Returns
        -------
        dict
            Contains kw_hvac, baseline, hvac_predicted.
        """
        y_hvac = feeder_minus_background - wh_predicted

        A_hvac = np.column_stack([x_hvac, np.ones(len(x_hvac))])
        hvac_estimate, _, _, _ = np.linalg.lstsq(A_hvac, y_hvac, rcond=None)
        kw_hvac, baseline = hvac_estimate

        hvac_predicted = kw_hvac * x_hvac

        return {
            'kw_hvac':        kw_hvac,
            'baseline':       baseline,
            'hvac_predicted': hvac_predicted,
        }
    
    def _compute_deltas(
        self,
        x_wh: np.ndarray,
        x_hvac: np.ndarray,
        feeder_minus_background: np.ndarray,
        ) -> dict:
        """
        Compute the first-order differences (deltas) of the WH signal,
        HVAC signal, and feeder signal.

        Instead of regressing on raw values, we regress on how much each
        signal CHANGES between consecutive chunks. This makes WH and HVAC
        more separable because:
        - WH units switch abruptly → large spikes in Δx_wh
        - HVAC units change gradually → smaller changes in Δx_hvac

        Parameters
        ----------
        x_wh : np.ndarray
            Raw WH probability sum across 144 chunks.

        x_hvac : np.ndarray
            Raw HVAC probability sum across 144 chunks.

        feeder_minus_background : np.ndarray
            Feeder signal with background removed across 144 chunks.

        Returns
        -------
        dict
            Contains:
            - 'delta_x_wh'   : np.ndarray, shape (143,), change in WH signal
            - 'delta_x_hvac' : np.ndarray, shape (143,), change in HVAC signal
            - 'delta_y'      : np.ndarray, shape (143,), change in feeder signal
        """
        # np.diff computes x[t] - x[t-1] for each consecutive pair
        # result has 143 values instead of 144 — first value is lost
        delta_x_wh   = np.diff(x_wh)
        delta_x_hvac = np.diff(x_hvac)
        delta_y      = np.diff(feeder_minus_background)

        return {
            'delta_x_wh':   delta_x_wh,
            'delta_x_hvac': delta_x_hvac,
            'delta_y':      delta_y,
        }

    def _run_delta_ols(
            self,
            x_wh: np.ndarray,
            x_hvac: np.ndarray,
            feeder_minus_background: np.ndarray,
            ) -> dict:

            """
            Run OLS on the delta (differenced) signals to estimate kW scaling
            factors that are more robust to multicollinearity.

            By regressing on changes rather than levels, we exploit the fact
            that WH and HVAC have different switching dynamics — WH switches
            abruptly while HVAC changes gradually. These different dynamics
            make the two signals easier to separate in delta space.

            Parameters
            ----------
            x_wh : np.ndarray
                Raw WH probability sum across 144 chunks.

            x_hvac : np.ndarray
                Raw HVAC probability sum across 144 chunks.

            feeder_minus_background : np.ndarray
                Feeder signal with background removed across 144 chunks.

            Returns
            -------
            dict
                Contains:
                - 'kw_wh'          : float, estimated watts per WH unit
                - 'kw_hvac'        : float, estimated watts per HVAC unit
                - 'wh_predicted'   : np.ndarray, predicted WH demand (144 chunks)
                - 'hvac_predicted' : np.ndarray, predicted HVAC demand (144 chunks)
            """
            # Step 1 - compute deltas
            deltas = self._compute_deltas(
                x_wh=x_wh,
                x_hvac=x_hvac,
                feeder_minus_background=feeder_minus_background,
            )

            # Step 2 - solve OLS in delta space
            # Note: no intercept here — delta of a constant is zero
            A_delta = np.column_stack([deltas['delta_x_wh'], deltas['delta_x_hvac']])
            kw_estimate, _, _, _ = np.linalg.lstsq(A_delta, deltas['delta_y'], rcond=None)
            kw_wh, kw_hvac = kw_estimate

            # Step 3 - apply scaling factors to original (non-differenced) signals
            # The kW factors found in delta space apply back to the level signals
            wh_predicted   = kw_wh   * x_wh
            hvac_predicted = kw_hvac * x_hvac

            return {
                'kw_wh':          kw_wh,
                'kw_hvac':        kw_hvac,
                'wh_predicted':   wh_predicted,
                'hvac_predicted': hvac_predicted,
            }

    def run(self):
        # Step 1 - build WH state matrix
        wh_state_matrix = self._build_state_matrix(all_dfs=self.wh_all_dfs)
        hvac_state_matrix = self._build_state_matrix(all_dfs=self.hvac_all_dfs)

        # Step 2 - build mean matrices
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # Step 3 - simultaneous OLS
        simultaneous_results = self._run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        # Step 4 - sequential OLS to refine HVAC estimate
        sequential_results = self._run_sequential_ols(
            x_hvac=simultaneous_results['x_hvac'],
            wh_predicted=simultaneous_results['wh_predicted'],
            feeder_minus_background=simultaneous_results['feeder_minus_background'],
        )

        # Step 5 - delta OLS for comparison
        delta_results = self._run_delta_ols(
            x_wh=simultaneous_results['x_wh'],
            x_hvac=simultaneous_results['x_hvac'],
            feeder_minus_background=simultaneous_results['feeder_minus_background'],
        )

        # Step 6 - return results
        return {
            'wh_predicted':            simultaneous_results['wh_predicted'],
            'x_wh'        :            simultaneous_results['x_wh'],
            'x_hvac'        :          simultaneous_results['x_hvac'],
            'hvac_predicted':          sequential_results['hvac_predicted'],
            'kw_wh':                   simultaneous_results['kw_wh'],
            'kw_hvac':                 sequential_results['kw_hvac'],
            'baseline':                sequential_results['baseline'],
            'feeder_minus_background': simultaneous_results['feeder_minus_background'],
            'combined_predicted':      simultaneous_results['wh_predicted'] + sequential_results['hvac_predicted'],
            # Delta OLS results — for comparison against standard OLS
            'delta_wh_predicted':      delta_results['wh_predicted'],
            'delta_hvac_predicted':    delta_results['hvac_predicted'],
            'delta_kw_wh':             delta_results['kw_wh'],
            'delta_kw_hvac':           delta_results['kw_hvac'],
            # General info for plotting purposes
            'wh_state_matrix':         wh_state_matrix,
            'hvac_state_matrix':       hvac_state_matrix,
            'wh_mean_matrix':          wh_mean_matrix,
            'hvac_mean_matrix':        hvac_mean_matrix,
        }

        # # Step 6 - return results
        # return {
        #     'wh_predicted':            simultaneous_results['wh_predicted'],
        #     'hvac_predicted':          sequential_results['hvac_predicted'],
        #     'kw_wh':                   simultaneous_results['kw_wh'],
        #     'kw_hvac':                 sequential_results['kw_hvac'],
        #     'baseline':                sequential_results['baseline'],
        #     'feeder_minus_background': simultaneous_results['feeder_minus_background'],
        #     'combined_predicted':      simultaneous_results['wh_predicted'] + sequential_results['hvac_predicted'],
        # }