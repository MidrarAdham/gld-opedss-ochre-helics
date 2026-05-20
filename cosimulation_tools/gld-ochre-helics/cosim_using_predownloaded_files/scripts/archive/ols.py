'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

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
        # combine WH and HVAC into one matrix
        combined = pd.concat([wh_mean_matrix, hvac_mean_matrix], axis=1)

        # compute total ON-probability across all DERs at each chunk
        total = combined.sum(axis=1)

        # divide each DER by the total to get its fractional share
        normalized = combined.div(total, axis=0)

        # scale by background-subtracted feeder demand to convert to Watts
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
        hvac_predicted = kw_hvac * x_hvac

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac':                 kw_hvac,
            'wh_predicted':            wh_predicted,
            'hvac_predicted':          hvac_predicted,
            'x_wh':                    x_wh,
            'x_hvac':                  x_hvac,
            'feeder_minus_background': feeder_minus_background,
        }
    
    def _run_simultaneous_ols_multiple_regressors (
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        ) -> dict:
        """
        Run standard OLS simultaneously for WH and HVAC to estimate
        the kW scaling factor for each DER type.

        Solves: kw_wh * x_wh + kw_hvac1 * x_hvac1 + kw_hvac2 * x_hvac2 +kw_hvac3 * x_hvac3 + kw_hvac4 * x_hvac4 + kw_hvac_rest * x_hvac_rest≈ feeder - background

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
        
        # Split HVAC mean matrix into individual columns
        # ── Rank HVAC devices by activity (mean ON-probability across all chunks) ──
        activity_scores = hvac_mean_matrix.mean(axis=0).sort_values(ascending=False)
        ranked_cols     = activity_scores.index.tolist()

        # ── Keep top 4 most active as separate regressors, sum the rest ────────────
        top_cols  = ranked_cols[:4]
        rest_cols = ranked_cols[4:]

        x_hvac_1    = hvac_mean_matrix[top_cols[0]].values
        x_hvac_2    = hvac_mean_matrix[top_cols[1]].values
        x_hvac_3    = hvac_mean_matrix[top_cols[2]].values
        x_hvac_4    = hvac_mean_matrix[top_cols[3]].values
        x_hvac_rest = hvac_mean_matrix[rest_cols].sum(axis=1).values

        # ── Design matrix ─────────────────────────────────────────────────────────
        background_constant      = self.feeder_demand['power_out'].values.min()
        feeder_minus_background  = self.feeder_demand['power_out'].values - background_constant
        A = np.column_stack([x_wh, x_hvac_1, x_hvac_2, x_hvac_3, x_hvac_4, x_hvac_rest])

        kw_estimate, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)
        kw_wh, kw_hvac_1, kw_hvac_2, kw_hvac_3, kw_hvac_4, kw_hvac_rest = kw_estimate

        wh_predicted = kw_wh * x_wh

        hvac_predicted = (kw_hvac_1 * x_hvac_1 + 
                          kw_hvac_2 * x_hvac_2 + 
                          kw_hvac_3 * x_hvac_3 + 
                          kw_hvac_4 * x_hvac_4 + 
                          kw_hvac_rest * x_hvac_rest
                          )

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac_1':               kw_hvac_1,
            'kw_hvac_2':               kw_hvac_2,
            'kw_hvac_3':               kw_hvac_3,
            'kw_hvac_4':               kw_hvac_4,
            'kw_hvac_rest':            kw_hvac_rest,
            'wh_predicted':            wh_predicted,
            'hvac_predicted':          hvac_predicted,
            'combined_predicted':      wh_predicted + hvac_predicted,
            'x_wh':                    x_wh,
            'x_hvac_1':                x_hvac_1,
            'x_hvac_2':                x_hvac_2,
            'x_hvac_3':                x_hvac_3,
            'x_hvac_4':                x_hvac_4,
            'x_hvac_rest':             x_hvac_rest,
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

    def _run_hvac_frequency_split_after_wh(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        window: int = 6
        ) -> dict:

        x_wh = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        # Keep your existing delta OLS behavior
        delta_results = self._run_delta_ols(
            x_wh=x_wh,
            x_hvac=x_hvac,
            feeder_minus_background=feeder_minus_background
        )

        wh_predicted = delta_results["wh_predicted"]

        # Remaining signal after WH removal
        y_remaining = feeder_minus_background - wh_predicted

        # Low-frequency HVAC target
        y_remaining_low = (
            pd.Series(y_remaining)
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .values
        )

        # HVAC regression against smoothed remaining target
        A_hvac = np.column_stack([
            x_hvac,
            np.ones(len(x_hvac))
        ])

        coef, _, _, _ = np.linalg.lstsq(
            A_hvac,
            y_remaining_low,
            rcond=None
        )

        kw_hvac, baseline = coef

        hvac_predicted = kw_hvac * x_hvac + baseline

        return {
            "kw_wh": delta_results["kw_wh"],
            "kw_hvac": kw_hvac,
            "hvac_baseline": baseline,

            "wh_predicted": wh_predicted,
            "hvac_predicted": hvac_predicted,

            "x_wh": x_wh,
            "x_hvac": x_hvac,

            "feeder_minus_background": feeder_minus_background,
            "y_remaining": y_remaining,
            "y_remaining_low": y_remaining_low,
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
            # compute deltas
            deltas = self._compute_deltas(
                x_wh=x_wh,
                x_hvac=x_hvac,
                feeder_minus_background=feeder_minus_background,
            )

            # solve OLS in delta space
            # Note: no intercept here — delta of a constant is zero
            A_delta = np.column_stack([deltas['delta_x_wh'], deltas['delta_x_hvac']])
            kw_estimate, _, _, _ = np.linalg.lstsq(A_delta, deltas['delta_y'], rcond=None)
            kw_wh, kw_hvac = kw_estimate

            # apply scaling factors to original (non-differenced) signals
            # The kW factors found in delta space apply back to the level signals
            wh_predicted   = kw_wh   * x_wh
            hvac_predicted = kw_hvac * x_hvac

            return {
                'kw_wh':          kw_wh,
                'kw_hvac':        kw_hvac,
                'wh_predicted':   wh_predicted,
                'hvac_predicted': hvac_predicted,
            }
    
    

    def _run_ridge_ols(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        alpha: float = 1.0
        ) -> dict:

        # ── Rank HVAC devices by activity ────────────────────────────
        activity_scores = hvac_mean_matrix.mean(axis=0).sort_values(ascending=False)
        ranked_cols     = activity_scores.index.tolist()
        top_cols        = ranked_cols[:4]
        rest_cols       = ranked_cols[4:]

        x_wh        = wh_mean_matrix.sum(axis=1).values
        x_hvac_1    = hvac_mean_matrix[top_cols[0]].values
        x_hvac_2    = hvac_mean_matrix[top_cols[1]].values
        x_hvac_3    = hvac_mean_matrix[top_cols[2]].values
        x_hvac_4    = hvac_mean_matrix[top_cols[3]].values
        x_hvac_rest = hvac_mean_matrix[rest_cols].sum(axis=1).values

        # ── Background subtraction ────────────────────────────────────
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # ── Design matrix ─────────────────────────────────────────────
        A = np.column_stack([x_wh, x_hvac_1, x_hvac_2,
                            x_hvac_3, x_hvac_4, x_hvac_rest])

        # ── Normalize columns to unit variance ────────────────────────
        # This ensures Ridge penalizes all regressors equally regardless
        # of their scale, and makes coefficients more comparable.
        col_stds = A.std(axis=0)
        col_stds[col_stds == 0] = 1.0  # avoid division by zero
        A_normalized = A / col_stds

        # ── Ridge regression on normalized design matrix ──────────────
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(A_normalized, feeder_minus_background)

        print(f"\nalpha={alpha}")
        print(f"kw_wh:       {ridge.coef_[0]:.1f} W")
        print(f"kw_hvac_1:   {ridge.coef_[1]:.1f} W")
        print(f"kw_hvac_2:   {ridge.coef_[2]:.1f} W")
        print(f"kw_hvac_3:   {ridge.coef_[3]:.1f} W")
        print(f"kw_hvac_4:   {ridge.coef_[4]:.1f} W")
        print(f"kw_hvac_rest:{ridge.coef_[5]:.1f} W")
        # quit()

        # ── Rescale coefficients back to original units ───────────────
        # coef_normalized = kW / std → kW = coef_normalized / std
        coef_rescaled = ridge.coef_ / col_stds
        kw_wh, kw_hvac_1, kw_hvac_2, kw_hvac_3, kw_hvac_4, kw_hvac_rest = coef_rescaled

        # ── Predictions ───────────────────────────────────────────────
        wh_predicted   = kw_wh * x_wh
        hvac_predicted = (kw_hvac_1    * x_hvac_1 +
                        kw_hvac_2    * x_hvac_2 +
                        kw_hvac_3    * x_hvac_3 +
                        kw_hvac_4    * x_hvac_4 +
                        kw_hvac_rest * x_hvac_rest)

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac_1':               kw_hvac_1,
            'kw_hvac_2':               kw_hvac_2,
            'kw_hvac_3':               kw_hvac_3,
            'kw_hvac_4':               kw_hvac_4,
            'kw_hvac_rest':            kw_hvac_rest,
            'wh_predicted':            wh_predicted,
            'hvac_predicted':          hvac_predicted,
            'combined_predicted':      wh_predicted + hvac_predicted,
            'x_wh':                    x_wh,
            'x_hvac_1':                x_hvac_1,
            'x_hvac_2':                x_hvac_2,
            'x_hvac_3':                x_hvac_3,
            'x_hvac_4':                x_hvac_4,
            'x_hvac_rest':             x_hvac_rest,
            'feeder_minus_background': feeder_minus_background,
            'col_stds':                col_stds,  # useful for debugging
        }
    
    def run_n_hvac_experiment(
        self,
        n_values: list = [4, 8, 12, 16, 20, 24, 28],
        wh_ground_truth: np.ndarray = None,
        hvac_ground_truth: np.ndarray = None
        ) -> dict:
        """
        Run delta OLS for increasing numbers of HVAC devices and
        record WH and HVAC R² at each N to find the optimal device count.

        Parameters
        ----------
        n_values : list
            List of HVAC device counts to test.
        wh_ground_truth : np.ndarray
            Ground truth WH demand in Watts.
        hvac_ground_truth : np.ndarray
            Ground truth HVAC demand in Watts.

        Returns
        -------
        dict
            Keys are N values, values are dicts with wh_r2 and hvac_r2.
        """
        # Build matrices once for all 28 devices
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # Rank HVAC columns by activity once
        activity_scores = hvac_mean_matrix.mean(axis=0).sort_values(ascending=False)
        ranked_cols     = activity_scores.index.tolist()

        # Background subtraction — same for all N
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # x_wh stays the same for all N
        x_wh = wh_mean_matrix.sum(axis=1).values

        results_by_n = {}

        for n in n_values:

            # Subset top N HVAC columns by activity
            top_n_cols = ranked_cols[:n]
            x_hvac     = hvac_mean_matrix[top_n_cols].sum(axis=1).values

            print(f"N={n} | x_hvac mean: {x_hvac.mean():.1f} | "
    f"hvac_truth mean: {pd.to_numeric(hvac_ground_truth, errors='coerce').mean():.1f}")
            # Delta signals
            delta_y     = np.diff(feeder_minus_background)
            delta_x_wh  = np.diff(x_wh)
            delta_x_hvac = np.diff(x_hvac)

            # Design matrix
            A_delta = np.column_stack([delta_x_wh, delta_x_hvac])
            kw_estimate, _, _, _ = np.linalg.lstsq(A_delta, delta_y, rcond=None)
            kw_wh, kw_hvac = kw_estimate

            # Predictions in level space
            wh_predicted   = kw_wh   * x_wh
            hvac_predicted = kw_hvac * x_hvac

            # R² calculations
            wh_truth   = pd.to_numeric(wh_ground_truth,   errors='coerce').values
            hvac_truth = pd.to_numeric(hvac_ground_truth, errors='coerce').values

            wh_r2   = 1 - np.sum((wh_predicted   - wh_truth)   ** 2) / \
                        np.sum((wh_truth   - wh_truth.mean())   ** 2)
            hvac_r2 = 1 - np.sum((hvac_predicted - hvac_truth) ** 2) / \
                        np.sum((hvac_truth - hvac_truth.mean()) ** 2)

            results_by_n[n] = {
                'wh_r2':   wh_r2,
                'hvac_r2': hvac_r2,
                'kw_wh':   kw_wh,
                'kw_hvac': kw_hvac,
            }

            print(f"N={n:2d} | WH R²: {wh_r2:.3f} | HVAC R²: {hvac_r2:.3f} | "
                f"kw_wh: {kw_wh:.0f} W | kw_hvac: {kw_hvac:.0f} W")

        return results_by_n
    
    def run_lag_sweep_experiment(
        self,
        lag_values: list = [0, 1, 2, 3, 5, 10],
        wh_ground_truth: np.ndarray = None,
        hvac_ground_truth: np.ndarray = None
        ) -> dict:
        """
        Run delta OLS with multi-step lagged HVAC features.

        Adds L lagged delta_x_hvac columns to capture HVAC thermal inertia:

            Δ̃Y(t) = kw_wh · Δx_wh(t)
                + Σ_{l=0}^{L} kw_hvac_l · Δx_hvac(t-l)

        Predictions are then reconstructed in the level domain by applying
        each lag coefficient to its corresponding lagged level signal:

            P̂_HVAC(t) = Σ_{l=0}^{L} kw_hvac_l · x_hvac(t-l)

        Parameters
        ----------
        lag_values : list
            List of L values to test (L=0 means no lags, just current chunk).
        wh_ground_truth : np.ndarray
            Ground truth WH demand in Watts.
        hvac_ground_truth : np.ndarray
            Ground truth HVAC demand in Watts.

        Returns
        -------
        dict
            Keys are L values, values are dicts with wh_r2 and hvac_r2.
        """
        # Build matrices once
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # Background subtraction
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # Aggregated regressors
        x_wh   = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        # Delta signals (length T-1 = 143)
        delta_y      = np.diff(feeder_minus_background)
        delta_x_wh   = np.diff(x_wh)
        delta_x_hvac = np.diff(x_hvac)

        wh_truth   = pd.to_numeric(wh_ground_truth,   errors='coerce').values
        hvac_truth = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        results_by_lag = {}

        for L in lag_values:
            # ── Fit regression in delta domain ───────────────────────
            valid_len = len(delta_y) - L

            # WH regressor — current chunk only
            regressors = [delta_x_wh[L:]]

            # HVAC regressors — current + L lags
            for l in range(L + 1):
                start = L - l
                end   = start + valid_len
                regressors.append(delta_x_hvac[start:end])

            A_delta = np.column_stack(regressors)
            y_trim  = delta_y[L:]

            kw_estimate, _, _, _ = np.linalg.lstsq(A_delta, y_trim, rcond=None)
            kw_wh        = kw_estimate[0]
            kw_hvac_lags = kw_estimate[1:]  # L+1 coefficients

            # ── Reconstruct predictions in level domain (FIXED) ──────
            # Each lag coefficient applied to its corresponding lagged level
            wh_predicted   = kw_wh * x_wh
            hvac_predicted = np.zeros_like(x_hvac)

            for l, kw_lag in enumerate(kw_hvac_lags):
                if l == 0:
                    hvac_predicted += kw_lag * x_hvac
                else:
                    hvac_predicted[l:] += kw_lag * x_hvac[:-l]

            # ── R² calculations ──────────────────────────────────────
            wh_r2   = 1 - np.sum((wh_predicted   - wh_truth)   ** 2) / \
                        np.sum((wh_truth   - wh_truth.mean())   ** 2)
            hvac_r2 = 1 - np.sum((hvac_predicted - hvac_truth) ** 2) / \
                        np.sum((hvac_truth - hvac_truth.mean()) ** 2)

            results_by_lag[L] = {
                'wh_r2':         wh_r2,
                'hvac_r2':       hvac_r2,
                'kw_wh':         kw_wh,
                'kw_hvac_lags':  kw_hvac_lags,
                'wh_predicted':  wh_predicted,
                'hvac_predicted': hvac_predicted,
            }

            lag_str = ', '.join([f'{k:.0f}' for k in kw_hvac_lags])
            print(f"L={L:2d} | WH R²: {wh_r2:.3f} | HVAC R²: {hvac_r2:.3f} | "
                f"kw_wh: {kw_wh:.0f} | kw_hvac_lags: [{lag_str}]")

        return results_by_lag
    
    
    
    def _build_variance_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the posterior variance values from each DER's Bayesian
        history and organize them into a matrix matching the structure
        of the mean matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'variance' list of
            144 values.

        Returns
        -------
        pd.DataFrame
            A (144 x n_ders) DataFrame where each column is one DER's
            posterior variance sequence across all chunks.
        """
        variance_matrix = {}
        for filename, history in histories.items():
            variance_matrix[filename] = history['variance']
        return pd.DataFrame(variance_matrix)


    def _run_variance_weighted_delta_ols(
            self,
            wh_mean_matrix:    pd.DataFrame,
            hvac_mean_matrix:  pd.DataFrame,
            hvac_variance_matrix: pd.DataFrame,
            epsilon: float = 1e-6
            ) -> dict:
        """
        Run delta OLS with precision-weighted HVAC regressor.

        Each HVAC device's mean contribution is weighted by its inverse
        posterior variance (precision). Devices we are more certain about
        contribute more to the aggregated regressor:

            x_hvac_weighted(t) = Σ_i [ M_i(t) / (V_i(t) + ε) ]

        The OLS keeps physical interpretability — kw_hvac is still in
        watts per (precision-weighted) expected ON device.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
        hvac_variance_matrix : pd.DataFrame
            Output of _build_variance_matrix() for HVAC histories.
        epsilon : float
            Small constant added to variance to avoid division by zero.

        Returns
        -------
        dict
            Contains kw_wh, kw_hvac, wh_predicted, hvac_predicted,
            and the regressors used.
        """
        # ── WH regressor — unchanged from standard delta OLS ─────────
        x_wh = wh_mean_matrix.sum(axis=1).values

        # ── HVAC precision-weighted regressor ────────────────────────
        # For each device: M_i(t) / (V_i(t) + ε)
        # Then sum across all devices at each chunk
        precisions   = 1.0 / (hvac_variance_matrix.values + epsilon)
        weighted_M   = hvac_mean_matrix.values * precisions
        x_hvac       = weighted_M.sum(axis=1)

        # ── Background subtraction ───────────────────────────────────
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # ── Delta signals ────────────────────────────────────────────
        delta_y      = np.diff(feeder_minus_background)
        delta_x_wh   = np.diff(x_wh)
        delta_x_hvac = np.diff(x_hvac)

        # ── Delta OLS ────────────────────────────────────────────────
        A_delta = np.column_stack([delta_x_wh, delta_x_hvac])
        kw_estimate, _, _, _ = np.linalg.lstsq(A_delta, delta_y, rcond=None)
        kw_wh, kw_hvac = kw_estimate

        # ── Predictions in level space ───────────────────────────────
        wh_predicted   = kw_wh   * x_wh
        hvac_predicted = kw_hvac * x_hvac

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac':                 kw_hvac,
            'wh_predicted':            wh_predicted,
            'hvac_predicted':          hvac_predicted,
            'combined_predicted':      wh_predicted + hvac_predicted,
            'x_wh':                    x_wh,
            'x_hvac':                  x_hvac,
            'feeder_minus_background': feeder_minus_background,
        }
    

    def _compute_on_duration_matrix(
        self,
        mean_matrix: pd.DataFrame,
        threshold: float = 0.5
        ) -> pd.DataFrame:

        """
        For each device, compute the number of consecutive chunks it has
        been ON (mean > threshold). A device that just switched ON has
        duration=1; one that's been ON for 5 chunks has duration=5.
        """
        binary       = (mean_matrix > threshold).astype(int).values
        on_duration  = np.zeros_like(binary, dtype=float)

        for i in range(binary.shape[1]):
            for t in range(binary.shape[0]):
                if binary[t, i] == 1:
                    on_duration[t, i] = on_duration[t-1, i] + 1 if t > 0 else 1
                else:
                    on_duration[t, i] = 0

        return pd.DataFrame(on_duration, columns=mean_matrix.columns,
                            index=mean_matrix.index)


    def _run_duration_augmented_delta_ols(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame
    ) -> dict:

        # Aggregated posterior regressors
        x_wh   = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        # Build actual HVAC state matrix
        hvac_state_matrix = self._build_state_matrix(self.hvac_all_dfs)

        # Compute ON-duration from actual quantized HVAC states
        on_duration_matrix = self._compute_on_duration_matrix(
            hvac_state_matrix,
            threshold=0.5
        )

        # Number of HVACs ON at each chunk
        s_hvac = (hvac_state_matrix > 0.5).astype(int).sum(axis=1).values

        # Sum and average ON-duration
        d_hvac_sum = on_duration_matrix.sum(axis=1).values
        d_hvac_avg = d_hvac_sum / (s_hvac + 1e-6)

        # Duration-augmented HVAC feature
        duration_feature = x_hvac * d_hvac_avg

        # Background subtraction
        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        # Deltas
        delta_y = np.diff(feeder_minus_background)
        delta_x_wh = np.diff(x_wh)
        delta_x_hvac = np.diff(x_hvac)
        delta_duration_feature = np.diff(duration_feature)

        # Delta OLS
        A_delta = np.column_stack([
            delta_x_wh,
            delta_x_hvac,
            delta_duration_feature
        ])

        coefs, _, _, _ = np.linalg.lstsq(A_delta, delta_y, rcond=None)
        kw_wh, kw_hvac, gamma = coefs

        # Reconstruct in level space
        wh_predicted = kw_wh * x_wh

        hvac_predicted = (
            kw_hvac * x_hvac
            + gamma * duration_feature
        )

        return {
            'kw_wh': kw_wh,
            'kw_hvac': kw_hvac,
            'gamma': gamma,

            'wh_predicted': wh_predicted,
            'hvac_predicted': hvac_predicted,
            'combined_predicted': wh_predicted + hvac_predicted,

            'x_wh': x_wh,
            'x_hvac': x_hvac,
            's_hvac': s_hvac,
            'd_hvac_sum': d_hvac_sum,
            'd_hvac_avg': d_hvac_avg,
            'duration_feature': duration_feature,

            'feeder_minus_background': feeder_minus_background,
        }

    def _run_temperature_interaction_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            ambient_temp: np.ndarray,
            t_base: float = 70.0,
            ) -> dict:

        x_wh = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        ambient_temp = np.asarray(ambient_temp, dtype=float)

        T_cool = np.maximum(0, ambient_temp - t_base)

        A = np.column_stack([
            x_wh,
            x_hvac,
            x_hvac * T_cool,
            np.ones(len(x_wh)),
        ])

        coef, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)

        kw_wh, kw_hvac_base, kw_hvac_temp, baseline = coef

        wh_predicted = kw_wh * x_wh

        hvac_predicted = (
            kw_hvac_base * x_hvac
            + kw_hvac_temp * x_hvac * T_cool
        )

        combined_predicted = wh_predicted + hvac_predicted + baseline

        return {
            "kw_wh": kw_wh,
            "kw_hvac_base": kw_hvac_base,
            "kw_hvac_temp": kw_hvac_temp,
            "baseline": baseline,
            "wh_predicted": wh_predicted,
            "hvac_predicted": hvac_predicted,
            "combined_predicted": combined_predicted,
            "x_wh": x_wh,
            "x_hvac": x_hvac,
            "T_cool": T_cool,
            "feeder_minus_background": feeder_minus_background,
        }
    
    def _compute_startup_count(
        self,
        state_matrix: pd.DataFrame,
        threshold: float = 0.5
        ) -> np.ndarray:
        """
        Count how many devices switch from OFF to ON at each chunk.

        startup_count[t] = number of HVAC units where:
            state[t-1] = 0 and state[t] = 1

        The first chunk is assigned 0 because there is no previous chunk.
        """
        binary = (state_matrix > threshold).astype(int).values

        startup_matrix = np.zeros_like(binary, dtype=float)

        # OFF -> ON transition
        startup_matrix[1:, :] = np.maximum(
            0,
            binary[1:, :] - binary[:-1, :]
        )

        startup_count = startup_matrix.sum(axis=1)

        return startup_count
    
    def _run_startup_augmented_delta_ols(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame
        ) -> dict:
        """
        Delta OLS with HVAC startup/event intensity as a third regressor:

            ΔY(t) = kw_wh Δx_wh(t)
                + kw_hvac Δx_hvac(t)
                + gamma Δu_hvac(t)

        where u_hvac(t) is the number of HVAC units that just switched ON.
        """

        # Aggregated posterior regressors
        x_wh = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        # Actual HVAC state matrix
        hvac_state_matrix = self._build_state_matrix(self.hvac_all_dfs)

        # Startup count: number of OFF -> ON transitions at each chunk
        u_hvac = self._compute_startup_count(
            state_matrix=hvac_state_matrix,
            threshold=0.5
        )

        # Background subtraction
        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        # Deltas
        delta_y = np.diff(feeder_minus_background)
        delta_x_wh = np.diff(x_wh)
        delta_x_hvac = np.diff(x_hvac)
        delta_u_hvac = np.diff(u_hvac)

        # Delta OLS
        A_delta = np.column_stack([
            delta_x_wh,
            delta_x_hvac,
            delta_u_hvac
        ])

        coefs, _, _, _ = np.linalg.lstsq(A_delta, delta_y, rcond=None)

        kw_wh, kw_hvac, gamma = coefs

        # Reconstruct in level space
        wh_predicted = kw_wh * x_wh

        hvac_predicted = (
            kw_hvac * x_hvac
            + gamma * u_hvac
        )

        return {
            'kw_wh': kw_wh,
            'kw_hvac': kw_hvac,
            'gamma': gamma,

            'wh_predicted': wh_predicted,
            'hvac_predicted': hvac_predicted,
            'combined_predicted': wh_predicted + hvac_predicted,

            'x_wh': x_wh,
            'x_hvac': x_hvac,
            'u_hvac': u_hvac,

            'feeder_minus_background': feeder_minus_background,
        }
    
    def diagnose_hvac_multicollinearity(
        self,
        hvac_ground_truth: pd.Series,
        corr_threshold: float = 0.85,
        max_pairs: int = 10,
        min_nonzero_count: int = 3,
        min_std: float = 0.05,
        print_summary: bool = True,
        ) -> dict:
        """
        Diagnose HVAC device-level behavior and pairwise multicollinearity.

        This method intentionally separates two questions:
        1) Which individual HVAC devices are active and correlated with the
           aggregate HVAC ground truth?
        2) Among active devices, are any pairs highly correlated with each other?

        Flat or nearly inactive devices are filtered out before pairwise
        correlation is evaluated, because two all-zero vectors can otherwise
        appear perfectly correlated while carrying no useful information.
        """
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)
        hvac_truth = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        summary = pd.DataFrame(index=hvac_mean_matrix.columns)
        summary['mean'] = hvac_mean_matrix.mean(axis=0)
        summary['std'] = hvac_mean_matrix.std(axis=0)
        summary['nonzero_count'] = (hvac_mean_matrix > 0.5).sum(axis=0)
        summary['corr_with_truth'] = [
            np.corrcoef(hvac_mean_matrix[col].values, hvac_truth)[0, 1]
            for col in hvac_mean_matrix.columns
        ]

        active_cols = summary.index[
            (summary['nonzero_count'] >= min_nonzero_count) &
            (summary['std'] > min_std)
        ]
        hvac_active = hvac_mean_matrix[active_cols]
        corr_matrix = hvac_active.corr()

        high_corr_pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= corr_threshold:
                    col_i = cols[i]
                    col_j = cols[j]
                    high_corr_pairs.append({
                        'device_i': col_i,
                        'device_j': col_j,
                        'pair_corr': corr_val,
                        'device_i_truth_corr': summary.loc[col_i, 'corr_with_truth'],
                        'device_j_truth_corr': summary.loc[col_j, 'corr_with_truth'],
                    })

        high_corr_pairs = sorted(
            high_corr_pairs,
            key=lambda x: abs(x['pair_corr']),
            reverse=True,
        )

        top_positive = summary.sort_values('corr_with_truth', ascending=False).head(5)
        top_negative = summary.sort_values('corr_with_truth', ascending=True).head(5)

        if print_summary:
            print('\nHVAC device summary, sorted by activity:')
            print(summary.sort_values('nonzero_count', ascending=False))

            print('\nTop positive HVAC devices:')
            print(top_positive)

            print('\nTop negative HVAC devices:')
            print(top_negative)

            print(f'\nActive HVAC devices after filtering: {len(active_cols)} / {hvac_mean_matrix.shape[1]}')
            print('\nHighly correlated active HVAC device pairs:')
            if high_corr_pairs:
                for pair in high_corr_pairs[:max_pairs]:
                    print(
                        f"{pair['device_i']}  <->  {pair['device_j']} | "
                        f"pair corr = {pair['pair_corr']:.3f} | "
                        f"truth corr = ({pair['device_i_truth_corr']:.3f}, "
                        f"{pair['device_j_truth_corr']:.3f})"
                    )
            else:
                print('No highly correlated active HVAC pairs found.')

        return {
            'hvac_mean_matrix': hvac_mean_matrix,
            'hvac_active_matrix': hvac_active,
            'corr_matrix': corr_matrix,
            'summary': summary,
            'active_cols': list(active_cols),
            'top_positive': top_positive,
            'top_negative': top_negative,
            'high_corr_pairs': high_corr_pairs,
            'hvac_truth': hvac_truth,
        }

    def run_outlier_hvac_experiment(
        self,
        hvac_ground_truth: pd.Series,
        outlier_device: str = None,
        outlier_rank: int = 0,
        ) -> dict:
        """
        Isolate one suspicious HVAC device from the rest of the HVAC population.

        By default, the selected outlier is the HVAC device with the most
        negative correlation with aggregate HVAC ground truth. The model is:

            y = kw_wh*x_wh + kw_hvac_rest*x_hvac_rest
                + kw_hvac_outlier*x_hvac_outlier + baseline

        This is a diagnostic for heterogeneity/outlier behavior, not a claim
        that negative device coefficients are physically meaningful.
        """
        wh_mean_matrix = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)
        hvac_truth = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        summary = pd.DataFrame(index=hvac_mean_matrix.columns)
        summary['mean'] = hvac_mean_matrix.mean(axis=0)
        summary['std'] = hvac_mean_matrix.std(axis=0)
        summary['nonzero_count'] = (hvac_mean_matrix > 0.5).sum(axis=0)
        summary['corr_with_truth'] = [
            np.corrcoef(hvac_mean_matrix[col].values, hvac_truth)[0, 1]
            for col in hvac_mean_matrix.columns
        ]

        if outlier_device is None:
            sorted_negative = summary.sort_values('corr_with_truth', ascending=True)
            outlier_device = sorted_negative.index[outlier_rank]

        x_wh = wh_mean_matrix.sum(axis=1).values
        x_hvac_outlier = hvac_mean_matrix[outlier_device].values
        x_hvac_rest = hvac_mean_matrix.drop(columns=[outlier_device]).sum(axis=1).values

        background_constant = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        A = np.column_stack([
            x_wh,
            x_hvac_rest,
            x_hvac_outlier,
            np.ones(len(x_wh)),
        ])

        coefs, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)
        kw_wh, kw_hvac_rest, kw_hvac_outlier, baseline = coefs

        wh_predicted = kw_wh * x_wh
        hvac_rest_predicted = kw_hvac_rest * x_hvac_rest
        hvac_outlier_predicted = kw_hvac_outlier * x_hvac_outlier
        hvac_predicted = hvac_rest_predicted + hvac_outlier_predicted

        hvac_r2 = 1 - np.sum((hvac_predicted - hvac_truth) ** 2) / np.sum((hvac_truth - hvac_truth.mean()) ** 2)

        return {
            'outlier_device': outlier_device,
            'summary': summary,
            'kw_wh': kw_wh,
            'kw_hvac_rest': kw_hvac_rest,
            'kw_hvac_outlier': kw_hvac_outlier,
            'baseline': baseline,
            'x_wh': x_wh,
            'x_hvac_rest': x_hvac_rest,
            'x_hvac_outlier': x_hvac_outlier,
            'wh_predicted': wh_predicted,
            'hvac_rest_predicted': hvac_rest_predicted,
            'hvac_outlier_predicted': hvac_outlier_predicted,
            'hvac_predicted': hvac_predicted,
            'hvac_r2': hvac_r2,
            'feeder_minus_background': feeder_minus_background,
        }
    
    def _run_kw_clustered_hvac_only_delta_ols(
            self,
            hvac_mean_matrix: pd.DataFrame,
            hvac_target: np.ndarray,
            n_clusters: int = 3,
            min_nonzero_count: int = 3,
            min_std: float = 0.05,
            threshold: float = 0.5,
        ) -> dict:
        """
        Cluster active HVAC devices by estimated effective kW, then run
        HVAC-only delta OLS against an HVAC residual target.

        This method assumes WH has already been estimated and removed:

            hvac_target(t) = feeder_minus_background(t) - wh_predicted(t)

        Then it models:

            Δhvac_target(t) ≈ Σ_k kw_cluster_k · Δx_cluster_k(t)

        No WH coefficient is estimated here.
        """
        from sklearn.cluster import KMeans

        hvac_target = np.asarray(hvac_target, dtype=float)

        # ── Filter active HVAC devices ───────────────────────────────────
        summary = pd.DataFrame(index=hvac_mean_matrix.columns)
        summary["mean"] = hvac_mean_matrix.mean(axis=0)
        summary["std"] = hvac_mean_matrix.std(axis=0)
        summary["nonzero_count"] = (hvac_mean_matrix > threshold).sum(axis=0)

        active_cols = summary.index[
            (summary["nonzero_count"] >= min_nonzero_count) &
            (summary["std"] >= min_std)
        ].tolist()

        inactive_cols = [
            col for col in hvac_mean_matrix.columns
            if col not in active_cols
        ]

        if len(active_cols) == 0:
            raise ValueError("No active HVAC devices survived filtering.")

        if n_clusters > len(active_cols):
            n_clusters = len(active_cols)

        hvac_active = hvac_mean_matrix[active_cols]

        # ── Estimate per-device effective kW against HVAC target ─────────
        device_kws = {}

        for col in active_cols:
            m_i = hvac_active[col].values
            denom = np.sum(m_i ** 2)

            if denom < 1e-6:
                continue

            kw_i = np.sum(m_i * hvac_target) / denom
            device_kws[col] = kw_i

        if len(device_kws) == 0:
            raise ValueError("No valid per-device kW estimates after filtering.")

        if n_clusters > len(device_kws):
            n_clusters = len(device_kws)

        device_names = list(device_kws.keys())
        kw_array = np.array(list(device_kws.values())).reshape(-1, 1)

        # ── Cluster active HVAC devices by estimated kW ──────────────────
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(kw_array)

        clusters = {k: [] for k in range(n_clusters)}

        for name, label in zip(device_names, cluster_labels):
            clusters[label].append(name)

        # ── Build cluster aggregate regressors ───────────────────────────
        cluster_x = {}

        for k, members in clusters.items():
            cluster_x[k] = hvac_mean_matrix[members].sum(axis=1).values

        # Optional inactive/rest signal
        if len(inactive_cols) > 0:
            inactive_x = hvac_mean_matrix[inactive_cols].sum(axis=1).values
        else:
            inactive_x = np.zeros_like(hvac_target)

        include_inactive = inactive_x.std() > 1e-6

        # ── Delta OLS on HVAC residual only ──────────────────────────────
        delta_target = np.diff(hvac_target)

        regressors = []

        for k in range(n_clusters):
            regressors.append(np.diff(cluster_x[k]))

        if include_inactive:
            regressors.append(np.diff(inactive_x))

        A_delta = np.column_stack(regressors)

        coefs, _, _, _ = np.linalg.lstsq(
            A_delta,
            delta_target,
            rcond=None
        )

        kw_clusters = coefs[:n_clusters]

        if include_inactive:
            kw_inactive = coefs[-1]
        else:
            kw_inactive = 0.0

        # ── Reconstruct HVAC prediction in level domain ──────────────────
        hvac_predicted = np.zeros_like(hvac_target, dtype=float)

        for k in range(n_clusters):
            hvac_predicted += kw_clusters[k] * cluster_x[k]

        if include_inactive:
            hvac_predicted += kw_inactive * inactive_x

        # ── Diagnostics ──────────────────────────────────────────────────
        print("\nHVAC-only clustered delta OLS")
        print("-----------------------------")
        print(f"Total HVAC devices:       {hvac_mean_matrix.shape[1]}")
        print(f"Active clustered devices: {len(active_cols)}")
        print(f"Inactive/rest devices:    {len(inactive_cols)}")

        print("\nDevice kW estimates after filtering:")
        for name, kw in device_kws.items():
            short_name = name.split("ochre_load_")[-1].replace(".csv", "")
            print(f"  Unit {short_name}: {kw:.0f} W")

        print(f"\nCluster centroids (W): {kmeans.cluster_centers_.flatten()}")
        print(f"Cluster sizes: {[len(clusters[k]) for k in range(n_clusters)]}")

        print("\nFinal HVAC cluster coefficients:")
        for k, kw in enumerate(kw_clusters):
            print(f"  kw_cluster_{k}: {kw:.0f} W (n={len(clusters[k])})")

        if include_inactive:
            print(f"  kw_inactive_rest: {kw_inactive:.0f} W (n={len(inactive_cols)})")

        return {
            "kw_clusters": kw_clusters,
            "kw_inactive": kw_inactive,

            "device_kws": device_kws,
            "cluster_assignments": clusters,
            "cluster_centroids": kmeans.cluster_centers_.flatten(),

            "active_cols": active_cols,
            "inactive_cols": inactive_cols,
            "summary": summary,

            "cluster_x": cluster_x,
            "inactive_x": inactive_x,

            "hvac_target": hvac_target,
            "hvac_predicted": hvac_predicted,
        }
    
    def _run_per_device_hvac_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            exclude : list = None
            ) -> dict:
        """
        Run OLS with a separate regressor per HVAC device to estimate
        each device's individual rated power.

        Solves:
            feeder(t) - background ≈ kw_wh * x_wh(t)
                                    + Σ_i [ kw_hvac_i * ON_i(t) ]

        Where ON_i(t) is the thresholded binary state (mean > 0.5)
        for each HVAC device, not the continuous posterior mean.
        This reflects the physical reality that a device draws its
        rated power when ON, regardless of the posterior uncertainty.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
            Shape: (144, n_wh).

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
            Shape: (144, n_hvac).

        Returns
        -------
        dict
            Contains:
            - 'kw_wh'            : float, estimated watts per WH unit
            - 'kw_hvac_per_device': pd.Series, estimated kW per HVAC device
            - 'wh_predicted'     : np.ndarray, shape (144,)
            - 'hvac_predicted'   : np.ndarray, shape (144,)
            - 'x_wh'             : np.ndarray, shape (144,)
            - 'on_matrix'        : pd.DataFrame, binary ON states, shape (144, n_hvac)
            - 'feeder_minus_background': np.ndarray, shape (144,)
        """
        # ── WH aggregated regressor ──────────────────────────────────
        exclude = exclude or []
        x_wh = wh_mean_matrix.sum(axis=1).values

        # ── Drop never-ON HVAC devices (near-zero posterior mean) ────
        active_cols = hvac_mean_matrix.columns[
            hvac_mean_matrix.max(axis=0) > 0.01
        ]
        active_cols = [c for c in active_cols if c not in exclude]
        hvac_active = hvac_mean_matrix[active_cols]

        # ── HVAC binary states — threshold posterior mean at 0.5 ────
        # on_matrix = (hvac_mean_matrix > 0.5).astype(float)

        # ── Background subtraction ───────────────────────────────────
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = self.feeder_demand['power_out'].values - background_constant

        # ── Design matrix: [x_wh | ON_1 | ON_2 | ... | ON_27] ──────
        A = np.column_stack([x_wh, hvac_active.values])

        # ── Solve OLS ────────────────────────────────────────────────
        coefs, _, _, _ = np.linalg.lstsq(A, feeder_minus_background, rcond=None)
        kw_wh          = coefs[0]
        # ── Assign coefficients — zero for never-ON devices ─────────
        kw_hvac_per_device = pd.Series(0.0, index=hvac_mean_matrix.columns)
        kw_hvac_per_device[active_cols] = coefs[1:]

        # ── Predictions ──────────────────────────────────────────────
        wh_predicted   = kw_wh * x_wh
        hvac_predicted = hvac_active.values @ coefs[1:]

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac_per_device':      kw_hvac_per_device,
            'wh_predicted':            wh_predicted,
            'hvac_predicted':          hvac_predicted,
            'x_wh':                    x_wh,
            'hvac_active':             hvac_active,
            'feeder_minus_background': feeder_minus_background,
        }

    def run(
        self,
        ambient_temp: np.ndarray = None,
        t_base: float = 70.0
        ):

        """
        Run the OLS analysis.

        If ambient_temp is provided, also run the temperature-interaction
        HVAC model: (I made up temp values and did not work. Maybe give up on this)

            P(t) = beta_0
                + beta_1 x_wh(t)
                + beta_2 x_hvac(t)
                + beta_3 x_hvac(t) T_cool(t)

        where:

            T_cool(t) = max(0, T_amb(t) - T_base)

        Parameters
        ----------
        ambient_temp : np.ndarray, optional
            Ambient/weather temperature aligned with the 10-minute chunks.
            Expected shape: (144,)

        t_base : float
            Cooling balance/base temperature. Default is 70.0.
        """

        # ── Build state matrices ─────────────────────────────────────────
        wh_state_matrix = self._build_state_matrix(
            all_dfs=self.wh_all_dfs
        )

        hvac_state_matrix = self._build_state_matrix(
            all_dfs=self.hvac_all_dfs
        )

        # ── Build posterior mean matrices ────────────────────────────────
        wh_mean_matrix = self._build_mean_matrix(
            histories=self.wh_histories
        )

        hvac_mean_matrix = self._build_mean_matrix(
            histories=self.hvac_histories
        )

        # ── Build posterior variance matrix for HVAC ─────────────────────
        hvac_variance_matrix = self._build_variance_matrix(
            histories=self.hvac_histories
        )

        # ── Standard simultaneous OLS ────────────────────────────────────
        simultaneous_results = self._run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        # ── Delta OLS ────────────────────────────────────────────────────
        delta_results = self._run_delta_ols(
            x_wh=simultaneous_results['x_wh'],
            x_hvac=simultaneous_results['x_hvac'],
            feeder_minus_background=simultaneous_results['feeder_minus_background'],
        )

        seq_results = self._run_sequential_ols (
            x_hvac=simultaneous_results['x_hvac'],
            wh_predicted=delta_results['wh_predicted'],
            feeder_minus_background = simultaneous_results['feeder_minus_background'],
        )

        # ── Ridge OLS ────────────────────────────────────────────────────
        ridge_results = self._run_ridge_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            alpha=1.0,
        )

        # ── Variance-weighted Delta OLS ──────────────────────────────────
        variance_weighted_results = self._run_variance_weighted_delta_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            hvac_variance_matrix=hvac_variance_matrix,
        )

        # ── Optional: Temperature-interaction OLS ────────────────────────
        temp_results = None

        if ambient_temp is not None:
            temp_results = self._run_temperature_interaction_ols(
                wh_mean_matrix=wh_mean_matrix,
                hvac_mean_matrix=hvac_mean_matrix,
                ambient_temp=ambient_temp,
                t_base=t_base,
            )

        # ── Consider the ON time ────────────────────────
        duration_results = self._run_duration_augmented_delta_ols(
        wh_mean_matrix=wh_mean_matrix,
        hvac_mean_matrix=hvac_mean_matrix
        )

        startup_results = self._run_startup_augmented_delta_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix
        )

        per_device_results = self._run_per_device_hvac_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            exclude=['../results/hvac_cosim/ochre_load_16.csv']
        )

        # freq_split = self._run_hvac_frequency_split_after_wh (
        #     wh_mean_matrix=wh_mean_matrix,
        #     hvac_mean_matrix=hvac_mean_matrix
        # )
        # print(type(hvac_mean_matrix))
        # print(len(hvac_mean_matrix.columns))
        # for col in hvac_mean_matrix.columns:
        #     print(col)
        # quit()

        # on_matrix = (hvac_mean_matrix > 0.5).astype(float)
        # Use hvac_active from per_device_results, not the full hvac_mean_matrix
        hvac_active = per_device_results['hvac_active']
        corr = hvac_active.corr()

        high_corr = (corr.abs() > 0.8) & (corr != 1.0)
        # print(f"High correlation pairs: {high_corr.sum().sum() // 2}")

        A = np.column_stack([per_device_results['x_wh'], hvac_active.values])
        # print(f"Condition number: {np.linalg.cond(A):.1f}")
        # quit()
        # print(f"Design matrix rank: {np.linalg.matrix_rank(A)}")
        # print(f"Design matrix shape: {A.shape}")

        # print("\n\n------------------------\n\n")
        problem_devices = [
            '../results/hvac_cosim/ochre_load_16.csv',
            '../results/hvac_cosim/ochre_load_14.csv',
            '../results/hvac_cosim/ochre_load_15.csv',
            '../results/hvac_cosim/ochre_load_9.csv',
            '../results/hvac_cosim/ochre_load_11.csv',
        ]

        hvac_active = per_device_results['hvac_active']
        # print(hvac_active[problem_devices].corr().round(2))

        # Also check how often each is ON
        # print("\nMean posterior (fraction of time ON):")
        # print(hvac_active[problem_devices].mean().round(3))
        # print("Mean posterior for all devices (sorted):")
        # print(hvac_active.mean().sort_values(ascending=False).round(3))

        # quit()

        

        # quit()

        # ── Return results ───────────────────────────────────────────────
        results = {
            
            # Per device validation:

            'per_d_kw_wh':                   per_device_results['kw_wh'],
            'per_d_kw_hvac':                 per_device_results['kw_hvac_per_device'],
            'per_d_wh_predicted':            per_device_results['wh_predicted'],
            'per_d_hvac_predicted':          per_device_results['hvac_predicted'],
            'per_d_x_wh':                    per_device_results['x_wh'],
            'per_d_feeder_minus_background': per_device_results['feeder_minus_background'],
            'hvac_active':                   per_device_results['hvac_active'],


            # Standard OLS
            'wh_predicted':            simultaneous_results['wh_predicted'],
            'hvac_predicted':          simultaneous_results['hvac_predicted'],
            'x_wh':                    simultaneous_results['x_wh'],
            'x_hvac':                  simultaneous_results['x_hvac'],
            'kw_wh':                   simultaneous_results['kw_wh'],
            'kw_hvac':                 simultaneous_results['kw_hvac'],
            'feeder_minus_background': simultaneous_results['feeder_minus_background'],

            # Delta OLS
            'delta_wh_predicted':      delta_results['wh_predicted'],
            'delta_hvac_predicted':    delta_results['hvac_predicted'],
            'delta_kw_wh':             delta_results['kw_wh'],
            'delta_kw_hvac':           delta_results['kw_hvac'],

            # Sequential OLS:
            'seq_kw_hvac':             seq_results['kw_hvac'],
            'seq_baseline':            seq_results['baseline'],
            'seq_hvac_predicted':      seq_results['hvac_predicted'],

            # Ridge OLS
            'ridge_wh_predicted':      ridge_results['wh_predicted'],
            'ridge_hvac_predicted':    ridge_results['hvac_predicted'],
            'ridge_kw_wh':             ridge_results['kw_wh'],
            'ridge_kw_hvac_1':         ridge_results['kw_hvac_1'],
            'ridge_kw_hvac_2':         ridge_results['kw_hvac_2'],
            'ridge_kw_hvac_3':         ridge_results['kw_hvac_3'],
            'ridge_kw_hvac_4':         ridge_results['kw_hvac_4'],
            'ridge_kw_hvac_rest':      ridge_results['kw_hvac_rest'],

            # Variance-weighted Delta OLS
            'vw_wh_predicted':         variance_weighted_results['wh_predicted'],
            'vw_hvac_predicted':       variance_weighted_results['hvac_predicted'],
            'vw_kw_wh':                variance_weighted_results['kw_wh'],
            'vw_kw_hvac':              variance_weighted_results['kw_hvac'],
            'vw_x_hvac':               variance_weighted_results['x_hvac'],

            # The ON-Time approach:
            'on_kw_wh':                   duration_results['kw_wh'],
            'on_kw_hvac':                 duration_results['kw_hvac'],
            'on_gamma':                   duration_results['gamma'],
            'on_wh_predicted':            duration_results['wh_predicted'],
            'on_hvac_predicted':          duration_results['hvac_predicted'],
            'on_combined_predicted':      duration_results['combined_predicted'],
            'on_x_wh':                    duration_results['x_wh'],
            'on_x_hvac':                  duration_results['x_hvac'],
            'on_s_hvac':                  duration_results['s_hvac'],
            'on_d_hvac_sum':              duration_results['d_hvac_sum'],
            'on_d_hvac_avg':              duration_results['d_hvac_avg'],
            'on_duration_feature':        duration_results['duration_feature'],
            'on_feeder_minus_background': duration_results['feeder_minus_background'],
            # Startup/Event-intensity approach
            'startup_kw_wh':              startup_results['kw_wh'],
            'startup_kw_hvac':            startup_results['kw_hvac'],
            'startup_gamma':              startup_results['gamma'],
            'startup_wh_predicted':       startup_results['wh_predicted'],
            'startup_hvac_predicted':     startup_results['hvac_predicted'],
            'startup_combined_predicted': startup_results['combined_predicted'],
            'startup_x_wh':               startup_results['x_wh'],
            'startup_x_hvac':             startup_results['x_hvac'],
            'startup_u_hvac':             startup_results['u_hvac'],
        }
        # 'kw_wh': kw_wh,
        #     'kw_hvac': kw_hvac,
        #     'gamma': gamma,

        #     'wh_predicted': wh_predicted,
        #     'hvac_predicted': hvac_predicted,
        #     'combined_predicted': wh_predicted + hvac_predicted,

        #     'x_wh': x_wh,
        #     'x_hvac': x_hvac,
        #     's_hvac': s_hvac,
        #     'd_hvac_sum': d_hvac_sum,
        #     'd_hvac_avg': d_hvac_avg,
        #     'duration_feature': duration_feature,

        #     'feeder_minus_background': feeder_minus_background,

        if temp_results is not None:
            results.update({
                # Temperature-interaction OLS
                'temp_wh_predicted':       temp_results['wh_predicted'],
                'temp_hvac_predicted':     temp_results['hvac_predicted'],
                'temp_combined_predicted': temp_results['combined_predicted'],

                'temp_kw_wh':              temp_results['kw_wh'],
                'temp_kw_hvac_base':       temp_results['kw_hvac_base'],
                'temp_kw_hvac_temp':       temp_results['kw_hvac_temp'],
                'temp_baseline':           temp_results['baseline'],

                'temp_T_cool':             temp_results['T_cool'],
                'temp_x_wh':               temp_results['x_wh'],
                'temp_x_hvac':             temp_results['x_hvac'],
            })

        return results