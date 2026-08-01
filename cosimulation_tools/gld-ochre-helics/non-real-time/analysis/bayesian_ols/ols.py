'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import numpy as np
import pandas as pd


class OrdinaryLeastSquare:

    def __init__(
        self,
        wh_histories:   dict,
        hvac_histories: dict,
        wh_all_dfs:     dict,
        hvac_all_dfs:   dict,
        feeder_demand:  pd.DataFrame,
        hvac_sizes : dict
    ):
        """
        Store everything OLS needs to run its analysis.

        Parameters
        ----------
        wh_histories : dict
            Output of BayesianEstimator.fit_many() for WH data.
            Keys are filenames, values are history dicts containing
            'mean', 'variance', etc. for each chunk.

        hvac_histories : dict
            Same as wh_histories but for HVAC data.

        wh_all_dfs : dict
            Output of DataLoader.all_dfs for WH data.

        hvac_all_dfs : dict
            Output of DataLoader.all_dfs for HVAC data.

        feeder_demand : pd.DataFrame
            Transformer-level feeder signal from
            DataLoader.load_transformer_data(). This is the signal
            we are trying to decompose.
        """
        self.wh_histories   = wh_histories
        self.hvac_histories = hvac_histories
        self.wh_all_dfs     = wh_all_dfs
        self.hvac_all_dfs   = hvac_all_dfs
        self.feeder_demand  = feeder_demand
        self.hvac_sizes = hvac_sizes

    # ── Matrix builders ───────────────────────────────────────────────────────

    def _build_mean_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the posterior mean values from each DER's Bayesian history
        and organize them into a single matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'mean' list.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each column is one DER's posterior
            mean ON-probability sequence across all chunks.
        """
        return pd.DataFrame(
            {filename: history['mean']
             for filename, history in histories.items()}
        )

    def _build_variance_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the posterior variance values from each DER's Bayesian
        history and organize them into a matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'variance' list.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each column is one DER's posterior
            variance sequence across all chunks.
        """
        return pd.DataFrame(
            {filename: history['variance']
             for filename, history in histories.items()}
        )

    def _build_state_matrix(self, all_dfs: dict) -> pd.DataFrame:
        """
        Build a binary state matrix from raw per-device dataframes,
        resampled to 10-minute chunks.

        Parameters
        ----------
        all_dfs : dict
            Output of DataLoader.all_dfs for WH or HVAC data.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each value is the mean state
            (fraction of minutes ON) within each 10-minute chunk.
        """
        state_matrix = {}
        for filename, df in all_dfs.items():
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            states = df.set_index('time')['state']
            states.index = pd.to_datetime(states.index)
            states = states.resample('10min').mean()
            state_matrix[filename] = states.values
        return pd.DataFrame(state_matrix)

    # ── OLS methods ───────────────────────────────────────────────────────────

    def _run_simultaneous_ols(
        self,
        wh_mean_matrix:   pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        ) -> dict:
        """
        Stage 1 — Warm-up OLS.

        Estimates a single rated power per DER type using aggregated
        Bayesian posterior means as regressors.

        Solves:
            feeder(t) - background ≈ kw_wh x x_wh(t) + kw_hvac x x_hvac(t)

        This is the fallback method used when insufficient history has
        accumulated for per-device estimation.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.

        Returns
        -------
        dict
            Contains kw_wh, kw_hvac, wh_predicted, hvac_predicted,
            x_wh, x_hvac, feeder_minus_background.
        """
        x_wh   = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        A = np.column_stack([x_wh, x_hvac])
        kw_estimate, _, _, _ = np.linalg.lstsq(A, feeder_minus_background,
                                                rcond=None)
        kw_wh, kw_hvac = kw_estimate

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac':                 kw_hvac,
            'wh_predicted':            kw_wh   * x_wh,
            'hvac_predicted':          kw_hvac * x_hvac,
            'x_wh':                    x_wh,
            'x_hvac':                  x_hvac,
            'feeder_minus_background': feeder_minus_background,
        }

    def _run_per_device_hvac_ols(
        self,
        wh_mean_matrix:   pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        exclude:          list = None
        ) -> dict:
        """
        Stage 2 — Per-device OLS.

        Estimates a separate rated power for each HVAC device using
        its individual Bayesian posterior mean as a regressor. Requires
        sufficient history (empirically ~10 days) for stable estimates.

        Solves:
            feeder(t) - background ≈ kw_wh × x_wh(t)
                                   + Σ_i [ kw_hvac_i × mean_i(t) ]

        Devices that were never ON (max posterior mean ≤ 0.01) are
        automatically excluded as they carry no information. Additional
        devices can be excluded via the `exclude` parameter (e.g.
        two-state devices that violate the ON/OFF power assumption).

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
            Shape: (n_chunks, n_wh).

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
            Shape: (n_chunks, n_hvac).

        exclude : list, optional
            List of HVAC device filenames to exclude from estimation.
            Excluded devices receive a coefficient of 0.0.

        Returns
        -------
        dict
            Contains:
            - 'kw_wh'             : float, estimated W per WH unit
            - 'kw_hvac_per_device': pd.Series, estimated W per HVAC device
            - 'wh_predicted'      : np.ndarray, shape (n_chunks,)
            - 'hvac_predicted'    : np.ndarray, shape (n_chunks,)
            - 'x_wh'              : np.ndarray, shape (n_chunks,)
            - 'hvac_active'       : pd.DataFrame, active device mean matrix
            - 'feeder_minus_background': np.ndarray, shape (n_chunks,)
        """
        exclude = exclude or []

        # ── WH aggregated regressor ──────────────────────────────────
        x_wh = wh_mean_matrix.sum(axis=1).values

        # ── Drop never-ON and explicitly excluded HVAC devices ───────
        active_cols = [col for col in hvac_mean_matrix.columns if hvac_mean_matrix[col].max() > 0.01 and col not in exclude]
        hvac_active = hvac_mean_matrix[active_cols]



        # ── Background subtraction ───────────────────────────────────
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        # ── Design matrix: [x_wh | mean_1 | mean_2 | ... | mean_N] ─
        A = np.column_stack([x_wh, hvac_active.values])

        # ── Solve OLS ────────────────────────────────────────────────
        coefs, _, _, _ = np.linalg.lstsq(A, feeder_minus_background,rcond=None)
        kw_wh = coefs[0]

        # Assign coefficients — zero for excluded/never-ON devices
        kw_hvac_per_device = pd.Series(0.0, index=hvac_mean_matrix.columns)
        kw_hvac_per_device[active_cols] = coefs[1:]

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac_per_device':      kw_hvac_per_device,
            'wh_predicted':            kw_wh * x_wh,
            'hvac_predicted':          hvac_active.values @ coefs[1:],
            'x_wh':                    x_wh,
            'hvac_active':             hvac_active,
            'feeder_minus_background': feeder_minus_background,
        }



    def _get_fixed_hvac_kw_bins(self) -> list[tuple[float, float, str]]:
        """
        Define fixed bins for HVAC electric power, and, later, water heaters.

        This method is used internally by OLS.

        Units:
            kW

        Returns
        -------
        list of tuples
            Each tuple is:
                lower_kw, upper_kw, bin_label
        """
        return [
            (0.0, 0.5, "0_0p5_kw"),
            (0.5, 1.0, "0p5_1_kw"),
            (1.0, 1.5, "1_1p5_kw"),
            (1.5, 2.0, "1p5_2_kw"),
            (2.0, 2.5, "2_2p5_kw"),
            (2.5, 3.0, "2p5_3_kw"),
            (3.0, 4.0, "3_4_kw"),
            (4.0, float("inf"), "4plus_kw"),
        ]

    def _assign_devices_to_fixed_bins(
            self,
            hvac_mean_matrix: pd.DataFrame,
            hvac_kw_by_device: dict,
            ) -> dict:
        """
        STAGE 1 
        Assign each HVAC device to a fixed kW bin.

        Parameters
        ----------
        hvac_mean_matrix : pd.DataFrame
            HVAC posterior mean matrix.
            Columns are HVAC device filenames.

        hvac_kw_by_device : dict
            Dictionary mapping HVAC device filename to representative kW.

            Example:
                {
                    "../results/hvac_cosim/ochre_load_1.csv": 0.72,
                    "../results/hvac_cosim/ochre_load_2.csv": 1.34,
                }

        Returns
        -------
        dict
            Mapping from HVAC filename to bin label.
        """
        fixed_bins = self._get_fixed_hvac_kw_bins()
        device_to_bin = {}

        for device_name in hvac_mean_matrix.columns:
            if device_name not in hvac_kw_by_device:
                continue

            device_kw = hvac_kw_by_device[device_name]

            if pd.isna(device_kw):
                continue

            for lower_kw, upper_kw, bin_label in fixed_bins:
                if lower_kw <= device_kw < upper_kw:
                    device_to_bin[device_name] = bin_label
                    break

        return device_to_bin

    def _build_binned_hvac_mean_matrix(
            self,
            hvac_mean_matrix: pd.DataFrame,
            device_to_bin: dict,
            ) -> pd.DataFrame:
        """
        Aggregate device-level HVAC posterior means into fixed-bin regressors.

        Each output column is one HVAC bin.

        Example:
            bin_1(t) = mean_device_a(t) + mean_device_b(t) + ...
            bin_2(t) = mean_device_c(t) + mean_device_d(t) + ...

        Parameters
        ----------
        hvac_mean_matrix : pd.DataFrame
            Shape:
                n_chunks x n_hvac_devices

        device_to_bin : dict
            Mapping from HVAC device filename to bin label.

        Returns
        -------
        pd.DataFrame
            Shape:
                n_chunks x n_bins
        """
        binned_data = {}

        for device_name, bin_label in device_to_bin.items():
            if device_name not in hvac_mean_matrix.columns:
                continue

            if bin_label not in binned_data:
                binned_data[bin_label] = hvac_mean_matrix[device_name].copy()
            else:
                binned_data[bin_label] = binned_data[bin_label] + hvac_mean_matrix[device_name]

        binned_matrix = pd.DataFrame(binned_data)

        # Keep a stable column order based on the fixed engineering bins.
        fixed_bin_order = [
            bin_label
            for _, _, bin_label in self._get_fixed_hvac_kw_bins()
            if bin_label in binned_matrix.columns
        ]

        binned_matrix = binned_matrix[fixed_bin_order]

        return binned_matrix

    def _run_fixed_bin_hvac_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            hvac_kw_by_device: dict,
            ) -> dict:
        """
        Stage 3 — Fixed bin HVAC OLS.

        Estimates one HVAC coefficient per fixed kW bin.

        Solves:
            feeder(t) - background ≈ kw_wh * x_wh(t)
                                + beta_bin_1 * x_bin_1(t)
                                + beta_bin_2 * x_bin_2(t)
                                + ...

        where:
            x_bin_j(t) = number of expected ON devices in bin j
                        based on Bayesian posterior means.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            WH posterior mean matrix.

        hvac_mean_matrix : pd.DataFrame
            HVAC posterior mean matrix.

        hvac_kw_by_device : dict
            Mapping from HVAC device filename to representative kW.

        Returns
        -------
        dict
            Fixed-bin OLS results.
        """
        x_wh = wh_mean_matrix.sum(axis=1).values

        device_to_bin = self._assign_devices_to_fixed_bins(
            hvac_mean_matrix=hvac_mean_matrix,
            hvac_kw_by_device=hvac_kw_by_device,
        )

        hvac_binned = self._build_binned_hvac_mean_matrix(
            hvac_mean_matrix=hvac_mean_matrix,
            device_to_bin=device_to_bin,
        )

        background_constant = self.feeder_demand["power_out"].values.min()
        feeder_minus_background = (
            self.feeder_demand["power_out"].values - background_constant
        )

        A = np.column_stack([x_wh, hvac_binned.values])

        coefs, _, _, _ = np.linalg.lstsq(
            A,
            feeder_minus_background,
            rcond=None,
        )

        kw_wh = coefs[0]
        kw_hvac_per_bin = pd.Series(coefs[1:], index=hvac_binned.columns)

        hvac_predicted = hvac_binned.values @ coefs[1:]

        return {
            "kw_wh": kw_wh,
            "kw_hvac_per_bin": kw_hvac_per_bin,
            "wh_predicted": kw_wh * x_wh,
            "hvac_predicted": hvac_predicted,
            "x_wh": x_wh,
            "hvac_binned": hvac_binned,
            "device_to_bin": device_to_bin,
            "feeder_minus_background": feeder_minus_background,
        }

    # ── Public API ────────────────────────────────────────────────────────────
    def run(
            self,
            exclude_hvac: list = None,
            subset_devices: list = None,
            hvac_kw_by_device: dict = None,
            ) -> dict:
        """
        Run the full OLS pipeline.

        Builds the mean matrices from Bayesian histories, then runs
        both Stage 1 (simultaneous OLS) and Stage 2 (per-device OLS).

        Parameters
        ----------
        exclude_hvac : list, optional
            HVAC device filenames to exclude from per-device estimation.
            Used for two-state devices or other anomalous devices.

        Returns
        -------
        dict
            Combined results from both OLS stages. Keys are prefixed:
            - 'sim_*'    : Stage 1 simultaneous OLS results
            - 'per_d_*'  : Stage 2 per-device OLS results
        """
        # ── Build matrices ───────────────────────────────────────────
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        if subset_devices is not None:
            # wh_mean_matrix   = wh_mean_matrix[subset_devices]
            hvac_mean_matrix = hvac_mean_matrix[subset_devices]


        # ── Stage 1: Simultaneous OLS ────────────────────────────────
        sim_results = self._run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        # ── Stage 2: Per-device OLS ──────────────────────────────────
        per_device_results = self._run_per_device_hvac_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            exclude=exclude_hvac
        )

        fixed_bin_results = None
        if hvac_kw_by_device is not None:

            fixed_bin_results = self._run_fixed_bin_hvac_ols(

            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            hvac_kw_by_device=hvac_kw_by_device,
            )
        results = {
            # Stage 1 — simultaneous OLS
            "sim_kw_wh": sim_results["kw_wh"],
            "sim_kw_hvac": sim_results["kw_hvac"],
            "sim_wh_predicted": sim_results["wh_predicted"],
            "sim_hvac_predicted": sim_results["hvac_predicted"],
            "sim_x_wh": sim_results["x_wh"],
            "sim_x_hvac": sim_results["x_hvac"],
            "sim_feeder_minus_bg": sim_results["feeder_minus_background"],

            # Stage 2 — per-device OLS
            "per_d_kw_wh": per_device_results["kw_wh"],
            "per_d_kw_hvac": per_device_results["kw_hvac_per_device"],
            "per_d_wh_predicted": per_device_results["wh_predicted"],
            "per_d_hvac_predicted": per_device_results["hvac_predicted"],
            "per_d_x_wh": per_device_results["x_wh"],
            "per_d_hvac_active": per_device_results["hvac_active"],
            "per_d_feeder_minus_bg": per_device_results["feeder_minus_background"],
        }

        if fixed_bin_results is not None:
            results.update({
                "fixed_bin_kw_wh": fixed_bin_results["kw_wh"],
                "fixed_bin_kw_hvac": fixed_bin_results["kw_hvac_per_bin"],
                "fixed_bin_wh_predicted": fixed_bin_results["wh_predicted"],
                "fixed_bin_hvac_predicted": fixed_bin_results["hvac_predicted"],
                "fixed_bin_x_wh": fixed_bin_results["x_wh"],
                "fixed_bin_hvac_binned": fixed_bin_results["hvac_binned"],
                "fixed_bin_device_to_bin": fixed_bin_results["device_to_bin"],
                "fixed_bin_feeder_minus_bg": fixed_bin_results["feeder_minus_background"],
            })

        return results