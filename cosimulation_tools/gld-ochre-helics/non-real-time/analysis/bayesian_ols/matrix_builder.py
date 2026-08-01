'''
Author: MidrarAdham
Created: Sat Aug 01 2026
'''
"""
matrix_builder.py

Build only the matrices and signals needed by the current Bayesian + OLS
workflow.

Current scope:
    Stage 1: simultaneous WH/HVAC OLS
    Stage 2: per-device HVAC OLS

This file intentionally does not include variance matrices yet because the
current OLS run path does not use them.

Main idea:
    BayesianEstimator produces histories.
    MatrixBuilder converts those histories into mean matrices.
    OLSModel uses those mean matrices to estimate coefficients.
"""
import pandas as pd

class MatrixBuilder:
    """
    Build mean matrices and target signals for the Bayesian + OLS workflow.

    In this project:
        - rows are 10-minute time windows
        - columns are expected ON signals from Bayesian posterior means
        - target signals are measured transformer or feeder power signals
    """
    def build_mean_matrix(
        self,
        histories: dict,
    ) -> pd.DataFrame:
        """
        Build a posterior mean matrix from Bayesian histories.

        Parameters
        ----------
        histories:
            Output from BayesianEstimator.fit_many().

            Expected structure:
                {
                    device_name: {
                        "mean": [...]
                    }
                }

        Returns
        -------
        pd.DataFrame
            Posterior mean matrix.

            Rows:
                10-minute time windows.

            Columns:
                One column per device.

            Values:
                Posterior mean ON probability.
        """
        mean_matrix = pd.DataFrame(
            {
                device_name: device_history["mean"]
                for device_name, device_history in histories.items()
            }
        )

        return mean_matrix

    def build_expected_on_signal(
            self,
            mean_matrix: pd.DataFrame,
            signal_name: str,
            ) -> pd.Series:
        """
        Sum a posterior mean matrix across devices.

        This produces the expected number of ON devices at each time window.

        Example
        -------
        If one row has:

            device_1 = 0.8
            device_2 = 0.3
            device_3 = 0.1

        then:

            expected_on = 1.2

        Parameters
        ----------
        mean_matrix:
            Posterior mean matrix.

        signal_name:
            Name assigned to the output signal.

        Returns
        -------
        pd.Series
            Expected ON-count signal.
        """
        expected_on_signal = mean_matrix.sum(axis=1)
        expected_on_signal.name = signal_name

        return expected_on_signal

    def build_simultaneous_mean_matrix(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            ) -> pd.DataFrame:
        """
        Build the mean matrix for simultaneous WH/HVAC OLS.

        Output columns:
            wh_expected_on
            hvac_expected_on

        This corresponds to the current _run_simultaneous_ols logic.

        Parameters
        ----------
        wh_mean_matrix:
            Posterior mean matrix for water heaters.

        hvac_mean_matrix:
            Posterior mean matrix for HVAC devices.
        
        ev_mean_matrix (Not Implemented):
            Posterior mean matrix for EVs.

        Returns
        -------
        pd.DataFrame
            Two-column OLS input matrix.
        """

        wh_expected_on_signal = self.build_expected_on_signal(
            mean_matrix=wh_mean_matrix,
            signal_name="wh_expected_on",
        )

        hvac_expected_on_signal = self.build_expected_on_signal(
            mean_matrix=hvac_mean_matrix,
            signal_name="hvac_expected_on",
        )

        simultaneous_mean_matrix = pd.concat(
            [
                wh_expected_on_signal,
                hvac_expected_on_signal,
            ],
            axis=1,
        )

        return simultaneous_mean_matrix

    def build_per_device_hvac_mean_matrix(
        self,
        wh_mean_matrix: pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        excluded_hvac_devices: list[str] | None = None,
        minimum_hvac_mean: float = 0.01,
    ) -> pd.DataFrame:
        """
        Build the mean matrix for per-device HVAC OLS.

        Output columns:
            wh_expected_on
            one column for each active HVAC device

        This corresponds to the current _run_per_device_hvac_ols logic.

        Parameters
        ----------
        wh_mean_matrix:
            Posterior mean matrix for water heaters.

        hvac_mean_matrix:
            Posterior mean matrix for HVAC devices.

        excluded_hvac_devices:
            Optional list of HVAC device columns to exclude.

        minimum_hvac_mean:
            Remove HVAC devices whose maximum posterior mean is less than or
            equal to this value. This follows the old logic that removed
            effectively inactive HVAC devices.

        Returns
        -------
        pd.DataFrame
            OLS input matrix for per-device HVAC estimation.
        """
        if excluded_hvac_devices is None:
            excluded_hvac_devices = []

        wh_expected_on_signal = self.build_expected_on_signal(
            mean_matrix=wh_mean_matrix,
            signal_name="wh_expected_on",
        )

        active_hvac_device_names = [
            device_name
            for device_name in hvac_mean_matrix.columns
            if device_name not in excluded_hvac_devices
            and hvac_mean_matrix[device_name].max() > minimum_hvac_mean
        ]

        active_hvac_mean_matrix = hvac_mean_matrix[
            active_hvac_device_names
        ].copy()

        per_device_hvac_mean_matrix = pd.concat(
            [
                wh_expected_on_signal,
                active_hvac_mean_matrix,
            ],
            axis=1,
        )

        return per_device_hvac_mean_matrix

    def build_target_signal(
        self,
        feeder_demand: pd.DataFrame,
        power_column: str = "power_out",
        signal_name: str = "target_signal",
    ) -> pd.Series:
        """
        Extract the measured transformer or feeder power signal.

        Parameters
        ----------
        feeder_demand:
            DataFrame that contains the measured power signal.

        power_column:
            Name of the column containing measured power.

        signal_name:
            Name assigned to the output signal.

        Returns
        -------
        pd.Series
            Measured target signal.
        """
        if power_column not in feeder_demand.columns:
            raise KeyError(
                f"Column '{power_column}' was not found in feeder_demand."
            )

        target_signal = feeder_demand[power_column].copy()
        target_signal = pd.to_numeric(target_signal, errors="coerce")
        target_signal.name = signal_name

        return target_signal

    def build_background_adjusted_target_signal(
        self,
        feeder_demand: pd.DataFrame,
        power_column: str = "power_out",
        signal_name: str = "background_adjusted_target_signal",
    ) -> pd.Series:
        """
        Build the background-adjusted target signal used by the current OLS.

        The legacy workflow used the minimum feeder power as a simple
        background estimate:

            background = minimum measured feeder power

            background_adjusted_target_signal = target_signal - background

        Parameters
        ----------
        feeder_demand:
            DataFrame that contains the measured feeder or transformer power.

        power_column:
            Name of the measured power column.

        signal_name:
            Name assigned to the output signal.

        Returns
        -------
        pd.Series
            Background-adjusted measured power signal.
        """
        target_signal = self.build_target_signal(
            feeder_demand=feeder_demand,
            power_column=power_column,
            signal_name="raw_target_signal",
        )

        background_value = target_signal.min()

        background_adjusted_target_signal = target_signal - background_value
        background_adjusted_target_signal.name = signal_name

        return background_adjusted_target_signal