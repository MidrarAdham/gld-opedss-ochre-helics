'''
Author: MidrarAdham
Created: Sat Aug 01 2026
'''
"""
aggregation_ols.py

Coordinator for the Bayesian + OLS aggregation workflow.

Current scope:
    Stage 1: simultaneous WH/HVAC OLS
    Stage 2: per-device HVAC OLS

This file connects:
    - Bayesian histories
    - MatrixBuilder
    - OLSModel

This file does not:
    - load raw data
    - compute Bayesian histories
    - solve OLS directly
    - define binning methods

Those responsibilities belong to:
    - data_loader.py
    - bayesian_estimator.py
    - ols.py
    - future binning modules
"""
import pandas as pd
from dataclasses import dataclass
from ols import OLSModel, OLSResult
from matrix_builder import MatrixBuilder

@dataclass
class AggregationOLSResult:
    """
    Container for the current Bayesian + OLS aggregation results.

    Attributes
    ----------
    simultaneous_result:
        OLS result for the two-column simultaneous WH/HVAC model.

    per_device_hvac_result:
        OLS result for the per-device HVAC model.

    wh_mean_matrix:
        Posterior mean matrix for water heaters.

    hvac_mean_matrix:
        Posterior mean matrix for HVAC devices.

    simultaneous_mean_matrix:
        Two-column matrix used by simultaneous OLS.

    per_device_hvac_mean_matrix:
        Matrix used by per-device HVAC OLS.

    target_signal:
        Background-adjusted transformer or feeder signal used as the OLS target.
    """
    simultaneous_result: OLSResult
    per_device_hvac_result: OLSResult
    wh_mean_matrix: pd.DataFrame
    hvac_mean_matrix: pd.DataFrame
    simultaneous_mean_matrix: pd.DataFrame
    per_device_hvac_mean_matrix: pd.DataFrame
    target_signal: pd.Series

class AggregationOLS:
    """
    Run the current Bayesian + OLS aggregation workflow.

    This class preserves the current model structure:

        Stage 1:
            simultaneous WH/HVAC OLS

        Stage 2:
            per-device HVAC OLS

    NOT IMPLEMENTED:

        Stage 3:
            fixed engineering-bin HVAC OLS

        Stage 4:
            offline learned-bin HVAC OLS

        Stage 5:
            dynamic learned-bin HVAC OLS
    """

    def __init__(
            self,
            matrix_builder: MatrixBuilder | None = None,
            ols_model: OLSModel | None = None,
            ):
        """
        Create the aggregation OLS coordinator.

        Parameters
        ----------
        matrix_builder:
            Object responsible for building OLS-ready matrices.

        ols_model:
            Object responsible for solving ordinary least squares.
        """
        if matrix_builder is None:
            matrix_builder = MatrixBuilder()

        if ols_model is None:
            ols_model = OLSModel()

        self.matrix_builder = matrix_builder
        self.ols_model = ols_model

    def run(
            self,
            wh_histories: dict,
            hvac_histories: dict,
            feeder_demand: pd.DataFrame,
            power_column: str = "power_out",
            excluded_hvac_devices: list[str] | None = None,
            minimum_hvac_mean: float = 0.01,
            ) -> AggregationOLSResult:
        """
        Run the current Bayesian + OLS workflow.

        Parameters
        ----------
        wh_histories:
            Bayesian histories for water heaters.

        hvac_histories:
            Bayesian histories for HVAC devices.

        feeder_demand:
            Transformer or feeder demand DataFrame.

        power_column:
            Column in feeder_demand used as the measured power signal.

        excluded_hvac_devices:
            Optional list of HVAC device names to remove from the per-device
            HVAC model.

        minimum_hvac_mean:
            HVAC devices whose maximum posterior mean is less than or equal to
            this value are removed from the per-device HVAC model.

        Returns
        -------
        AggregationOLSResult
            All current OLS results and the matrices used to produce them.
        """
        wh_mean_matrix = self.matrix_builder.build_mean_matrix(
            histories=wh_histories,
        )

        hvac_mean_matrix = self.matrix_builder.build_mean_matrix(
            histories=hvac_histories,
        )

        target_signal = self.matrix_builder.build_background_adjusted_target_signal(
            feeder_demand=feeder_demand,
            power_column=power_column,
        )

        simultaneous_result = self.run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            target_signal=target_signal,
        )

        per_device_hvac_result = self.run_per_device_hvac_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            target_signal=target_signal,
            excluded_hvac_devices=excluded_hvac_devices,
            minimum_hvac_mean=minimum_hvac_mean,
        )

        simultaneous_mean_matrix = self.matrix_builder.build_simultaneous_mean_matrix(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        per_device_hvac_mean_matrix = (
            self.matrix_builder.build_per_device_hvac_mean_matrix(
                wh_mean_matrix=wh_mean_matrix,
                hvac_mean_matrix=hvac_mean_matrix,
                excluded_hvac_devices=excluded_hvac_devices,
                minimum_hvac_mean=minimum_hvac_mean,
            )
        )

        return AggregationOLSResult(
            simultaneous_result=simultaneous_result,
            per_device_hvac_result=per_device_hvac_result,
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            simultaneous_mean_matrix=simultaneous_mean_matrix,
            per_device_hvac_mean_matrix=per_device_hvac_mean_matrix,
            target_signal=target_signal,
        )

    def run_simultaneous_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            target_signal: pd.Series,
            ) -> OLSResult:
        """
        Run Stage 1: simultaneous WH/HVAC OLS.

        The OLS input matrix has two columns:

            wh_expected_on
            hvac_expected_on

        The model estimates:

            target_signal ≈
                wh_expected_on * wh_coefficient
                + hvac_expected_on * hvac_coefficient

        Parameters
        ----------
        wh_mean_matrix:
            Posterior mean matrix for water heaters.

        hvac_mean_matrix:
            Posterior mean matrix for HVAC devices.

        target_signal:
            Background-adjusted transformer or feeder signal.

        Returns
        -------
        OLSResult
            Fitted OLS result.
        """
        simultaneous_mean_matrix = self.matrix_builder.build_simultaneous_mean_matrix(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        simultaneous_result = self.ols_model.fit(
            mean_matrix=simultaneous_mean_matrix,
            target_signal=target_signal,
        )

        return simultaneous_result

    def run_per_device_hvac_ols(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            target_signal: pd.Series,
            excluded_hvac_devices: list[str] | None = None,
            minimum_hvac_mean: float = 0.01,
            ) -> OLSResult:
        """
        Run Stage 2: per-device HVAC OLS.

        The OLS input matrix has:

            wh_expected_on
            one column for each active HVAC device

        The model estimates:

            target_signal ≈
                wh_expected_on * wh_coefficient
                + device_1_mean * device_1_coefficient
                + device_2_mean * device_2_coefficient
                + ...

        Parameters
        ----------
        wh_mean_matrix:
            Posterior mean matrix for water heaters.

        hvac_mean_matrix:
            Posterior mean matrix for HVAC devices.

        target_signal:
            Background-adjusted transformer or feeder signal.

        excluded_hvac_devices:
            Optional HVAC device names to remove.

        minimum_hvac_mean:
            HVAC devices whose maximum posterior mean is less than or equal
            to this value are removed.

        Returns
        -------
        OLSResult
            Fitted OLS result.
        """
        per_device_hvac_mean_matrix = (
            self.matrix_builder.build_per_device_hvac_mean_matrix(
                wh_mean_matrix=wh_mean_matrix,
                hvac_mean_matrix=hvac_mean_matrix,
                excluded_hvac_devices=excluded_hvac_devices,
                minimum_hvac_mean=minimum_hvac_mean,
            )
        )

        per_device_hvac_result = self.ols_model.fit(
            mean_matrix=per_device_hvac_mean_matrix,
            target_signal=target_signal,
        )

        return per_device_hvac_result

    def summarize_results(
            self,
            aggregation_result: AggregationOLSResult,
            ) -> dict:
        """
        Create a simple dictionary summary of the current OLS results.

        This is useful for quick printing in main.py.
        """
        simultaneous_coefficients = (
            aggregation_result.simultaneous_result.coefficient_by_column
        )

        per_device_coefficients = (
            aggregation_result.per_device_hvac_result.coefficient_by_column
            )

        summary = {
            "simultaneous_r_squared": (
                aggregation_result.simultaneous_result.r_squared
            ),
            "per_device_hvac_r_squared": (
                aggregation_result.per_device_hvac_result.r_squared
            ),
            "simultaneous_coefficients": simultaneous_coefficients,
            "per_device_hvac_coefficients": per_device_coefficients,
            "number_of_wh_devices": (
                aggregation_result.wh_mean_matrix.shape[1]
            ),
            "number_of_hvac_devices": (
                aggregation_result.hvac_mean_matrix.shape[1]
            ),
            "number_of_active_hvac_devices": (
                aggregation_result.per_device_hvac_mean_matrix.shape[1] - 1
            ),
            "number_of_time_windows": (
                aggregation_result.target_signal.shape[0]
            ),
        }

        return summary