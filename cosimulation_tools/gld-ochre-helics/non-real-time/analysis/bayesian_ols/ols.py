'''
Author: MidrarAdham
Created: Sat Aug 01 2026
'''
"""
ols.py

Ordinary least squares model for the Bayesian + OLS aggregation workflow.

This file is intentionally small.

Its job:
    Take a matrix of Bayesian posterior mean signals and estimate one
    coefficient for each column.

The main equation is:

    target_signal ≈ mean_matrix @ coefficient_by_column

In this project:
    - each row of mean_matrix is one 10-minute time window
    - each column of mean_matrix is one aggregated signal - say 10 devices
      such as expected WH ON count, expected HVAC ON count,
      one HVAC device, or one HVAC bin
    - target_signal is the feeder power signal (background)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass # avoid the __init__ and other functions
class OLSResult:
    """
    Results from one ordinary least squares fit.

    Attributes
    ----------
    coefficient_by_column:
        Estimated OLS coefficient for each column in the input mean matrix.

    predicted_signal:
        Signal predicted by the OLS model (feeder).

    residual_signal:
        Difference between the measured target signal (ground truth) and the predicted signal (predicted feeder signal).

    r_squared:
        oreditction accuracy metric.

    mean_matrix_columns:
        Column names used in the input mean matrix.
    """

    coefficient_by_column: pd.Series
    predicted_signal: pd.Series
    residual_signal: pd.Series
    r_squared: float
    mean_matrix_columns: list[str]

class OLSModel:
    """
    Small ordinary least squares solver.

    This class does not know whether the columns represent WH devices,
    HVAC devices, EVs, fixed bins, learned bins, or dynamic bins.

    It only knows:

        target_signal ≈ mean_matrix @ coefficient_by_column
    """

    def _validate_inputs (
            self,
            mean_matrix : pd.DataFrame,
            target_signal : pd.Series | np.ndarray,
    ) -> None:
        """
    Check that the OLS inputs are valid before fitting.
    """
        # Check if the mean matrix is actually a pd.Dataframe
        if not isinstance(mean_matrix, pd.DataFrame):
            raise TypeError("mean_matrix must be a pandas DataFrame.")
        # Check if teh mean matrix is empty
        if mean_matrix.empty:
            raise ValueError("mean_matrix cannot be empty.")
        # Check if the mean matrix is populated
        if mean_matrix.shape[1] == 0:
            raise ValueError("mean_matrix must contain at least one column.")
        # Ensure the target signal and the mean matrix are in the same size.
        if len(target_signal) != len(mean_matrix):
            raise ValueError(
                "mean_matrix and target_signal must have the same number "
                "of rows. "
                f"mean_matrix has {len(mean_matrix)} rows, "
                f"target_signal has {len(target_signal)} values."
            )
        # Ensure the mean matrix does not have NaNs
        if mean_matrix.isna().any().any():
            raise ValueError("mean_matrix contains missing values.")

        target_signal_series = self._convert_target_signal_to_series(
            target_signal=target_signal,
            index=mean_matrix.index,
        )
        # Ensure the target signal (background) does not have NaNs
        if target_signal_series.isna().any():
            raise ValueError("target_signal contains missing values.")

    def _convert_target_signal_to_series(
            self,
            target_signal: pd.Series | np.ndarray,
            index: pd.Index,
            ) -> pd.Series:
        """
        Convert target_signal into a pandas Series.

        This keeps the same index as mean_matrix so predicted and residual
        signals are easier to compare later.
        """
        # Python error msg does not explain this clearly, but both the background and mean matrix
        # must have the same data structure type.
        if isinstance(target_signal, pd.Series):
            return target_signal.copy()

        return pd.Series(
            data=np.asarray(target_signal, dtype=float),
            index=index,
            name="target_signal",
        )

    def _calculate_r_squared(
            self,
            target_signal: pd.Series,
            predicted_signal: pd.Series,
            ) -> float:
        """
        Calculate r_squared for the fitted signal. There is a function to do this in scipy,
        but I wanted to check why sometimes I get a negative r_squared, hence, the detailed calc.
        """
        residual_sum_of_squares = (
            (target_signal - predicted_signal) ** 2
        ).sum()

        total_sum_of_squares = (
            (target_signal - target_signal.mean()) ** 2
        ).sum()

        if total_sum_of_squares == 0:
            return float("nan")

        return 1 - residual_sum_of_squares / total_sum_of_squares

    def fit(
            self,
            mean_matrix: pd.DataFrame,
            target_signal: pd.Series | np.ndarray,
            ) -> OLSResult:
        
        """
        Fit an ordinary least squares model.

        Parameters
        ----------
        mean_matrix:
            Regression matrix built from Bayesian posterior means.

            Rows:
                10-minute time windows. This may change later.

            Columns:
                Expected ON signals. Examples:
                    wh_expected_on
                    hvac_expected_on
                    hvac_device_1
                    hvac_device_2
                    hvac_bin_0p5_1_kw

        target_signal (background):
            Measured signal to estimate.

            In this project, this is usually the transformer or feeder
            power signal after any background adjustment.

        Returns
        -------
        OLSResult
            Coefficients, predicted signal, residual signal, r_squared,
            and column names.
        """
        # Check the inputs; read more in the method _validate_inputs above
        self._validate_inputs(
            mean_matrix=mean_matrix,
            target_signal=target_signal,
        )
        # Get the columns of the mean matrix
        mean_matrix_columns = list(mean_matrix.columns)
        # easier to work witn numpy
        mean_matrix_array = mean_matrix.to_numpy(dtype=float)
        # Read the method above
        target_signal_series = self._convert_target_signal_to_series(
            target_signal=target_signal,
            index=mean_matrix.index,
        )

        target_signal_array = target_signal_series.to_numpy(dtype=float)

        coefficient_array, _, _, _ = np.linalg.lstsq(
            mean_matrix_array,
            target_signal_array,
            rcond=None,
        )
        # python requires @ when multiplying matrices
        predicted_signal_array = mean_matrix_array @ coefficient_array
        residual_signal_array = target_signal_array - predicted_signal_array
        # report the results block:
        coefficient_by_column = pd.Series(
            data=coefficient_array,
            index=mean_matrix_columns,
            name="coefficient",
        )

        predicted_signal = pd.Series(
            data=predicted_signal_array,
            index=mean_matrix.index,
            name="predicted_signal",
        )

        residual_signal = pd.Series(
            data=residual_signal_array,
            index=mean_matrix.index,
            name="residual_signal",
        )

        r_squared = self._calculate_r_squared(
            target_signal=target_signal_series,
            predicted_signal=predicted_signal,
        )

        return OLSResult(
            coefficient_by_column=coefficient_by_column,
            predicted_signal=predicted_signal,
            residual_signal=residual_signal,
            r_squared=r_squared,
            mean_matrix_columns=mean_matrix_columns,
        )