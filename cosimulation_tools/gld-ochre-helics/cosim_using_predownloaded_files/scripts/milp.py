'''
Author: Midrar Adham
Created: Sat Apr 25 2026
'''
import numpy as np
import pandas as pd
import pulp


class MILP:
    """
    Uses Mixed Integer Linear Programming to estimate the individual
    power rating of each DER by finding the best assignment that
    minimizes the difference between the reconstructed and observed
    feeder demand.

    Unlike OLS which assigns one shared scaling factor per DER type
    (e.g. one kw_wh for all water heaters), MILP assigns an individual
    power rating p_i to each DER. This better handles heterogeneous
    devices like HVAC units which have different sizes and power draws.

    The optimization problem is:

        minimize:  sum over t of |feeder[t] - background - sum_i(state[i,t] * p_i)|

        subject to:
            p_min_wh   <= p_i <= p_max_wh    for each WH unit
            p_min_hvac <= p_i <= p_max_hvac  for each HVAC unit
            p_i >= 0                          for all devices

    Parameters
    ----------
    wh_histories : dict
        Output of BayesianEstimator.fit_many() for WH data.
        Keys are filenames, values are history dicts.

    hvac_histories : dict
        Output of BayesianEstimator.fit_many() for HVAC data.
        Keys are filenames, values are history dicts.

    wh_all_dfs : dict
        Output of DataLoader.all_dfs for WH data.
        Needed to build the WH state matrix.

    hvac_all_dfs : dict
        Output of DataLoader.all_dfs for HVAC data.
        Needed to build the HVAC state matrix.

    feeder_demand : pd.DataFrame
        Full house transformer data from DataLoader.load_transformer_data().
        This is the signal we are trying to decompose.

    p_min_wh : float
        Minimum power rating in Watts for a WH unit. Default: 3000 W.

    p_max_wh : float
        Maximum power rating in Watts for a WH unit. Default: 6000 W.

    p_min_hvac : float
        Minimum power rating in Watts for an HVAC unit. Default: 1000 W.

    p_max_hvac : float
        Maximum power rating in Watts for an HVAC unit. Default: 8000 W.
    """

    def __init__(self,
                 wh_histories: dict,
                 hvac_histories: dict,
                 wh_all_dfs: dict,
                 hvac_all_dfs: dict,
                 feeder_demand: pd.DataFrame,
                 p_min_wh: float = 3000.0,
                 p_max_wh: float = 6000.0,
                 p_min_hvac: float = 1000.0,
                 p_max_hvac: float = 8000.0,
                 ):
        self.wh_histories   = wh_histories
        self.hvac_histories = hvac_histories
        self.wh_all_dfs     = wh_all_dfs
        self.hvac_all_dfs   = hvac_all_dfs
        self.feeder_demand  = feeder_demand
        self.p_min_wh       = p_min_wh
        self.p_max_wh       = p_max_wh
        self.p_min_hvac     = p_min_hvac
        self.p_max_hvac     = p_max_hvac

    def _build_state_matrix(self, all_dfs: dict) -> pd.DataFrame:
        """
        Resample each DER's binary state sequence from 1-minute to
        10-minute intervals and stack into a single matrix.

        Parameters
        ----------
        all_dfs : dict
            Output of DataLoader.all_dfs for WH or HVAC data.

        Returns
        -------
        pd.DataFrame
            Shape: (144 x n_ders). Each column is one DER's resampled
            binary state sequence.
        """
        state_matrix = {}
        for filename, df in all_dfs.items():
            df['time'] = pd.to_datetime(df['time'])
            states = df.set_index('time')['state']
            states.index = pd.to_datetime(states.index)
            states = states.resample('10min').mean()
            state_matrix[filename] = states.values
        return pd.DataFrame(state_matrix)

    def _estimate_background(self, wh_state_matrix: pd.DataFrame) -> float:
        """
        Estimate the background constant as the mean feeder demand
        during chunks where all WH units are simultaneously OFF.

        Parameters
        ----------
        wh_state_matrix : pd.DataFrame
            Output of _build_state_matrix() for WH data.

        Returns
        -------
        float
            Estimated background demand in Watts.
        """
        wh_all_off = (wh_state_matrix.sum(axis=1) == 0)
        wh_off_indices = np.where(wh_all_off.values)[0]
        background = self.feeder_demand['power_out'].values[wh_off_indices].mean()
        return background

    def _build_mean_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the Bayesian posterior mean ON-probabilities for each
        DER and organize into a (144 x n_ders) matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many().

        Returns
        -------
        pd.DataFrame
            Shape: (144 x n_ders). Each value is the posterior mean
            ON-probability for that DER at that chunk.
        """
        mean_matrix = {}
        for filename, history in histories.items():
            mean_matrix[filename] = history['mean']
        return pd.DataFrame(mean_matrix)

    def _solve(self,
               wh_mean_matrix: pd.DataFrame,
               hvac_mean_matrix: pd.DataFrame,
               background: float
               ) -> dict:
        """
        Formulate and solve the MILP problem to find the best individual
        power rating for each DER.

        The objective minimizes the absolute error between the observed
        feeder demand (minus background) and the reconstructed demand
        from all DERs:

            minimize:  sum_t of e_t

            subject to:
                e_t >= feeder[t] - background - sum_i(mean[i,t] * p_i)
                e_t >= -(feeder[t] - background - sum_i(mean[i,t] * p_i))
                p_min_wh   <= p_i <= p_max_wh    for WH units
                p_min_hvac <= p_i <= p_max_hvac  for HVAC units
                e_t >= 0

        Note: the absolute value |x| is linearized using two constraints
        and a slack variable e_t, which is standard practice in MILP.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.

        background : float
            Estimated background constant in Watts.

        Returns
        -------
        dict
            Contains:
            - 'wh_powers'        : dict of {filename: estimated power rating}
            - 'hvac_powers'      : dict of {filename: estimated power rating}
            - 'wh_predicted'     : np.ndarray, predicted WH demand per chunk
            - 'hvac_predicted'   : np.ndarray, predicted HVAC demand per chunk
            - 'combined_predicted': np.ndarray, total predicted demand per chunk
            - 'status'           : str, solver status ('Optimal' if successful)
        """
        num_chunks   = len(wh_mean_matrix)
        wh_devices   = list(wh_mean_matrix.columns)
        hvac_devices = list(hvac_mean_matrix.columns)

        feeder = self.feeder_demand['power_out'].values

        # ── Define the problem ────────────────────────────────────────
        problem = pulp.LpProblem('DER_Demand_Disaggregation', pulp.LpMinimize)

        # ── Decision variables ────────────────────────────────────────
        # Power rating for each WH unit
        p_wh = {
            dev: pulp.LpVariable(f'p_wh_{i}',
                                 lowBound=self.p_min_wh,
                                 upBound=self.p_max_wh)
            for i, dev in enumerate(wh_devices)
        }

        # Power rating for each HVAC unit
        p_hvac = {
            dev: pulp.LpVariable(f'p_hvac_{i}',
                                 lowBound=self.p_min_hvac,
                                 upBound=self.p_max_hvac)
            for i, dev in enumerate(hvac_devices)
        }

        # Slack variables for absolute value linearization — one per chunk
        e = {
            t: pulp.LpVariable(f'e_{t}', lowBound=0)
            for t in range(num_chunks)
        }

        # ── Objective: minimize total absolute error ──────────────────
        problem += pulp.lpSum(e[t] for t in range(num_chunks))

        # ── Constraints ───────────────────────────────────────────────
        for t in range(num_chunks):

            # Reconstructed demand at chunk t
            reconstructed = (
                pulp.lpSum(
                    wh_mean_matrix.iloc[t][dev] * p_wh[dev]
                    for dev in wh_devices
                ) +
                pulp.lpSum(
                    hvac_mean_matrix.iloc[t][dev] * p_hvac[dev]
                    for dev in hvac_devices
                )
            )

            residual = feeder[t] - background

            # Linearized absolute value: e_t >= residual - reconstructed
            problem += e[t] >= residual - reconstructed
            # And:                       e_t >= reconstructed - residual
            problem += e[t] >= reconstructed - residual

        # ── Solve ─────────────────────────────────────────────────────
        # Use CBC solver (bundled with pulp), suppress output
        solver = pulp.PULP_CBC_CMD(msg=0)
        problem.solve(solver)

        status = pulp.LpStatus[problem.status]
        print(f"MILP solver status: {status}")

        # ── Extract results ───────────────────────────────────────────
        wh_powers   = {dev: pulp.value(p_wh[dev])   for dev in wh_devices}
        hvac_powers = {dev: pulp.value(p_hvac[dev]) for dev in hvac_devices}

        print(f"Estimated WH power ratings (W):")
        for dev, power in wh_powers.items():
            print(f"  {dev}: {power:.1f} W")

        print(f"Estimated HVAC power ratings (W):")
        for dev, power in hvac_powers.items():
            print(f"  {dev}: {power:.1f} W")

        # ── Reconstruct demand time series ────────────────────────────
        wh_predicted   = np.zeros(num_chunks)
        hvac_predicted = np.zeros(num_chunks)

        for t in range(num_chunks):
            wh_predicted[t] = sum(
                wh_mean_matrix.iloc[t][dev] * wh_powers[dev]
                for dev in wh_devices
            )
            hvac_predicted[t] = sum(
                hvac_mean_matrix.iloc[t][dev] * hvac_powers[dev]
                for dev in hvac_devices
            )

        return {
            'wh_powers':          wh_powers,
            'hvac_powers':        hvac_powers,
            'wh_predicted':       wh_predicted,
            'hvac_predicted':     hvac_predicted,
            'combined_predicted': wh_predicted + hvac_predicted,
            'status':             status,
        }

    def run(self) -> dict:
        """
        Run the full MILP disaggregation pipeline.

        Steps:
            1. Build WH and HVAC state matrices
            2. Estimate background demand
            3. Build Bayesian mean matrices
            4. Solve the MILP problem
            5. Return predictions and estimated power ratings

        Returns
        -------
        dict
            Contains:
            - 'wh_predicted'      : np.ndarray, predicted WH demand
            - 'hvac_predicted'    : np.ndarray, predicted HVAC demand
            - 'combined_predicted': np.ndarray, total predicted demand
            - 'wh_powers'         : dict of estimated WH power ratings
            - 'hvac_powers'       : dict of estimated HVAC power ratings
            - 'background'        : float, estimated background constant
            - 'status'            : str, MILP solver status
        """
        # Step 1 - build state matrices
        wh_state_matrix = self._build_state_matrix(all_dfs=self.wh_all_dfs)

        # Step 2 - estimate background from WH-off chunks
        background = self._estimate_background(wh_state_matrix=wh_state_matrix)
        print(f"Background constant: {background:.1f} W")

        # Step 3 - build mean matrices from Bayesian histories
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # Step 4 - solve MILP
        results = self._solve(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            background=background,
        )

        # Step 5 - return everything
        return {
            'wh_predicted':       results['wh_predicted'],
            'hvac_predicted':     results['hvac_predicted'],
            'combined_predicted': results['combined_predicted'],
            'wh_powers':          results['wh_powers'],
            'hvac_powers':        results['hvac_powers'],
            'background':         background,
            'status':             results['status'],
        }