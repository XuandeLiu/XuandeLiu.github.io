import numpy as np
import pandas as pd

class RobustKalmanVolumeModel:
    """
    Robust Kalman Filter model for intraday volume forecasting.
    Implements EM parameter estimation and Lasso-based robust filtering for outliers.
    """
    def __init__(self, lasso_lambda=3.0, max_iter=1000, tol=1e-4):
        """
        Initialize the model.
        :param lasso_lambda: float, threshold factor (in std deviations) for Lasso-based outlier filtering.
        :param max_iter: int, maximum EM iterations for parameter estimation.
        :param tol: float, convergence tolerance for EM.
        """
        self.lasso_lambda = lasso_lambda
        self.max_iter = max_iter
        self.tol = tol
        # Model parameters (learned after fit)
        self.phi = None          # seasonal intraday pattern (np.array of length n_bins)
        self.a_eta = None        # AR coefficient for intraday state
        self.a_mu = None         # AR coefficient for daily state
        self.sigma_eta = None    # standard deviation of intraday shock
        self.sigma_mu = None     # standard deviation of daily shock
        self.r = None            # observation noise variance
        # Internal state for filtering
        self.current_state_mean = None  # np.array([eta, mu]) current state estimate
        self.current_state_cov = None   # np.array 2x2 current state covariance
        self.n_bins = None       # number of intraday intervals per day
        self.last_day = None     # last day processed in training (date)
    
    def fit(self, df):
        """
        Fit the model to intraday log-volume data using EM algorithm.
        :param df: pandas DataFrame with 'log_amount' column (intraday log-volumes) and a DateTime index or separate date indicator.
        :return: self
        """
        # Prepare data as matrix (n_days x n_bin)
        # Determine grouping by day
        if df.index.dtype == 'datetime64[ns]' or isinstance(df.index, pd.DatetimeIndex):
            # Use index date part for grouping
            df = df.copy()
            df['date'] = df.index.date
            grouped = df.groupby('date')['log_amount']
        elif 'date' in df.columns:
            grouped = df.groupby('date')['log_amount']
        else:
            # If no explicit date, treat entire sequence as one day
            df = df.copy()
            df['__day__'] = 0
            grouped = df.groupby('__day__')['log_amount']
        unique_days = list(grouped.groups.keys())
        N_days = len(unique_days)
        # Determine number of intraday intervals per day (assume constant or use maximum)
        day_lengths = [len(grouped.get_group(day)) for day in unique_days]
        self.n_bins = max(day_lengths)
        n_bins = self.n_bins
        # Build data matrix (pad missing intervals with NaN if days have different lengths)
        data_matrix = np.full((N_days, n_bins), np.nan)
        for i, day in enumerate(unique_days):
            vals = grouped.get_group(day).values
            data_matrix[i, :len(vals)] = vals
        # Initial parameter estimates
        # Initial phi: average log-volume at each intraday interval (minus overall mean for identifiability)
        phi = np.nanmean(data_matrix, axis=0)
        phi = np.where(np.isnan(phi), 0.0, phi)  # replace NaN with 0 for missing trailing intervals
        phi = phi - np.nanmean(phi)  # center phi to zero mean
        # Initial daily effects (baseline per day)
        daily_means = np.nanmean(data_matrix - phi, axis=1)
        daily_means = np.where(np.isnan(daily_means), 0.0, daily_means)
        # Initial a_mu: autocorrelation of daily baseline
        if N_days > 1:
            mu_series = daily_means
            a_mu = (np.cov(mu_series[:-1], mu_series[1:])[0, 1] / np.var(mu_series[:-1]) 
                    if np.var(mu_series[:-1]) > 1e-8 else 0.0)
            a_mu = max(min(a_mu, 0.99), -0.99)
        else:
            a_mu = 0.0
        # Initial sigma_mu: std of daily effect innovations
        if N_days > 1:
            mu_pred = a_mu * daily_means[:-1]
            sigma_mu = np.sqrt(np.nanmean((daily_means[1:] - mu_pred)**2))
        else:
            sigma_mu = 0.1
        # Initial a_eta: assume some persistence within day
        a_eta = 0.5
        # Initial sigma_eta: std of intraday shock (use variance of first-interval deviations)
        if N_days > 0:
            first_int = data_matrix[:, 0] - (phi[0] + daily_means)
            first_int = first_int[~np.isnan(first_int)]
            sigma_eta = np.sqrt(max(np.var(first_int) - 1e-3, 1e-6)) if first_int.size > 0 else 0.1
        else:
            sigma_eta = 0.1
        # Initial measurement noise variance r: variance of residual after removing phi and daily means
        residuals = data_matrix - (daily_means[:, None] + phi[None, :])
        r = np.nanmean((residuals)**2)
        if not np.isfinite(r) or r <= 0:
            r = 1e-2
        # Kalman initial state distribution for first day
        x0 = np.array([0.0, daily_means[0] if N_days > 0 else 0.0])
        # P0: intraday state var = sigma_eta^2, daily state var = var of daily_means or a large number
        mu_var0 = np.var(daily_means) if N_days > 1 else (sigma_mu**2 / (1 - a_mu**2 + 1e-9))
        P0 = np.diag([sigma_eta**2, mu_var0])
        # EM algorithm
        for it in range(self.max_iter):
            old_params = (a_eta, a_mu, sigma_eta, sigma_mu, r, phi.copy())
            # E-step: Kalman filter and smoother with current parameters
            # Storage for filtered stats
            total_steps = 0
            # Placeholders for filter results (size: sum of day_lengths)
            # (Allocate arrays conservatively with NaNs to accommodate all steps)
            max_steps = np.sum(~np.isnan(data_matrix))
            x_filt = np.zeros((max_steps, 2))
            P_filt = np.zeros((max_steps, 2, 2))
            x_pred = np.zeros((max_steps, 2))
            P_pred = np.zeros((max_steps, 2, 2))
            # Filter through each day
            t_global = 0
            prev_x_f = None
            prev_P_f = None
            for i, day in enumerate(unique_days):
                # Initialize state for this day
                if i == 0:
                    x_pred_t = x0.copy()
                    P_pred_t = P0.copy()
                else:
                    # Transition from previous day (intraday reset + daily AR)
                    mu_last = prev_x_f[1]
                    mu_var_last = prev_P_f[1,1]
                    x_pred_t = np.array([0.0, a_mu * mu_last])
                    P_pred_t = np.array([[sigma_eta**2, 0.0],
                                         [0.0, a_mu**2 * mu_var_last + sigma_mu**2]])
                day_data = data_matrix[i]
                n_i = np.count_nonzero(~np.isnan(day_data))
                # filter within day
                for j in range(n_i):
                    # Observation prediction
                    y_hat = phi[j] + x_pred_t[0] + x_pred_t[1]
                    # Save prediction and predicted state
                    x_pred[t_global] = x_pred_t
                    P_pred[t_global] = P_pred_t
                    # Measurement update
                    obs_val = day_data[j]
                    # If missing (NaN), skip update (just propagate)
                    if np.isnan(obs_val):
                        x_f = x_pred_t
                        P_f = P_pred_t
                    else:
                        resid = obs_val - y_hat
                        S = P_pred_t[0,0] + P_pred_t[1,1] + 2*P_pred_t[0,1] + r
                        K = P_pred_t.dot(np.array([1.0, 1.0])) / S  # Kalman gain (2,)
                        x_f = x_pred_t + K * resid
                        P_f = P_pred_t - np.outer(K, np.array([1.0, 1.0]).dot(P_pred_t))
                    # Save filtered results
                    x_filt[t_global] = x_f
                    P_filt[t_global] = P_f
                    prev_x_f, prev_P_f = x_f, P_f
                    t_global += 1
                    # Time update within day (intraday AR)
                    if j < n_i - 1:
                        x_pred_t = np.array([a_eta * x_f[0], x_f[1]])
                        P_pred_t = np.array([[a_eta**2 * P_f[0,0] + 0.0, a_eta * P_f[0,1]],
                                             [P_f[1,0] * a_eta, P_f[1,1] + 0.0]])
                        # (Note: Q within-day = 0, so process noise omitted)
                # End of day filter
            total_steps = t_global
            # Backward smoother
            sm_mean = x_filt[:total_steps].copy()
            sm_cov = P_filt[:total_steps].copy()
            # Compute prefix indices for each day (to identify boundaries)
            prefixes = [0] * N_days
            for i in range(1, N_days):
                prefixes[i] = prefixes[i-1] + np.count_nonzero(~np.isnan(data_matrix[i-1]))
            # Rauch-Tung-Striebel smoothing
            for t in range(total_steps-2, -1, -1):
                # Determine if transition t->t+1 crosses a day boundary
                # Find day index for t and t+1
                day_t = next(di for di in range(N_days) if prefixes[di] <= t < prefixes[di] + np.count_nonzero(~np.isnan(data_matrix[di])))
                day_tp1 = next(di for di in range(N_days) if prefixes[di] <= t+1 < prefixes[di] + np.count_nonzero(~np.isnan(data_matrix[di])))
                if day_tp1 != day_t:
                    F_t = np.array([[0.0, 0.0],[0.0, a_mu]])
                else:
                    F_t = np.array([[a_eta, 0.0],[0.0, 1.0]])
                P_pred_next = P_pred[t+1]
                # Compute smoother gain
                inv_P_pred = np.linalg.pinv(P_pred_next)
                J = P_filt[t].dot(F_t.T).dot(inv_P_pred)
                # Smoothed state
                sm_mean[t] = x_filt[t] + J.dot(sm_mean[t+1] - x_pred[t+1])
                sm_cov[t] = P_filt[t] + J.dot(sm_cov[t+1] - P_pred_next).dot(J.T)
            # M-step: update parameters using smoothed expectations
            # Update phi
            phi_new = phi.copy()
            for j in range(n_bins):
                vals = []
                for i in range(N_days):
                    if j < np.count_nonzero(~np.isnan(data_matrix[i])):
                        idx = prefixes[i] + j
                        obs_val = data_matrix[i, j]
                        if np.isnan(obs_val): 
                            continue
                        # E(eta+mu) = sm_mean[idx,0] + sm_mean[idx,1]
                        vals.append(obs_val - (sm_mean[idx,0] + sm_mean[idx,1]))
                if vals:
                    phi_new[j] = np.mean(vals)
            phi_new = phi_new - np.nanmean(phi_new)
            # Update a_eta
            num_eta = 0.0
            den_eta = 0.0
            for i in range(N_days):
                n_i = np.count_nonzero(~np.isnan(data_matrix[i]))
                start_idx = prefixes[i]
                for j in range(n_i - 1):
                    t = start_idx + j
                    tp1 = t + 1
                    # E(eta_t eta_{t+1}) = Cov(eta_t,eta_{t+1}) + E(eta_t)E(eta_{t+1})
                    # Approximate Cov by old a_eta * Var(eta_t) (no intraday noise)
                    cov_eta = a_eta * sm_cov[t][0,0]
                    num_eta += cov_eta + sm_mean[t,0] * sm_mean[tp1,0]
                    den_eta += sm_cov[t][0,0] + sm_mean[t,0]**2
            a_eta_new = num_eta / (den_eta + 1e-12)
            # Update a_mu
            num_mu = 0.0
            den_mu = 0.0
            for i in range(N_days - 1):
                end_idx = prefixes[i] + np.count_nonzero(~np.isnan(data_matrix[i])) - 1
                next_idx = prefixes[i+1]
                # Cov(mu_end, mu_next) ~ a_mu * Var(mu_end)
                cov_mu = a_mu * sm_cov[end_idx][1,1]
                num_mu += cov_mu + sm_mean[end_idx,1] * sm_mean[next_idx,1]
                den_mu += sm_cov[end_idx][1,1] + sm_mean[end_idx,1]**2
            a_mu_new = num_mu / (den_mu + 1e-12) if N_days > 1 else a_mu
            # Update sigma_eta (based on intraday initial shocks)
            eta_start_vars = []
            for i in range(N_days):
                if np.count_nonzero(~np.isnan(data_matrix[i])) > 0:
                    idx0 = prefixes[i]
                    eta_start_vars.append(sm_cov[idx0][0,0] + sm_mean[idx0,0]**2)
            sigma_eta_new = np.sqrt(np.mean(eta_start_vars)) if eta_start_vars else sigma_eta
            # Update sigma_mu (daily process noise)
            mu_res_vars = []
            for i in range(N_days - 1):
                end_idx = prefixes[i] + np.count_nonzero(~np.isnan(data_matrix[i])) - 1
                next_idx = prefixes[i+1]
                mu_end_sq = sm_cov[end_idx][1,1] + sm_mean[end_idx,1]**2
                mu_next_sq = sm_cov[next_idx][1,1] + sm_mean[next_idx,1]**2
                # E(mu_next mu_end)
                cov_mu = a_mu * sm_cov[end_idx][1,1]
                mu_cross = cov_mu + sm_mean[end_idx,1] * sm_mean[next_idx,1]
                mu_res_vars.append(mu_next_sq - 2 * a_mu_new * mu_cross + a_mu_new**2 * mu_end_sq)
            sigma_mu_new = np.sqrt(np.mean(mu_res_vars)) if mu_res_vars else sigma_mu
            # Update r (measurement noise variance)
            err_sum = 0.0
            count_obs = 0
            for t in range(total_steps):
                # Identify day and intraday index for t
                i = next(di for di in range(N_days) if prefixes[di] <= t < prefixes[di] + np.count_nonzero(~np.isnan(data_matrix[di])))
                j = t - prefixes[i]
                obs_val = data_matrix[i, j]
                if np.isnan(obs_val):
                    continue
                pred_mean = phi_new[j] + sm_mean[t,0] + sm_mean[t,1]
                resid_mean = obs_val - pred_mean
                var_sum = sm_cov[t][0,0] + sm_cov[t][1,1] + 2*sm_cov[t][0,1]
                err_sum += (resid_mean**2 + var_sum)
                count_obs += 1
            r_new = err_sum / (count_obs + 1e-12)
            # Assign new parameters
            a_eta, a_mu = a_eta_new, a_mu_new
            sigma_eta, sigma_mu = sigma_eta_new, sigma_mu_new
            r = max(r_new, 1e-12)
            phi = phi_new
            # Check convergence
            param_changes = [
                abs(a_eta - old_params[0]), abs(a_mu - old_params[1]),
                abs(sigma_eta - old_params[2]), abs(sigma_mu - old_params[3]),
                abs(r - old_params[4])
            ]
            phi_diff = np.nanmax(np.abs(phi - old_params[5]))
            if phi_diff > 0:
                param_changes.append(phi_diff)
            if max(param_changes) < self.tol:
                break
        # Save learned parameters
        self.phi = phi
        self.a_eta = a_eta
        self.a_mu = a_mu
        self.sigma_eta = sigma_eta
        self.sigma_mu = sigma_mu
        self.r = r
        # Set current state to last filtered state (end of training)
        self.current_state_mean = prev_x_f
        self.current_state_cov = prev_P_f
        self.last_day = unique_days[-1]
        return self
    
    def predict(self, df):
        """
        Generate 1-step-ahead predictions for new intraday data using the robust Kalman filter.
        :param df: DataFrame with 'log_amount' column (actual log-volumes).
        :return: pandas Series of predicted log-volume for each time in df.
        """
        if self.phi is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = []
        # Determine if we are starting a new day relative to training
        first_idx = df.index[0]
        first_day = first_idx.date() if hasattr(first_idx, 'date') else first_idx
        start_new_day = True
        if self.last_day is not None and first_day == self.last_day:
            # continuing same day as training ended (rare scenario)
            start_new_day = False
        # Initialize state for filtering
        if start_new_day or self.current_state_mean is None:
            # New day: use AR transition from last known state (if available) or initialize
            if self.current_state_mean is not None:
                mu_last = self.current_state_mean[1]
                mu_var_last = self.current_state_cov[1,1]
            else:
                mu_last = 0.0
                mu_var_last = self.sigma_mu**2 / (1 - self.a_mu**2 + 1e-9)
            x_pred_mean = np.array([0.0, self.a_mu * mu_last])
            P_pred = np.array([[self.sigma_eta**2, 0.0],
                               [0.0, self.a_mu**2 * mu_var_last + self.sigma_mu**2]])
        else:
            # Continue from current state
            x_pred_mean = self.current_state_mean.copy()
            P_pred = self.current_state_cov.copy()
        current_date = first_day
        intraday_counter = 0
        # Iterate through each observation
        for time, obs in df['log_amount'].iteritems():
            obs_date = time.date() if hasattr(time, 'date') else time
            if obs_date != current_date:
                # Day boundary encountered, propagate state
                current_date = obs_date
                intraday_counter = 0
                mu_last = x_pred_mean[1]
                mu_var_last = P_pred[1,1]
                x_pred_mean = np.array([0.0, self.a_mu * mu_last])
                P_pred = np.array([[self.sigma_eta**2, 0.0],
                                   [0.0, self.a_mu**2 * mu_var_last + self.sigma_mu**2]])
            # One-step prediction for current interval
            phi_val = self.phi[intraday_counter] if intraday_counter < len(self.phi) else self.phi[-1]
            y_hat = phi_val + x_pred_mean[0] + x_pred_mean[1]
            preds.append(y_hat)
            # Measurement update with robust filtering
            resid = obs - y_hat
            S = P_pred[0,0] + P_pred[1,1] + 2*P_pred[0,1] + self.r
            H_vec = np.array([1.0, 1.0])
            K = P_pred.dot(H_vec) / S  # Kalman gain (shape 2,)
            # Lasso-based residual adjustment
            resid_eff = resid
            if self.lasso_lambda is not None:
                thresh = self.lasso_lambda * np.sqrt(S)
                if abs(resid) > thresh:
                    e = resid - np.sign(resid) * thresh
                else:
                    e = 0.0
                resid_eff = resid - e
            # Update state estimate
            x_filt = x_pred_mean + K * resid_eff
            P_filt = P_pred - np.outer(K, H_vec.dot(P_pred))
            # Prepare for next interval
            self.current_state_mean = x_filt.copy()
            self.current_state_cov = P_filt.copy()
            if intraday_counter < self.n_bins - 1:
                # propagate within day
                x_pred_mean = np.array([self.a_eta * x_filt[0], x_filt[1]])
                P_pred = np.array([[self.a_eta**2 * P_filt[0,0], self.a_eta * P_filt[0,1]],
                                   [self.a_eta * P_filt[1,0], P_filt[1,1]]])
            intraday_counter += 1
        return pd.Series(preds, index=df.index)
    
    def get_state(self):
        """
        Get the current hidden state (eta, mu) estimate.
        :return: numpy array [eta, mu] or None if state is not set.
        """
        return None if self.current_state_mean is None else self.current_state_mean.copy()