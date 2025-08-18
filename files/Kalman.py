import numpy as np
import pandas as pd

class RobustKalmanVolumeModel:
    """ Robust Kalman Filter model for intraday volume forecasting.
    Implements EM parameter estimation and Lasso-based robust filtering for outliers. """
    def __init__(self, lasso_lambda=3.0, max_iter=1000, tol=1e-4):
        """ Initialize the model.
        :param lasso_lambda: float, threshold factor (in std deviations) for Lasso-based outlier filtering.
        :param max_iter: int, maximum EM iterations for parameter estimation.
        :param tol: float, convergence tolerance for EM. """
        self.lasso_lambda = lasso_lambda
        self.max_iter = max_iter
        self.tol = tol

        # Model parameters (learned after fit)
        
        self.phi       = None     # Seasonal intraday pattern (np.array of length n_bins)
        self.a_eta     = None     # AR coefficient for daily state
        self.a_mu      = None     # AR coefficient for intraday state
        self.sigma_eta = None     # standard deviation of daily state
        self.sigma_mu  = None     # standard deviation of intraday shock
        self.r         = None     # observation noise variance

        # Internal state for filtering
        self.current_state_mean = None   # np.array([eta, mu]) current state estimate
        self.current_state_cov  = None    # np.array 2x2 current state covariance
        self.n_bins    = None     # number of intraday intervals per day
        self.last_day  = None     # Last day processed in training (date)
        self.C = np.array([1.0, 1.0])

    def fit(self, df):
        """ Fit the model to intraday log-volume data using EM algorithm.
        :param df: pandas DataFrame with 'log_amount' column (intraday log-volumes) and a DateTime index or date column
        :return: self """

        ###############################
        # Data preprocess
        ###############################

        C = self.C

        # Sigrepare data as matrix (n_days x n_bin)
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
        day_lengths = np.array([np.count_nonzero(~np.isnan(row)) for row in data_matrix])

        day_of_t = np.repeat(np.arange(N_days), day_lengths)   # shape (T,)
        bin_of_t = np.concatenate([np.arange(l) for l in day_lengths])

        # 掩码
        mask = ~np.isnan(data_matrix)                           # (N_days, n_bin)
        day_end_pos = np.cumsum(day_lengths) - 1
        day_start_pos = np.r_[0, day_end_pos[:-1] + 1]

        ###############################
        # Initial estimation
        ###############################
        
        # Initial phi: average log-volume at each intraday interval (minus overall mean for identifiability)
        phi = np.nanmean(data_matrix, axis = 0)
        phi = np.where(np.isnan(phi), 0.0, phi)  # replace NaN with 0 for missing trailing intervals
        phi = phi - np.nanmean(phi)              # center phi to zero mean

        # Initial daily effects (baseline per day)
        daily_means = np.nanmean(data_matrix - phi, axis=1)   # n_t的估计

        # a_eta和σ_eta: 用daily_means估计 (常见基准水平的持续性)
        if N_days > 1:
            a_eta = (np.cov(daily_means[:-1], daily_means[1:])[0,1] /
                     np.var(daily_means[:-1]) if np.var(daily_means[:-1]) > 1e-8 else 0.0)
            a_eta = max(min(a_eta, 0.99), -0.99)
            eta_pred = a_eta * daily_means[:-1]
            sigma_eta = np.sqrt(np.nanmean((daily_means[1:] - eta_pred)**2))
        else:
            a_eta = 0.5
            sigma_eta = 0.1
        #a_mu和sigma_mu
        all_mu_estimates = np.concatenate([
            (data_matrix[i] - phi - daily_means[i])[~np.isnan(data_matrix[i])]
            for i in range(N_days)
        ])

        a_mu     = (np.cov(all_mu_estimates[:-1], all_mu_estimates[1:])[0,1] / np.var(all_mu_estimates[:-1]) if len(all_mu_estimates) > 1 else 0.0)
        mu_pred  = a_mu * all_mu_estimates[:-1] if len(all_mu_estimates) > 1 else []
        sigma_mu = np.sqrt(np.var(all_mu_estimates[1:] - mu_pred)) if len(all_mu_estimates) > 1 else 0.1

        # Initial measurement noise variance r; variance of residual after removing phi and daily means
        total_residual_var = np.var(all_mu_estimates)
        r        = max(total_residual_var - sigma_mu**2 + 1e-9, 1e-6)

        # Kalman initial state distribution for first day
        x0       = np.array([daily_means[0], 0.0])
        eta_var0 = np.var(daily_means) if N_days >1 else sigma_eta ** 2
        mu_var0  = sigma_mu ** 2 / (1 - a_mu ** 2 + 1e-9)
        Sig0     = np.diag([eta_var0, mu_var0])
        

        ###############################
        # EM algorithm
        ###############################
        
        for it in range(self.max_iter):
        
            old_params = (a_eta, a_mu, sigma_eta, sigma_mu, r, phi.copy())
        
            # E-step: Kalman filter and smoother with current parameters
            
            max_steps = np.sum(~np.isnan(data_matrix))
            x_filt = np.zeros((max_steps, 2))               # x_filt[t]   = hat{x}_{t|t}
            Sig_filt = np.zeros((max_steps, 2, 2))          # Sig_filt[t] = Sigma_{t|t}
            x_pred = np.zeros((max_steps, 2))               # x_pred[t]   = hat{x}_{t|t-1}
            Sig_pred = np.zeros((max_steps, 2, 2))          # Sig_pred[t] = Sigma_{t|t-1}
            t_global = 0
            prev_x_f = None
            prev_Sig_f = None
        
            for d, day in enumerate(unique_days):
                # Initialize state for this day
                if d == 0:
                    x_pred_t, Sig_pred_t = x0.copy(), Sig0.copy()
                else:
                    A, Q = self._transition(False, a_eta, a_mu, sigma_eta, sigma_mu)
                    x_pred_t = A @ prev_x_f
                    Sig_pred_t = A @ prev_Sig_f @ A.T + Q
        
                day_data = data_matrix[d]
                n_i = np.count_nonzero(~np.isnan(day_data))
                # filter within day
                for j in range(n_i):
        
                    # Save prediction and predicted state
                    x_pred[t_global] = x_pred_t
                    Sig_pred[t_global] = Sig_pred_t
        
                    # Measurement update
                    obs = day_data[j]
                    # If missing (NaN), skip update (just propagate)
                    if np.isnan(obs):
                        x_filt_t, Sig_filt_t = x_pred_t, Sig_pred_t
                    else:
                        K = Sig_pred_t @ C / ( C @ Sig_pred_t @ C + r )
                        x_filt_t = x_pred_t + K * ( obs - phi[j] - C @ x_pred_t )
                        # The following is the stable Joseph form for Kalman filter
                        KH = np.outer(K, C)
                        I2 = np.eye(2)
                        Sig_filt_t = (I2 - KH) @ Sig_pred_t @ (I2 - KH).T + np.outer(K, K) * r
                        Sig_filt_t = self._sym(Sig_filt_t)
        
                    # Save filtered results
                    x_filt[t_global] = x_filt_t
                    Sig_filt[t_global] = Sig_filt_t
                    prev_x_f, prev_Sig_f = x_filt_t, Sig_filt_t
        
                    # Time update within day (intraday AR)
                    if j < n_i - 1:
                        A, Q = self._transition(True, a_eta, a_mu, sigma_eta, sigma_mu)
                        x_pred_t = A @ x_filt_t
                        Sig_pred_t = A @ Sig_filt_t @ A.T + Q
    
                    t_global += 1
    
            # ---- 后向 RTS ----
            x_post = x_filt[:t_global].copy()         # x_post[t]    = hat{x}_{t|N}, x_post[N] = x_filt[N]
            Sig_post = Sig_filt[:t_global].copy()     # Sig_post[t]  = Sigma_{t|N},  Sig_post[N] = Sig_filt[N] 
            cross_cov = np.zeros((t_global, 2, 2))    # cross_cov[t] = Sigma_{t,t-1|N} 
        
            for t in range(t_global-2, -1, -1):    # 后向遍历 T-2 … 0
                same_day = (day_of_t[t] == day_of_t[t+1])
                A, _ = self._transition(same_day, a_eta, a_mu, sigma_eta, sigma_mu)
        
                # RTS 增益
                L_t = Sig_filt[t] @ A.T @ np.linalg.pinv(Sig_pred[t+1])
        
                # 平滑均值
                x_post[t] = x_filt[t] + L_t @ (x_post[t+1] - x_pred[t+1])
        
                # 平滑方差
                Sig_post[t] = self._sym(Sig_filt[t] + L_t @ (Sig_post[t+1] - Sig_pred[t+1]) @ L_t.T)
        
                # Lag-1 协方差 Σ_{t,t+1|N}
                cross_cov[t+1] = Sig_post[t+1] @ L_t.T
            
            idx_curr = day_start_pos[1:]   # 第 2…T 天的首 bin (τ)
            idx_prev = day_end_pos[:-1]    # 对应前一天下一 bin-1 (τ-1)
            
            # (17)(18) 首状态先验更新
            x0 = x_post[0].copy()
            Sig0 = Sig_post[0].copy()
        
            # (19) a_eta (只取“每日末 bin -> 次日首 bin”)
            num_eta = np.sum(
                cross_cov[idx_curr, 0, 0] + x_post[idx_curr, 0] * x_post[idx_prev, 0]
            )
            den_eta = np.sum(
                Sig_post[idx_prev, 0, 0] + x_post[idx_prev, 0]**2
            )
            a_eta_new = num_eta / (den_eta + 1e-12)
        
            # (20) a_mu 
            num_mu = np.sum(
                cross_cov[1:, 1, 1] + x_post[1:, 1] * x_post[:-1, 1]
            )
            den_mu = np.sum(
                Sig_post[:-1, 1, 1] + x_post[:-1, 1]**2
            )
            a_mu_new = num_mu / (den_mu + 1e-12) if N_days > 1 else a_mu
    
            # (21) σ_eta
        
            # 取对应[ηη]N, Σ_{t,t-1|N}, Σ_{t-1,t-1|N} 的 nn 分量
            Sigma_eta       = Sig_post[idx_curr, 0, 0] + x_post[idx_curr, 0]**2
            Sigma_eta_m1    = Sig_post[idx_prev, 0, 0] + x_post[idx_prev, 0]**2
            Sigma_eta_cross = cross_cov[idx_curr, 0, 0]+ x_post[idx_curr, 0]*x_post[idx_prev, 0]
        
            eta_vars = (
                Sigma_eta
                + (a_eta_new**2) * Sigma_eta_m1
                - 2 * a_eta_new * Sigma_eta_cross
            )
        
            sigma_eta_new = np.sqrt(np.mean(eta_vars))
        
            # (22) σ_mu
            # version 2
            mu_vars = (
                Sig_post[1:, 1, 1] + x_post[1:, 1]**2
                + (a_mu_new**2) * (Sig_post[:-1, 1, 1] + x_post[:-1, 1]**2)
                - 2 * a_mu_new * (cross_cov[1:, 1, 1] + x_post[1:, 1] * x_post[:-1, 1])
            )
            sigma_mu_new = np.sqrt(mu_vars.mean())
        
            # (24) φ 更新
            # 把 E[ηn]+ξ[n] 迭平后再回填
            eta_mu_flat = x_post[:t_global,0] + x_post[:t_global,1]   # Len = t_global
            eta_mu      = np.full_like(data_matrix, np.nan)
            eta_mu.ravel()[mask.ravel()] = eta_mu_flat
        
            num     = np.nansum((data_matrix - eta_mu) * mask, axis=0)
            den     = np.nansum(mask, axis=0)
            phi_new = np.divide(num, den, out=np.zeros_like(num), where=den>0)
            phi_new -= phi_new.mean()
        
            # (23) r (measurement noise variance)
            err_sum = 0.0
            count_obs = 0
            for t in range(t_global):
                # Identify day and intraday index for t
                i = day_of_t[t]     # day id
                j = bin_of_t[t]     # bin id
                obs = data_matrix[i, j]
                if np.isnan(obs):
                    continue
        
                pred_mean = phi_new[j] + x_post[t,0] + x_post[t,1]
                resid_mean = obs - pred_mean
                var_sum = Sig_post[t][0,0] + Sig_post[t][1,1] + 2*Sig_post[t][0,1]
                err_sum += (resid_mean**2 + var_sum)
                count_obs += 1
            r_new = err_sum / (count_obs + 1e-12)
        
            # Assign new parameters
            a_eta, a_mu = a_eta_new, a_mu_new
            sigma_eta, sigma_mu = sigma_eta_new, sigma_mu_new
            r = max(r_new, 1e-12)
            phi_old = phi.copy()
            phi = phi_new
        
            # Check convergence
            param_changes = [
                abs(a_eta - old_params[0]), abs(a_mu - old_params[1]),
                abs(sigma_eta - old_params[2]), abs(sigma_mu - old_params[3]),
                abs(r - old_params[4])
            ]
            phi_diff = np.nanmax(np.abs(phi - old_params[5]))
            phi_diff = np.nan_to_num(phi_diff, nan=0.0)
            if phi_diff > 0:
                param_changes.append(phi_diff)
            if max(param_changes) < self.tol:
                break
    
        # Save learned parameters
        self.phi       = phi
        self.a_eta     = a_eta
        self.a_mu      = a_mu
        self.sigma_eta = sigma_eta
        self.sigma_mu  = sigma_mu
        self.r         = r
        # Set current state to last filtered state (end of training)
        self.current_state_mean = prev_x_f
        self.current_state_cov  = prev_Sig_f
        self.last_day           = unique_days[-1]
        return self
        
    def predict(self, df):
        """
        Generate 1-step-ahead predictions for new intraday data using the robust Kalman filter.
        :param df: DataFrame with 'log_amount' column (actual log-volumes).
        :return: pandas Series of predicted log-volume for each time in df.
        """
        C = self.C
        if self.phi is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = []
    
        # Determine if we are starting a new day relative to training
        first_time = df.index[0]
        first_day = first_time.date() if hasattr(first_time, 'date') else first_time
    
        # continuing same day as training ended (rare scenario)
        start_new_day = self.last_day is None or (first_day != self.last_day)
    
        # Initialize state for filtering
        if start_new_day or self.current_state_mean is None:
            # New day: use AR transition from last known state (if available) or initialize
            if self.current_state_mean is None:
                x_last = np.zeros(2)
                P_last = np.diag([self.sigma_eta**2, self.sigma_mu**2 / (1 - self.a_mu**2 + 1e-9)])  # 给一个大的协方差先验
            else:
                x_last = self.current_state_mean
                P_last = self.current_state_cov
            A, Q = self._transition(same_day = False)
            x_pred_mean = A @ x_last
            P_pred = self._sym(A @ P_last @ A.T + Q)
        else:
            # Continue from current state
            x_pred_mean = self.current_state_mean.copy()
            P_pred = self.current_state_cov.copy()
        current_date = first_day
        intraday_counter = 0
    
        # Iterate through each observation
        for time, obs in df['log_amount'].items():
            obs_date = time.date() if hasattr(time, 'date') else time
            if obs_date != current_date:
                # Day boundary encountered, propagate state
                current_date = obs_date
                intraday_counter = 0
                A, Q = self._transition(same_day = False)
                x_pred_mean = A @ self.current_state_mean
                P_pred = self._sym(A @ self.current_state_cov @ A.T + Q)
    
            # One-step prediction for current interval
            if intraday_counter >= self.n_bins:
                raise ValueError(f"interval index {intraday_counter} >= training n_bins={self.n_bins}")
    
            phi_val = self.phi[intraday_counter] if intraday_counter < len(self.phi) else self.phi[-1]
            y_hat = phi_val + C @ x_pred_mean
            preds.append(y_hat)
    
            if np.isnan(obs):
                x_filt, P_filt = x_pred_mean, P_pred
            else:
                # Measurement update with robust filtering
                S = C @ P_pred @ C.T + self.r
                K = P_pred @ C / S
                resid = obs - y_hat
                if self.lasso_lambda is not None:
                    thresh = self.lasso_lambda * np.sqrt(S)
                    resid = np.sign(resid) * max(abs(resid) - thresh, 0.0)
                # Update state estimate
                x_filt = x_pred_mean + K * resid
    
                I2 = np.eye(2)
                KH = np.outer(K, C)          # K H
                P_filt = (I2 - KH) @ P_pred @ (I2 - KH).T + np.outer(K, K) * self.r
                P_filt = self._sym(P_filt)
    
            # Prepare for next interval
            self.current_state_mean, self.current_state_cov = x_filt.copy(), P_filt.copy()
    
            # A) 日内 time-update（到下一个bin）
            if intraday_counter < self.n_bins - 1:
                A, Q = self._transition(same_day = True)   # same_day=True
                x_pred_mean = A @ x_filt
                P_pred = self._sym(A @ P_filt @ A.T + Q)
    
            intraday_counter += 1

        return pd.Series(preds, index=df.index)
    
    def get_state(self):
        """
        Get the current hidden state (eta, mu) estimate.
        :return: numpy array [eta, mu] or None if state is not set.
        """
        print('phi', self.phi)
        print('a_eta', self.a_eta)
        print('a_mu', self.a_mu)
        print('sigma_eta', self.sigma_eta)
        print('sigma_mu', self.sigma_mu)
        print('r', self.r)
        return None if self.current_state_mean is None else self.current_state_mean.copy()
    
    @staticmethod
    def _sym(A, eps=0.0):
        """
        Force_symmetry (and optionally add a small jitter on the diagonal).
        Parameters
        A : 2-D ndarray
        eps : float, optional
        - If > 0, adds eps*I for numerical stability.
        """
        A_sym = 0.5 * (A + A.T)
        if eps > 0:
            A_sym += eps * np.eye(A.shape[0])
        return A_sym

    def _transition(self,
                    same_day: bool,
                    a_eta: float = None,
                    a_mu: float = None,
                    sigma_eta: float = None,
                    sigma_mu: float = None):
        """
        根据 same_day 返回 (F, Q)。
        若在此传入 a_eta、a_mu、sigma_eta、sigma_mu，则用参数式值；否则回到 self 上的属性。
        """
        a_eta = self.a_eta if a_eta is None else a_eta
        a_mu = self.a_mu if a_mu is None else a_mu
        sigma_eta = self.sigma_eta if sigma_eta is None else sigma_eta
        sigma_mu = self.sigma_mu if sigma_mu is None else sigma_mu
    
        if same_day:  # 日内
            F = np.array([[1., 0.],
                          [0., a_mu]])
            Q = np.diag([0., sigma_mu**2])
        else:         # 跨日
            F = np.array([[a_eta, 0.],
                          [0., a_mu]])
            Q = np.diag([sigma_eta**2, sigma_mu**2])
    
        return F, Q
