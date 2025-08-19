class FixedPointVolumeModel:
    def __init__(self, max_iter=100, tol=1e-6, clip=True, use_day5=True):
        """
        初始化模型参数。
        max_iter: 最大迭代次数
        tol: 收敛容忍度（mu和S变化的阈值）
        clip: 是否将预测结果裁剪为非负
        use_day5: 是否启用第5日滞后残差项
        """
        self.max_iter = max_iter
        self.tol = tol
        self.clip = clip
        self.use_day5 = use_day5
        # 模型拟合后的状态
        self.mu = None             # 日内形状 (pandas Series)
        self.S_train = None        # 训练集各日的S_d (pandas Series)
        self.coefficients = None   # 残差回归系数 (pandas DataFrame)
        self.resid_std = None      # 残差标准差 (pandas Series)
        self.train_residuals = None# 训练集残差矩阵 (pandas DataFrame)
        self._fitted = False
        self.converged_ = None
        self.iterations_ = None

    def fit(self, train_df):
        # 检查输入列
        required_cols = {'date', 'interval', 'amount'}
        if not required_cols.issubset(train_df.columns):
            raise ValueError(f"train_df must contain columns {required_cols}")
        # 按日期和时间段排序
        train_df = train_df.sort_values(['date', 'interval']).copy()
        # 构建日期×时间段的成交量矩阵
        volume_table = train_df.pivot(index='date', columns='interval', values='amount').fillna(0)
        A = volume_table.values  # 实际成交量矩阵，形状 (D天数, I时间段数)
        D, I = A.shape
        dates = volume_table.index

        # 初始化 mu 和 S
        daily_totals = A.sum(axis=1)
        # 计算各日相对分布并取均值作为初始 mu
        safe_totals = np.where(daily_totals == 0, 1, daily_totals)  # 避免除以0
        fractions = A / safe_totals[:, None]  # 每日各时段占比
        mu = fractions.mean(axis=0)
        if mu.sum() == 0:
            mu = np.full(I, 1.0/I)
        else:
            mu = mu / mu.sum()
        # 初始 S 设为各日实际总量
        S = daily_totals.copy()

        # 准备回归系数容器
        intercepts = np.zeros(I)
        coef_prev_int = np.zeros(I)
        coef_prev_day = np.zeros(I)
        coef_prev_5d = np.zeros(I)

        converged = False
        n_iter = 0
        # 迭代求解
        for it in range(self.max_iter):
            n_iter = it + 1
            # 计算残差 e = A - S * mu
            predicted = S[:, None] * mu[None, :]  # 外积计算预测成交量矩阵
            e = A - predicted

            # 分时段回归残差
            intercepts.fill(0.0); coef_prev_int.fill(0.0)
            coef_prev_day.fill(0.0); coef_prev_5d.fill(0.0)
            for j in range(I):
                if j == 0:
                    # i=1: e_{d,1} ~ const + e_{d-1,1} + (e_{d-5,1} if use_day5)
                    start_idx = 5 if self.use_day5 else 1
                    if D <= start_idx:
                        continue
                    Y = e[start_idx:, j]
                    X_list = [np.ones_like(Y)]
                    prev_day_vals = e[(start_idx-1):(D-1), j]
                    X_list.append(prev_day_vals)
                    if self.use_day5:
                        prev5_vals = e[(start_idx-5):(D-5), j]
                        X_list.append(prev5_vals)
                    X = np.vstack(X_list).T
                else:
                    # i>1: e_{d,i} ~ const + e_{d,i-1} + e_{d-1,i} + (e_{d-5,i} if use_day5)
                    start_idx = 5 if self.use_day5 else 1
                    if D <= start_idx:
                        continue
                    Y = e[start_idx:, j]
                    X_list = [np.ones_like(Y)]
                    prev_int_vals = e[start_idx:, j-1]
                    X_list.append(prev_int_vals)
                    prev_day_vals = e[(start_idx-1):(D-1), j]
                    X_list.append(prev_day_vals)
                    if self.use_day5:
                        prev5_vals = e[(start_idx-5):(D-5), j]
                        X_list.append(prev5_vals)
                    X = np.vstack(X_list).T
                # 最小二乘求解回归系数
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                intercepts[j] = beta[0] if len(beta) > 0 else 0.0
                if j == 0:
                    # i=1 无前一时段项
                    coef_prev_int[j] = 0.0
                    coef_prev_day[j] = beta[1] if len(beta) > 1 else 0.0
                    if self.use_day5:
                        coef_prev_5d[j] = beta[2] if len(beta) > 2 else 0.0
                else:
                    coef_prev_int[j] = beta[1] if len(beta) > 1 else 0.0
                    coef_prev_day[j] = beta[2] if len(beta) > 2 else 0.0
                    if self.use_day5:
                        coef_prev_5d[j] = beta[3] if len(beta) > 3 else 0.0

            # 更新 S 和 mu
            mu_sq_sum = np.sum(mu**2) or 1e-12  # 避免除零
            S_new = A.dot(mu) / mu_sq_sum
            S_new = np.where(S_new < 0, 0.0, S_new)
            S_sq_sum = np.sum(S_new**2) or 1e-12
            mu_new = A.T.dot(S_new) / S_sq_sum
            mu_new = np.where(mu_new < 0, 0.0, mu_new)
            # 归一化 mu_new 并相应调整 S_new
            sum_mu_new = mu_new.sum() or 1e-12
            mu_new /= sum_mu_new
            S_new *= sum_mu_new

            # 检查收敛
            if np.max(np.abs(mu_new - mu)) < self.tol and np.max(np.abs(S_new - S)) < self.tol:
                mu = mu_new; S = S_new; converged = True
                break
            mu, S = mu_new, S_new

        # 最终残差及系数（保证与最终 mu,S 一致）
        predicted_final = S[:, None] * mu[None, :]
        e_final = A - predicted_final
        intercepts.fill(0.0); coef_prev_int.fill(0.0)
        coef_prev_day.fill(0.0); coef_prev_5d.fill(0.0)
        for j in range(I):
            if j == 0:
                start_idx = 5 if self.use_day5 else 1
                if D <= start_idx:
                    continue
                Y = e_final[start_idx:, j]
                X_list = [np.ones_like(Y)]
                prev_day_vals = e_final[(start_idx-1):(D-1), j]
                X_list.append(prev_day_vals)
                if self.use_day5:
                    prev5_vals = e_final[(start_idx-5):(D-5), j]
                    X_list.append(prev5_vals)
                X = np.vstack(X_list).T
            else:
                start_idx = 5 if self.use_day5 else 1
                if D <= start_idx:
                    continue
                Y = e_final[start_idx:, j]
                X_list = [np.ones_like(Y)]
                prev_int_vals = e_final[start_idx:, j-1]
                X_list.append(prev_int_vals)
                prev_day_vals = e_final[(start_idx-1):(D-1), j]
                X_list.append(prev_day_vals)
                if self.use_day5:
                    prev5_vals = e_final[(start_idx-5):(D-5), j]
                    X_list.append(prev5_vals)
                X = np.vstack(X_list).T
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            intercepts[j] = beta[0] if len(beta) > 0 else 0.0
            if j == 0:
                coef_prev_int[j] = 0.0
                coef_prev_day[j] = beta[1] if len(beta) > 1 else 0.0
                if self.use_day5:
                    coef_prev_5d[j] = beta[2] if len(beta) > 2 else 0.0
            else:
                coef_prev_int[j] = beta[1] if len(beta) > 1 else 0.0
                coef_prev_day[j] = beta[2] if len(beta) > 2 else 0.0
                if self.use_day5:
                    coef_prev_5d[j] = beta[3] if len(beta) > 3 else 0.0

        # 保存模型状态
        self.mu = pd.Series(mu, index=volume_table.columns, name='mu')
        self.S_train = pd.Series(S, index=dates, name='S')
        coef_data = {'const': intercepts,
                     'coef_prev_interval': coef_prev_int,
                     'coef_prev_day': coef_prev_day}
        if self.use_day5:
            coef_data['coef_prev_5day'] = coef_prev_5d
        self.coefficients = pd.DataFrame(coef_data, index=volume_table.columns)
        resid_std_vals = e_final.std(axis=0, ddof=0)
        self.resid_std = pd.Series(resid_std_vals, index=volume_table.columns, name='resid_std')
        # 保存训练残差用于预测起点
        self.train_residuals = pd.DataFrame(e_final, index=dates, columns=volume_table.columns)
        self._fitted = True
        self.converged_ = converged
        self.iterations_ = n_iter
        return self

    def predict(self, test_df):
        """
        利用已训练模型预测新的交易日成交量。
        test_df 应包含 'date' 和 'interval' 列（'amount'列将被忽略）。
        返回包含 'date', 'interval', 'predicted' 列的 DataFrame。
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")
        required_cols = {'date', 'interval'}
        if not required_cols.issubset(test_df.columns):
            raise ValueError(f"test_df must contain at least {required_cols}")
        test_df = test_df.sort_values(['date', 'interval']).copy()
        test_dates = test_df['date'].unique()

        # 建立 mu 映射方便按interval取值
        mu_map = {interval: val for interval, val in zip(self.mu.index, self.mu.values)}
        # 构造包含训练+测试日期的时间轴索引，用于残差引用
        train_dates = list(self.train_residuals.index)
        all_dates = sorted(set(train_dates) | set(test_dates))
        date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
        total_days = len(all_dates)
        I = len(self.mu)

        # 合并残差数组和 S 数组
        e_all = np.zeros((total_days, I))
        for d in train_dates:
            idx = date_to_idx[d]
            e_all[idx, :] = self.train_residuals.loc[d].values
        S_all = np.zeros(total_days)
        for d in train_dates:
            idx = date_to_idx[d]
            S_all[idx] = self.S_train.loc[d]

        predictions = []  # 存储预测结果
        for date in test_dates:
            idx = date_to_idx[date]
            # 取前一日的 S 作为基准
            S_base = S_all[idx-1] if idx - 1 >= 0 else 0.0
            e_pred = np.zeros(I)
            # 获取该日所有间隔（假设与训练时段相同）
            intervals = sorted(test_df[test_df['date'] == date]['interval'].unique())
            for interval in intervals:
                # 找到 interval 对应的 mu 序号 j
                if interval not in mu_map:
                    raise ValueError(f"Interval {interval} not in model intervals.")
                # 用 mu.index 的顺序定位 j，也可直接用排序假定顺序一致
                j = list(self.mu.index).index(interval)
                if j == 0:
                    # i=1 残差
                    prev_day_val = e_all[idx-1, j] if idx-1 >= 0 else 0.0
                    prev5_val = e_all[idx-5, j] if (self.use_day5 and idx-5 >= 0) else 0.0
                    e_pred[j] = self.coefficients.at[interval, 'const']
                    e_pred[j] += self.coefficients.at[interval, 'coef_prev_day'] * prev_day_val
                    if self.use_day5:
                        e_pred[j] += self.coefficients.at[interval, 'coef_prev_5day'] * prev5_val
                else:
                    prev_int_val = e_pred[j-1]
                    prev_day_val = e_all[idx-1, j] if idx-1 >= 0 else 0.0
                    prev5_val = e_all[idx-5, j] if (self.use_day5 and idx-5 >= 0) else 0.0
                    e_pred[j] = self.coefficients.at[interval, 'const']
                    e_pred[j] += self.coefficients.at[interval, 'coef_prev_interval'] * prev_int_val
                    e_pred[j] += self.coefficients.at[interval, 'coef_prev_day'] * prev_day_val
                    if self.use_day5:
                        e_pred[j] += self.coefficients.at[interval, 'coef_prev_5day'] * prev5_val
            # 保存残差用于未来滞后引用
            e_all[idx, :] = e_pred
            # 计算预测成交量 A_pred = S_base * mu + e_pred
            mu_values = np.array([mu_map[intv] for intv in intervals])
            A_pred = S_base * mu_values + e_pred[:len(intervals)]
            if self.clip:
                A_pred = np.where(A_pred < 0, 0.0, A_pred)
            # 更新该日预测总量以供下一日使用
            predicted_total = A_pred.sum()
            S_all[idx] = predicted_total
            # 收集预测结果
            for val in A_pred:
                predictions.append(val)
        # 构建结果 DataFrame
        output_df = test_df[['date', 'interval']].copy()
        output_df['predicted'] = predictions
        return output_df

    def get_state(self):
        """
        返回模型拟合状态，包括 mu、coefficients、resid_std、S_train、converged、iterations 等。
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")
        return {
            'mu': self.mu,
            'coefficients': self.coefficients,
            'resid_std': self.resid_std,
            'S_train': self.S_train,
            'converged': self.converged_,
            'iterations': self.iterations_
        }