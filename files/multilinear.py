# Synthetic intraday data generator + quick sanity check for your class
# (No external internet; everything runs locally)
import numpy as np
import pandas as pd
import datetime as dt
import os
from typing import Any, Dict, Optional

# --- Your class pasted verbatim (kept as-is; we won't call get_state to avoid its minor 'schema' bug) ---
class LinearIntradayExcessModel():
    """
    A_{d,i} = S_d * mu_i + e_{d,i}
    e_{d,1} = a1 + phi1 * e_{d-1,1} + u
    e_{d,i} = ai + beta_i * e_{d,i-1} + gamma_i * e_{d-1,i} + u (i>1)

    预测：对同一交易日做 Gauss–Seidel 顺序更新，迭代到稳定；全程clip>=0。
    """

    def __init__(self,
                 enforce_daily_total: bool = False,  # 是否强制把日内预测缩放到目标总量
                 tol: float = 1e-6,                  # 收敛阈值
                 max_iter: int = 50):                # 最大迭代轮数
        self.enforce_daily_total = enforce_daily_total
        self.tol = tol
        self.max_iter = max_iter

        # 学到的量
        self.num_intervals_: Optional[int] = None
        self.mu_: Optional[pd.Series] = None             # 日内形状 mu_i (index=interval)
        self.daily_scale_model_: Dict[str, float] = {}   # S_d ≈ alpha + beta*S_{d-1}
        self.excess_coefs_: Dict[int, np.ndarray] = {}   # interval -> coef 向量
        self.excess_sigma_: Dict[int, float] = {}        # interval -> 残差std(诊断用)

        # 预测初始化缓存
        self.last_train_day_amt_: Optional[pd.Series] = None   # 训练集最后一日的 amount
        self.last_train_day_exc_: Optional[pd.Series] = None   # 训练集最后一日的 excess

    # ---------- 工具 ----------
    @staticmethod
    def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        if isinstance(x.index, pd.MultiIndex):
            x = x.reset_index()
        if 'datetime' in x.columns and 'date' not in x.columns:
            x['date'] = x['datetime'].dt.date
        if 'interval' not in x.columns:
            key = ['date', 'datetime'] if 'datetime' in x.columns else ['date']
            x = x.sort_values(key)
            x['interval'] = x.groupby('date').cumcount() + 1
        if 'amount' not in x.columns:
            cand = [c for c in ['volume', 'vol', 'amt'] if c in x.columns]
            if cand:
                x['amount'] = x[cand[0]]
            else:
                raise ValueError("DataFrame必须包含 'amount'（或 volume/vol/amt）列")
        return x[['date','interval','amount']]

    @staticmethod
    def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    # ---------- 训练 ----------
    def fit(self, train_df: pd.DataFrame, **fit_params) -> "LinearIntradayExcessModel":
        df = self._ensure_df(train_df)
        self.num_intervals_ = int(df['interval'].max())

        # (1) pivot (rows=date, cols=interval)
        M = df.pivot(index='date', columns='interval', values='amount').sort_index()
        dates = M.index
        cols = M.columns

        # (2) 估计 mu_i：用“日内比例”的中位数（稳健），保证 sum mu_i = 1
        S = M.sum(axis=1)
        P = M.div(S, axis=0)                   # 每天的日内比例
        mu = P.median(axis=0)                  # 每个interval的中位比例
        mu = mu / mu.sum()                     # 归一化
        self.mu_ = mu.astype(float)

        # (3) 训练阶段的 e_{d,i} 用“实际 S_d”分解：E = M - S_d * mu
        #     这样超额不掺入日尺度预测误差
        baseline = pd.DataFrame(
            np.outer(S.values, self.mu_.values),
            index=dates, columns=cols
        )
        E = M - baseline

        # (4) 简单的 S_d 模型：S_d ≈ alpha + beta*S_{d-1}（仅用于预测期给初值）
        S_lag = S.shift(1)
        scale_df = pd.DataFrame({'S': S, 'S_1': S_lag}).dropna()
        if len(scale_df) >= 3:
            X = np.column_stack([np.ones(len(scale_df)), scale_df['S_1'].values])
            y = scale_df['S'].values
            alpha, beta = self._ols(X, y)
        else:
            alpha, beta = 0.0, 1.0
        self.daily_scale_model_ = {'alpha': float(alpha), 'beta': float(beta)}

        # (5) 拟合每个 interval 的“超额线性方程”
        for i in range(1, self.num_intervals_ + 1):
            if i == 1:
                y = E[i].iloc[1:].values
                e_lag = E[i].iloc[:-1].values
                X = np.column_stack([np.ones_like(y), e_lag])
            else:
                y = E[i].iloc[1:].values
                e_prev_interval = E[i-1].iloc[1:].values
                e_prev_day_same = E[i].iloc[:-1].values
                X = np.column_stack([np.ones_like(y), e_prev_interval, e_prev_day_same])

            beta_i = self._ols(X, y)
            self.excess_coefs_[i] = beta_i

            y_hat = X @ beta_i
            resid = y - y_hat
            dof = max(1, len(y) - X.shape[1])
            self.excess_sigma_[i] = float(np.sqrt((resid**2).sum() / dof))

        # (6) 预测初始化需要的“上一交易日”参考
        self.last_train_day_amt_ = M.iloc[-1].copy()
        S_last = float(S.iloc[-1])
        self.last_train_day_exc_ = self.last_train_day_amt_ - S_last * self.mu_
        return self

    # ---------- 预测（循环到稳定） ----------
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.num_intervals_ is None or self.mu_ is None:
            raise RuntimeError("请先 fit()")

        df = self._ensure_df(test_df)
        dates = pd.Index(df['date'].unique())
        M_test = df.pivot(index='date', columns='interval', values='amount').reindex(dates).sort_index()

        # 结果容器：预测的 amount
        P = pd.DataFrame(index=dates, columns=range(1, self.num_intervals_+1), dtype=float)

        # 前一日参考（优先用真实amount；否则用上一日的预测），以及其超额
        prev_amt = self.last_train_day_amt_.copy()
        prev_exc = self.last_train_day_exc_.copy()

        alpha = self.daily_scale_model_['alpha']
        beta  = self.daily_scale_model_['beta']

        for d in dates:
            # (1) 给日尺度 S_d 初值：alpha + beta * S_{d-1}
            S_prev = float(prev_amt.sum())
            S_d = alpha + beta * S_prev

            # (2) 初始化 e^{(0)} 与 amount^{(0)}
            e_curr = pd.Series(0.0, index=self.mu_.index)
            amt_curr = S_d * self.mu_ + e_curr
            amt_curr = amt_curr.clip(lower=0.0)
            P.loc[d, :] = amt_curr.values

            # (3) 若需要“日总一致性”，确定目标总量
            target_S = None
            if self.enforce_daily_total:
                if d in M_test.index and not M_test.loc[d].isna().all():
                    target_S = float(M_test.loc[d].sum())  # 测试期有真实则用真实
                else:
                    target_S = float(S_d)                  # 否则用 S_d 初值

            # (4) Gauss–Seidel 迭代到稳定
            for _ in range(self.max_iter):
                prev_vec = P.loc[d, :].values.copy()
                # 顺序更新 e_{d,i} 并得到 A_{d,i}
                for i in range(1, self.num_intervals_ + 1):
                    coef = self.excess_coefs_[i]
                    if i == 1:
                        # e_hat = a1 + phi1 * e_{d-1,1}
                        e_hat = coef[0] + coef[1] * float(prev_exc.loc[1])
                    else:
                        # e_hat = ai + beta_i * e_{d,i-1} + gamma_i * e_{d-1,i}
                        e_im1 = e_curr.loc[i-1]
                        e_hat = coef[0] + coef[1] * float(e_im1) + coef[2] * float(prev_exc.loc[i])

                    e_curr.loc[i] = e_hat
                    a_hat = S_d * float(self.mu_.loc[i]) + e_hat
                    if a_hat < 0.0:
                        a_hat = 0.0  # 非负约束
                        # 同步修正 e，以维持 A = S*mu + e
                        e_curr.loc[i] = a_hat - S_d * float(self.mu_.loc[i])
                    P.loc[d, i] = a_hat

                # 可选：把日内值缩放到目标总量 target_S（若设置 enforce_daily_total）
                if self.enforce_daily_total and target_S is not None:
                    tot = float(P.loc[d, :].sum())
                    if tot > 0:
                        scale = target_S / tot
                        P.loc[d, :] *= scale
                        # e 也相应更新
                        for i in e_curr.index:
                            e_curr.loc[i] = float(P.loc[d, i]) - S_d * float(self.mu_.loc[i])

                # 收敛判据
                if np.nanmax(np.abs(P.loc[d, :].values - prev_vec)) < self.tol:
                    break

            # (5) 为下一天准备“前一日参考”
            prev_amt = P.loc[d, :].astype(float)
            prev_exc = e_curr.astype(float)

        # 按输入行顺序返回预测
        pred_flat = P.stack()
        pred_flat.index.names = ['date','interval']
        out = pd.merge(
            df[['date','interval']],
            pred_flat.rename('pred'),
            on=['date','interval'],
            how='left'
        )['pred'].values
        return out

# --- Helper: generate 5-min HKEX intraday timestamps (67 bars) ---
def hk_5min_timestamps(a_date: dt.date) -> list[dt.datetime]:
    times = []
    # Morning: 09:30..12:00 inclusive (31 bars)
    t = dt.datetime.combine(a_date, dt.time(9, 30))
    end_m = dt.datetime.combine(a_date, dt.time(12, 0))
    while t <= end_m:
        times.append(t)
        t += dt.timedelta(minutes=5)
    # Afternoon: 13:05..16:00 inclusive (36 bars)
    t = dt.datetime.combine(a_date, dt.time(13, 5))
    end_a = dt.datetime.combine(a_date, dt.time(16, 0))
    while t <= end_a:
        times.append(t)
        t += dt.timedelta(minutes=5)
    return times  # total 67

# --- Synthetic data generator consistent with the model ---
def make_synthetic_intraday(n_train_days=60, n_test_days=10, seed=42):
    rng = np.random.default_rng(seed)
    n_days = n_train_days + n_test_days
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B").date  # business days

    # Intraday seasonal shape mu (67 bars): bell-shaped AM and PM, small dip near lunch
    n_int = 67
    x = np.linspace(0, 1, n_int)
    # piecewise smooth shape
    mu_raw = (
        0.6 * np.exp(-((x-0.3)/0.18)**2)  # AM hump
        + 0.9 * np.exp(-((x-0.75)/0.15)**2)  # PM hump
        + 0.2
    )
    mu = mu_raw / mu_raw.sum()  # normalize to 1

    # Daily total process S_d (positive AR(1))
    S = np.zeros(n_days, dtype=float)
    S0 = 3.0e8
    alpha_s, rho_s, sigma_s = 5.0e7, 0.85, 3.0e7
    S[0] = S0
    for d in range(1, n_days):
        S[d] = max(1.0, alpha_s + rho_s*S[d-1] + rng.normal(0, sigma_s))

    # Excess e_{d,i} with small magnitude relative to baseline
    beta_i, gamma_i, phi1 = 0.45, 0.25, 0.35
    sigma_e = 0.05  # relative to S_d * mu_i
    E = np.zeros((n_days, n_int), dtype=float)
    for d in range(n_days):
        for i in range(n_int):
            base_sd = S[d] * mu[i]
            noise = rng.normal(0.0, sigma_e * np.sqrt(max(base_sd, 1.0)))  # heteroskedastic-ish
            if i == 0:
                prev_day_e1 = E[d-1, 0] if d > 0 else 0.0
                E[d, 0] = 0.0 + phi1 * prev_day_e1 + noise
            else:
                prev_e = E[d, i-1]
                prev_day_same = E[d-1, i] if d > 0 else 0.0
                E[d, i] = 0.0 + beta_i * prev_e + gamma_i * prev_day_same + noise

    A = S[:, None] * mu[None, :] + E
    A = np.maximum(A, 0.0)  # nonnegative

    # Build DataFrame
    rows = []
    for di, day in enumerate(dates):
        stamps = hk_5min_timestamps(day)
        for i, ts in enumerate(stamps, start=1):
            rows.append((ts, day, i, float(A[di, i-1])))
    df_all = pd.DataFrame(rows, columns=["datetime", "date", "interval", "amount"])

    train_df = df_all[df_all["date"] < dates[n_train_days]]
    test_df  = df_all[df_all["date"] >= dates[n_train_days]]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), mu, S

# --- Generate toy data and run a quick sanity check ---
train_df, test_df, mu_vec, S_vec = make_synthetic_intraday(n_train_days=60, n_test_days=10, seed=7)

model = LinearIntradayExcessModel(enforce_daily_total=False, tol=1e-6, max_iter=10)
model.fit(train_df)
pred = model.predict(test_df)

# Attach predictions for evaluation
test_with_pred = test_df.copy()
test_with_pred["pred"] = pred

# Compute simple metrics
eps = 1e-12
mape = (np.abs(test_with_pred["pred"] - test_with_pred["amount"]) / (test_with_pred["amount"] + eps)).mean()
rmse = np.sqrt(((test_with_pred["pred"] - test_with_pred["amount"])**2).mean())

# Save CSV for you to download / inspect
out_path = "sim_intraday.csv"
combo = pd.concat([train_df.assign(set="train"), test_with_pred.assign(set="test")], ignore_index=True)
combo.to_csv(out_path, index=False)

# Return a small summary
summary = {
    "n_train_rows": len(train_df),
    "n_test_rows": len(test_df),
    "intervals": int(train_df["interval"].max()),
    "days_train": train_df["date"].nunique(),
    "days_test": test_df["date"].nunique(),
    "MAPE_on_test": float(mape),
    "RMSE_on_test": float(rmse),
    "csv_path": out_path
}
summary
