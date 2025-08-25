# scripts/synth_intraday_data.py
# 随机生成“UAM + DeepLOBv”可用的长表数据：
# - training_df: 2020-01-01 ~ 2022-12-31
# - testing_df:  2023-01-01 ~ 2023-12-31
# 并保存到 data/interim/*.parquet

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import os
import numpy as np
import pandas as pd

@dataclass
class SynthConfig:
    # 市场与时间
    bars_per_day: int = 67
    start_train: str = "2020-01-01"
    end_train:   str = "2022-12-31"
    start_test:  str = "2023-01-01"
    end_test:    str = "2023-12-31"
    # 规模
    n_symbols: int = 60
    n_extra_features: int = 32
    # 稀疏/停牌控制
    p_halt_day: float = 0.02
    p_gap_block: float = 0.08
    gap_min: int = 3
    gap_max: int = 8
    p_drop_bar: float = 0.01
    # 随机种子
    seed: int = 2025

def business_days(start: str, end: str) -> pd.DatetimeIndex:
    # 默认工作日（周一~周五）
    return pd.bdate_range(start=start, end=end)  # 等价于 freq="B"

def make_intraday_profile(T: int, rng: np.random.Generator) -> np.ndarray:
    p = np.linspace(0, 1, T)
    left  = np.exp(-0.5 * ((p - 0.06) / 0.07) ** 2)
    right = np.exp(-0.5 * ((p - 0.94) / 0.07) ** 2)
    mid   = np.exp(-0.5 * ((p - 0.50) / 0.12) ** 2)
    u = left + right - 0.35 * mid
    u = (u - u.mean()) / (u.std() + 1e-8)
    u = u * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05, size=T)
    u = (u - u.mean()) / (u.std() + 1e-8)
    return u.astype(np.float32)

def generate_symbol_params(n_symbols: int, rng: np.random.Generator):
    alpha = rng.normal(6.0, 0.6, size=n_symbols)
    beta  = rng.uniform(0.4, 1.0, size=n_symbols)
    sig_day = rng.uniform(0.15, 0.35, size=n_symbols)
    sig_bar = rng.uniform(0.08, 0.20, size=n_symbols)
    p0    = rng.uniform(10.0, 200.0, size=n_symbols)
    ret_sigma = rng.uniform(0.0008, 0.0025, size=n_symbols)
    return alpha, beta, sig_day, sig_bar, p0, ret_sigma

def generate_market_day_factor(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    phi = 0.95
    eps = rng.normal(0, 0.15, size=len(dates))
    m = np.zeros(len(dates), dtype=np.float32)
    for i in range(1, len(dates)):
        m[i] = phi * m[i-1] + eps[i]
    return pd.Series(m, index=dates)

def dow_effects(dow: int) -> float:
    table = {0: +0.06, 1: +0.03, 2: 0.00, 3: -0.02, 4: -0.04}
    return float(table.get(dow, 0.0))

def inject_intraday_gaps(df_day: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    T = cfg.bars_per_day
    if rng.uniform() < cfg.p_gap_block:
        L = int(rng.integers(cfg.gap_min, cfg.gap_max + 1))
        start = int(rng.integers(1, T - L + 2))
        mask = ~df_day["time"].between(start, start + L - 1)
        df_day = df_day.loc[mask]
    if cfg.p_drop_bar > 0:
        keep_mask = rng.uniform(0, 1, size=len(df_day)) > cfg.p_drop_bar
        df_day = df_day.loc[keep_mask]
    return df_day

def generate_one_split(dates: pd.DatetimeIndex, cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    T = cfg.bars_per_day
    u_profile = make_intraday_profile(T, rng)
    alpha, beta, sig_day, sig_bar, p0, ret_sigma = generate_symbol_params(cfg.n_symbols, rng)
    mkt_day = generate_market_day_factor(dates, rng)

    syms = [f"S{idx:04d}" for idx in range(cfg.n_symbols)]
    rows: List[pd.DataFrame] = []

    for i, sym in enumerate(syms):
        n_steps = len(dates) * T
        rets = rng.normal(0.0, ret_sigma[i], size=n_steps).astype(np.float32)
        price = np.empty(n_steps, dtype=np.float32)
        price[0] = p0[i]
        for k in range(1, n_steps):
            price[k] = price[k-1] * float(np.exp(rets[k]))

        day_idio = rng.normal(0, sig_day[i], size=len(dates)).astype(np.float32)

        ptr = 0
        for di, d in enumerate(dates):
            if rng.uniform() < cfg.p_halt_day:
                ptr += T
                continue

            base_mu = alpha[i] + mkt_day.loc[d] + dow_effects(int(d.weekday())) + day_idio[di]
            t_idx = np.arange(1, T + 1, dtype=np.int16)
            mu_t = base_mu + beta[i] * u_profile
            # 旧：i.i.d.
            # eps_t = rng.normal(0.0, sig_bar[i], size=T).astype(np.float32)
            # 新：AR(1)，rho 可 0.5~0.8
            rho = 0.6
            eta = rng.normal(0.0, sig_bar[i], size=T).astype(np.float32)
            eps_t = np.empty(T, dtype=np.float32)
            eps_t[0] = eta[0]
            for k in range(1, T):
                eps_t[k] = rho * eps_t[k - 1] + eta[k]
            logv = mu_t + eps_t

            px  = price[ptr:ptr+T]
            r_1 = np.empty_like(px); r_1[0] = 0.0; r_1[1:] = np.diff(np.log(px))
            vwap = px * (1.0 + rng.normal(0, 0.0007, size=T).astype(np.float32))
            vol   = np.maximum(0.0, np.expm1(logv))
            amount = (vol * px).astype(np.float32)

            # 位置编码（bar_idx 1..T）
            phase = 2 * np.pi * (t_idx - 1) / T
            bar_sin = np.sin(phase).astype(np.float32)
            bar_cos = np.cos(phase).astype(np.float32)
            is_open = (t_idx <= 3).astype(np.float32)  # 头3格
            is_close = (t_idx >= T - 2).astype(np.float32)  # 尾3格

            df_day = pd.DataFrame({
                "date":     [d] * T,
                "time":     t_idx,
                "symbol":   [sym] * T,
                "y":        logv.astype(np.float32),
                "price":    px.astype(np.float32),
                "vwap":     vwap.astype(np.float32),
                "ret_1":    r_1.astype(np.float32),
                "amount":   amount,
                'bar_sin':  bar_sin,
                'bar_cos':  bar_cos,
                'is_open':  is_open,
                'is_close': is_close
            })

            df_day = inject_intraday_gaps(df_day, cfg, rng)

            # 再在“保留下来的行”上算 lag —— 表示上一条“实际可见”的记录

            rows.append(df_day)
            ptr += T

    df = pd.concat(rows, ignore_index=True)

    df['lag_y1'] = df.groupby('symbol')['y'].shift(1)
    df['lag_y2'] = df.groupby('symbol')['y'].shift(2)

    if cfg.n_extra_features > 0:
        rng2 = np.random.default_rng(cfg.seed + 123)
        for k in range(cfg.n_extra_features):
            w1, w2, w3 = rng2.normal(0, 1), rng2.normal(0, 1), rng2.normal(0, 1)
            noise = rng2.normal(0, 0.1, size=len(df)).astype(np.float32)
            col = (w1 * (df["price"].values / (df["price"].values.mean() + 1e-6)) +
                   w2 * df["ret_1"].values * 200.0 +
                   w3 * (np.log1p(df["amount"].values + 1.0)) / 10.0 +
                   noise)
            df[f"feat_{k:03d}"] = col.astype(np.float32)

    df["date"] = pd.to_datetime(df["date"])
    df["time"] = df["time"].astype("int16")
    df["symbol"] = df["symbol"].astype("string")
    df = df.sort_values(["symbol", "date", "time"]).reset_index(drop=True)

    # 在“同一 symbol、同一日”的可见行上重算 ret_1（log价差），缺第一根就置 0
    df['ret_1'] = (df.groupby(['symbol', 'date'])['price']
                   .transform(lambda s: np.log(s).diff().fillna(0.0))
                   .astype(np.float32))

    # 生成完 symbol 的所有日后、按 ['symbol','date','time'] 排好序：
    # 前一日收盘的 y（上一日最后一个bar的 y），前一日收盘价 px_close
    df['is_day_last'] = (df.groupby(['symbol', 'date'])['time'].transform('max') == df['time']).astype('int8')
    df['y_close_prev'] = df.groupby('symbol')['y'].shift(1).where(df.groupby('symbol')['is_day_last'].shift(1).eq(1))
    df['px_close_prev'] = df.groupby('symbol')['price'].shift(1).where(
        df.groupby('symbol')['is_day_last'].shift(1).eq(1))

    # 当天开盘价（time==1）与昨收的 log 差（overnight return）
    df['overnight_ret'] = np.log(df['price']) - np.log(df['px_close_prev'])
    # 对 NaN 做安全处理或保留 NaN（训练侧会 nan_to_num）

    return df

def generate_training_testing(cfg: SynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    dates_train = business_days(cfg.start_train, cfg.end_train)
    dates_test  = business_days(cfg.start_test,  cfg.end_test)
    training_df = generate_one_split(dates_train, cfg, rng)
    testing_df  = generate_one_split(dates_test,  cfg, rng)
    return training_df, testing_df

def main():
    cfg = SynthConfig(
        bars_per_day=67,
        n_symbols=60,
        n_extra_features=32,
        p_halt_day=0.02,
        p_gap_block=0.08, gap_min=3, gap_max=8,
        p_drop_bar=0.01,
        seed=2025,
    )
    os.makedirs("data/interim", exist_ok=True)
    training_df, testing_df = generate_training_testing(cfg)
    training_df.to_parquet("data/interim/training.parquet", index=False)  # engine='pyarrow' 可显式指定
    testing_df.to_parquet("data/interim/testing.parquet", index=False)

    print("Saved:")
    print("  data/interim/training.parquet  rows =", len(training_df))
    print("  data/interim/testing.parquet   rows =", len(testing_df))
    print("Columns:", list(training_df.columns))
    print("Example head:")
    print(training_df.head())

if __name__ == "__main__":
    # 仅脚本直跑时切到项目根（scripts/ 的上一级）
    import pathlib
    os.chdir(pathlib.Path(__file__).resolve().parents[1])
    main()
