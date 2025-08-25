# /src/data/uam_threeway_prep.py
# ---------------------------------------------------------------------
# 从 (training_df, testing_df) 构造三路 DataLoader：
#   - inner_train  = training_df 且 date < inner_split_date
#   - inner_test   = training_df 且 date >= inner_split_date
#   - outer_test   = testing_df (例如 2023 全年)
#
# 假设：你已完成特征工程/标准化，DataFrame 至少有列：
#   date, time, symbol, y 以及 120+ 数值特征（类别特征已编码）
#
# 两种窗口：
#   - dense：要求日历连续（global_bar_id 逐格相邻）
#   - sparse：不要求日历连续，向前凑满 L 条；建议加 delta_g/gap_flag 特征
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

# ========================
# Config
# ========================

@dataclass
class ThreeWayConfig:
    # 列名
    date_col: str = 'date'
    time_col: str = 'time'
    symbol_col: str = 'symbol'
    y_col: str = 'y'                       # 目标在时刻 t；内部会生成 y_next = shift(-1)
    feature_cols: Optional[Sequence[str]] = None  # 若 None，自动检测数值列

    # 内部分割点
    inner_split_date: str = '2022-06-01'  # inner_train: date < 该日；inner_test: date >= 该日

    # 窗口
    L_days: int = 10
    bars_per_day: Optional[int] = None    # None=自动按每天 bar 数众数推断 T
    min_valid_ratio: float = 0.7          # 窗口内“可算损失”的位置占比阈值

    # 窗口模式
    use_sparse: bool = True               # True=sparse；False=dense
    add_delta_g_feature: bool = True      # sparse 模式下附加 delta_g/gap_flag
    delta_cap: int = 400                  # 限幅

    # 训练采样（UAM）
    K_per_symbol: int = 40                # 每股/epoch 抽 K 个窗口

    # DataLoader
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True

    # 随机
    seed: int = 2025


# ========================
# Utilities
# ========================

def _ensure_datetime(s: pd.Series) -> pd.Series:
    return s if np.issubdtype(s.dtype, np.datetime64) else pd.to_datetime(s)

def _ensure_bar_idx(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, Dict]:
    """把 time 列（数字或 'HH:MM' 字符串）映射到全局 1..T 的 bar_idx。"""
    out = df.copy()
    uniq = out[time_col].dropna().unique()
    try:
        order = np.sort(uniq)
    except Exception:
        order = pd.Index(uniq).astype(str).sort_values().values
    mapping = {v: i+1 for i, v in enumerate(order)}
    out['bar_idx'] = out[time_col].map(mapping).astype('int16')
    return out, mapping

def _infer_T(df: pd.DataFrame, date_col: str, bar_col: str) -> int:
    counts = df.groupby(date_col)[bar_col].nunique()
    return int(counts.mode().iloc[0])

def _build_global_bar_id(df: pd.DataFrame, date_col: str, bar_col: str) -> np.ndarray:
    cal = df[[date_col, bar_col]].drop_duplicates().sort_values([date_col, bar_col]).reset_index(drop=True)
    cal['global_bar_id'] = np.arange(len(cal), dtype=np.int64)
    m = df.merge(cal, on=[date_col, bar_col], how='left')
    return m['global_bar_id'].values.astype(np.int64)

# ========================
# Arrays builder (three-way)
# ========================

def prepare_threeway_arrays(
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    cfg: ThreeWayConfig
) -> Dict[str, np.ndarray]:
    """
    生成数组与三路掩码：
      - inner_train：training_df 且 date < split
      - inner_test： training_df 且 date >= split
      - outer_test： testing_df 全部
    生成：
      X, Y(=y_next), M(mask_label), S(symbol_id), D(date), B(bar_idx), g(global_bar_id)
      以及 is_inner_train/_test/_outer_test 与它们的 next 版本（严格避免泄漏）
    """
    train = training_df.copy()
    test  = testing_df.copy()

    # 日期类型
    train[cfg.date_col] = _ensure_datetime(train[cfg.date_col])
    test[cfg.date_col]  = _ensure_datetime(test[cfg.date_col])

    # 标记来源并合并，统一映射
    train['_origin'] = 'train_outer'
    test['_origin']  = 'test_outer'
    all_df = pd.concat([train, test], ignore_index=True)

    # symbol → id
    syms = pd.Index(all_df[cfg.symbol_col].unique()).sort_values()
    sym2id = {s: i for i, s in enumerate(syms)}
    all_df['symbol_id'] = all_df[cfg.symbol_col].map(sym2id).astype('int32')

    # time → bar_idx
    all_df, time_mapping = _ensure_bar_idx(all_df, cfg.time_col)
    bar_col = 'bar_idx'

    # 排序
    all_df = all_df.sort_values(['symbol_id', cfg.date_col, bar_col]).reset_index(drop=True)

    # 存在标记与下一步可用
    all_df['mask_bar']  = 1
    all_df['mask_next'] = all_df.groupby('symbol_id')['mask_bar'].shift(-1).fillna(0).astype(int)
    all_df['mask_label'] = (all_df['mask_bar'] * all_df['mask_next']).astype('int16')

    # y_next
    if cfg.y_col not in all_df.columns:
        raise ValueError(f"找不到 y_col='{cfg.y_col}'")
    all_df['y_next'] = all_df.groupby('symbol_id')[cfg.y_col].shift(-1)

    # global_bar_id（dense 用；sparse 可用于 delta_g）
    all_df['global_bar_id'] = _build_global_bar_id(all_df, cfg.date_col, bar_col)

    # 三路掩码
    split_dt = np.datetime64(cfg.inner_split_date)
    in_train_outer = (all_df['_origin'] == 'train_outer').values
    inner_train_mask = in_train_outer & (all_df[cfg.date_col].values < split_dt)
    inner_test_mask  = in_train_outer & (all_df[cfg.date_col].values >= split_dt)
    outer_test_mask  = (all_df['_origin'] == 'test_outer').values

    # 组内 next 掩码（严格判断“下一步仍在本分割中”）
    all_df['is_inner_train'] = inner_train_mask.astype('int8')
    all_df['is_inner_test']  = inner_test_mask.astype('int8')
    all_df['is_outer_test']  = outer_test_mask.astype('int8')
    all_df['is_inner_train_next'] = all_df.groupby('symbol_id')['is_inner_train'].shift(-1).fillna(0).astype('int8')
    all_df['is_inner_test_next']  = all_df.groupby('symbol_id')['is_inner_test'].shift(-1).fillna(0).astype('int8')

    # 特征列
    if cfg.feature_cols is None:
        reserved = {
            cfg.date_col, cfg.time_col, cfg.symbol_col, 'symbol_id',
            cfg.y_col, 'y_next', 'mask_bar', 'mask_next', 'mask_label',
            'global_bar_id', 'bar_idx', '_origin',
            'is_inner_train', 'is_inner_test', 'is_outer_test',
            'is_inner_train_next', 'is_inner_test_next'
        }
        feat_cols = [c for c in all_df.columns
                     if c not in reserved and np.issubdtype(all_df[c].dtype, np.number)]
    else:
        feat_cols = list(cfg.feature_cols)

    # sparse 模式下，附加 delta_g/gap_flag（建议开启）
    if cfg.use_sparse and cfg.add_delta_g_feature:
        all_df['g_prev'] = all_df.groupby('symbol_id')['global_bar_id'].shift(1)
        dg = (all_df['global_bar_id'] - all_df['g_prev']).fillna(1).clip(lower=1, upper=cfg.delta_cap).astype('float32')
        gap_flag = (dg > 1).astype('float32')
        all_df['delta_g'] = dg
        all_df['gap_flag'] = gap_flag
        for extra in ['delta_g', 'gap_flag']:
            if extra not in feat_cols:
                feat_cols.append(extra)

    # 数组
    X = all_df[feat_cols].values.astype('float32')
    Y = all_df['y_next'].values.astype('float32'); Y[np.isnan(Y)] = 0.0
    M = all_df['mask_label'].values.astype('float32')
    S = all_df['symbol_id'].values.astype('int32')
    D = all_df[cfg.date_col].values
    B = all_df['bar_idx'].values.astype('int16')
    G = all_df['global_bar_id'].values.astype('int64')

    # 从 inner_train 行推断 T 与 L
    if cfg.bars_per_day is None:
        T = _infer_T(all_df[inner_train_mask], cfg.date_col, 'bar_idx')
    else:
        T = int(cfg.bars_per_day)
    L = int(cfg.L_days) * int(T)

    return {
        'X': X, 'Y': Y, 'M': M, 'S': S, 'D': D, 'B': B, 'g': G,
        'is_inner_train': inner_train_mask,
        'is_inner_test':  inner_test_mask,
        'is_outer_test':  outer_test_mask,
        'is_inner_train_next': all_df['is_inner_train_next'].values.astype('int8'),
        'is_inner_test_next':  all_df['is_inner_test_next'].values.astype('int8'),
        'feature_cols': feat_cols,
        'time_mapping': time_mapping,
        'symbol_mapping': {i: s for s, i in sym2id.items()},
        'T': T, 'L': L
    }

# ========================
# 窗口起点构造（dense / sparse）
# ========================

def _dense_train_starts(G, S, M, is_train, is_train_next, L, min_valid_ratio) -> Dict[int, np.ndarray]:
    """dense 训练起点：整窗在 inner_train；global_bar_id 连续；label-in-train 覆盖率达标。"""
    out: Dict[int, List[int]] = {}
    order = np.argsort(S, kind='mergesort')
    S_, G_, M_, TR_, TRN_ = S[order], G[order], M[order], is_train[order], is_train_next[order]
    uniq, starts = np.unique(S_, return_index=True)
    bounds = list(starts) + [len(S_)]
    for k in range(len(uniq)):
        sym = int(uniq[k]); a, b = bounds[k], bounds[k+1]
        if (b - a) < L: out[sym] = np.array([], dtype=np.int64); continue
        g = G_[a:b]; m = M_[a:b]; tr = TR_[a:b]; trn = TRN_[a:b]
        adj = (g[1:] - g[:-1] == 1).astype(np.int8)
        c_adj = np.concatenate(([0], np.cumsum(adj)))
        mtrain = (m.astype(np.int8) & tr.astype(np.int8) & trn.astype(np.int8))
        c_cov  = np.concatenate(([0], np.cumsum(mtrain)))
        c_tr   = np.concatenate(([0], np.cumsum(tr.astype(np.int32))))
        val_starts = []
        for j in range(0, (b - a) - L + 1):
            if (c_adj[j + (L-1)] - c_adj[j]) != (L - 1): continue
            if (c_tr[j + L] - c_tr[j]) != L: continue
            cov = c_cov[j + L] - c_cov[j]
            if cov < min_valid_ratio * L: continue
            val_starts.append(int(order[a + j]))
        out[sym] = np.array(val_starts, dtype=np.int64)
    return out

def _dense_eval_starts(G, S, M, split_mask, L) -> np.ndarray:
    """dense 评估起点：要求 adjacency；窗口右端在 split 且有标签。"""
    starts_all: List[int] = []
    order = np.argsort(S, kind='mergesort')
    S_, G_, M_, SP_ = S[order], G[order], M[order], split_mask[order]
    uniq, starts = np.unique(S_, return_index=True)
    bounds = list(starts) + [len(S_)]
    for k in range(len(uniq)):
        a, b = bounds[k], bounds[k+1]
        if (b - a) < L: continue
        g = G_[a:b]; m = M_[a:b]; sp = SP_[a:b]
        adj = (g[1:] - g[:-1] == 1).astype(np.int8)
        c_adj = np.concatenate(([0], np.cumsum(adj)))
        for j in range(0, (b - a) - L + 1):
            end = j + L - 1
            if (c_adj[j + (L-1)] - c_adj[j]) != (L - 1): continue
            if not (sp[end] and (m[end] == 1)): continue
            starts_all.append(int(order[a + j]))
    return np.array(starts_all, dtype=np.int64)

def _sparse_train_starts(S, M, is_train, is_train_next, L, min_valid_ratio) -> Dict[int, np.ndarray]:
    """sparse 训练起点：end 在 inner_train 且有标签且下一步仍在 inner_train；覆盖率达标。"""
    out: Dict[int, List[int]] = {}
    order = np.argsort(S, kind='mergesort')
    S_, M_, TR_, TRN_ = S[order], M[order], is_train[order], is_train_next[order]
    uniq, starts = np.unique(S_, return_index=True)
    bounds = list(starts) + [len(S_)]
    for k in range(len(uniq)):
        sym = int(uniq[k]); a, b = bounds[k], bounds[k+1]
        if (b - a) < L: out[sym] = np.array([], dtype=np.int64); continue
        m = M_[a:b].astype(np.int8)
        tr = TR_[a:b].astype(np.int8)
        trn= TRN_[a:b].astype(np.int8)
        mtrain = (m & tr & trn).astype(np.int8)
        c_cov = np.concatenate(([0], np.cumsum(mtrain)))
        # 候选 end：在 inner_train 且有标签且下一步仍在 inner_train
        ends = np.nonzero((tr == 1) & (m == 1) & (trn == 1))[0]
        val_starts = []
        for e_loc in ends:
            s_loc = e_loc - (L - 1)
            if s_loc < 0: continue
            cov = c_cov[e_loc + 1] - c_cov[s_loc]
            if cov < min_valid_ratio * L: continue
            val_starts.append(int(order[a + s_loc]))
        out[sym] = np.array(val_starts, dtype=np.int64)
    return out

def _sparse_eval_starts(S, M, split_mask, L) -> np.ndarray:
    """sparse 评估起点：end 在 split 且有标签；左端可在 split 前（用历史当上下文）。"""
    starts_all: List[int] = []
    order = np.argsort(S, kind='mergesort')
    S_, M_, SP_ = S[order], M[order], split_mask[order]
    uniq, starts = np.unique(S_, return_index=True)
    bounds = list(starts) + [len(S_)]
    for k in range(len(uniq)):
        a, b = bounds[k], bounds[k+1]
        if (b - a) < L: continue
        m = M_[a:b].astype(np.int8)
        sp= SP_[a:b].astype(np.int8)
        ends = np.nonzero((sp == 1) & (m == 1))[0]
        for e_loc in ends:
            s_loc = e_loc - (L - 1)
            if s_loc < 0: continue
            starts_all.append(int(order[a + s_loc]))
    return np.array(starts_all, dtype=np.int64)

# ========================
# Dataset & Sampler
# ========================

class UAMWindowDataset(Dataset):
    """从全局排序数组中切 [s, s+L) 窗口（dense/sparse 都适用）。"""
    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray, L: int, start_indices: Sequence[int]):
        self.X, self.Y, self.M = X, Y, M
        self.L = int(L)
        self.starts = np.asarray(start_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i]); e = s + self.L
        x = torch.from_numpy(self.X[s:e])   # (L, F)
        y = torch.from_numpy(self.Y[s:e])   # (L,)
        m = torch.from_numpy(self.M[s:e])   # (L,)
        return x, y, m

class BalancedPerSymbolSampler(Sampler[int]):
    """每个 epoch 为每只股票抽 K 个起点（均衡 UAM）。"""
    def __init__(self, sym2starts: Dict[int, np.ndarray], K_per_symbol: int, seed: int = 2025, shuffle: bool = True):
        self.sym2starts = sym2starts
        self.K = int(K_per_symbol)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.symbols = sorted(sym2starts.keys())

    def __iter__(self):
        out = []
        for s in self.symbols:
            pool = self.sym2starts[s]
            if len(pool) == 0:
                continue
            if self.shuffle:
                pool = self.rng.permutation(pool)
            take = pool[:self.K] if len(pool) >= self.K else self.rng.choice(pool, size=self.K, replace=True)
            out.append(take)
        if len(out) == 0:
            return iter([])
        arr = np.concatenate(out)
        if self.shuffle:
            arr = self.rng.permutation(arr)
        return iter(arr.tolist())

    def __len__(self) -> int:
        total = 0
        for pool in self.sym2starts.values():
            total += max(len(pool), self.K if len(pool) > 0 else 0)
        return total

# ========================
# Top-level: 构造三路 DataLoader
# ========================

def make_threeway_loaders(
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    cfg: ThreeWayConfig
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    返回 (train_loader, inner_test_loader, outer_test_loader, meta)。
      - train_loader：对 inner_train 做均衡采样，损失用“train-only 标签掩码”
      - inner_test_loader：窗口右端在 inner_test（评估通常只用最后一格）
      - outer_test_loader：窗口右端在 outer_test（评估只用最后一格）
    """
    pack = prepare_threeway_arrays(training_df, testing_df, cfg)
    X, Y, M, S, D, B, G = pack['X'], pack['Y'], pack['M'], pack['S'], pack['D'], pack['B'], pack['g']
    is_tr, is_te_in, is_te_out = pack['is_inner_train'], pack['is_inner_test'], pack['is_outer_test']
    is_trn, is_ten = pack['is_inner_train_next'], pack['is_inner_test_next']
    T, L = pack['T'], pack['L']

    if cfg.use_sparse:
        # TRAIN（sparse）
        sym_train_starts = _sparse_train_starts(S, M, is_tr, is_trn, L, cfg.min_valid_ratio)
        train_sampler = BalancedPerSymbolSampler(sym_train_starts, K_per_symbol=cfg.K_per_symbol, seed=cfg.seed, shuffle=True)
        train_indices = list(iter(train_sampler))
        # 训练掩码：仅在“训练且下一步仍在训练”的位置计损失
        M_train = (M * is_tr.astype(np.float32) * is_trn.astype(np.float32)).astype('float32')
        train_ds = UAMWindowDataset(X, Y, M_train, L, train_indices)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        # EVAL（end 在 split）
        inner_eval_indices = _sparse_eval_starts(S, M, is_te_in, L)
        outer_eval_indices = _sparse_eval_starts(S, M, is_te_out, L)
    else:
        # TRAIN（dense）
        sym_train_starts = _dense_train_starts(G, S, M, is_tr, is_trn, L, cfg.min_valid_ratio)
        train_sampler = BalancedPerSymbolSampler(sym_train_starts, K_per_symbol=cfg.K_per_symbol, seed=cfg.seed, shuffle=True)
        train_indices = list(iter(train_sampler))
        M_train = (M * is_tr.astype(np.float32) * is_trn.astype(np.float32)).astype('float32')
        train_ds = UAMWindowDataset(X, Y, M_train, L, train_indices)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        # EVAL（dense）
        inner_eval_indices = _dense_eval_starts(G, S, M, is_te_in, L)
        outer_eval_indices = _dense_eval_starts(G, S, M, is_te_out, L)

    inner_test_ds  = UAMWindowDataset(X, Y, M, L, inner_eval_indices)
    outer_test_ds  = UAMWindowDataset(X, Y, M, L, outer_eval_indices)
    inner_test_loader = DataLoader(inner_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    outer_test_loader = DataLoader(outer_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    meta = {**pack,
            'train_start_indices': np.array(train_indices, dtype=np.int64),
            'inner_test_start_indices': inner_eval_indices,
            'outer_test_start_indices': outer_eval_indices}
    return train_loader, inner_test_loader, outer_test_loader, meta
