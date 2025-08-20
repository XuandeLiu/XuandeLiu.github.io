# -*- coding: utf-8 -*-
"""
This cell generates ready-to-use Python modules for preparing UAM (Universal Asset Model)
training data and a DeepLOBv-style model. You can download them and import in your project.

Files created:
- /mnt/data/uam_data_prep.py
- /mnt/data/deeplobv_model.py

Both are self-contained and documented.
"""
from textwrap import dedent

uam_prep_code = dedent(r'''
# uam_data_prep.py
# -------------------------------------------
# Utilities to convert a tidy long DataFrame
#   [date, time, symbol, feature_1..feature_F, (target col)]
# into arrays and window samplers for UAM + DeepLOBv.
#
# Author: prepared by ChatGPT (GPT-5 Pro)
# -------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence
import warnings
import math
import torch
from torch.utils.data import Dataset, Sampler

# -----------------------------
# Configuration dataclass
# -----------------------------

@dataclass
class PrepConfig:
    date_col: str = 'date'      # column with trading date (string or datetime)
    time_col: str = 'time'      # column with intraday interval index (1..T or string like "09:35")
    symbol_col: str = 'symbol'  # column with symbol ticker
    target_col_candidates: Tuple[str, ...] = ('volume', 'dollar_volume', 'vol', 'turnover')
    # train/val/test temporal split boundaries (inclusive ranges)
    train_start: Optional[str] = '2020-01-01'
    train_end: Optional[str]   = '2022-12-31'
    val_start: Optional[str]   = None  # if None, will use entire train_end year split patterns you define externally
    val_end: Optional[str]     = None
    test_start: Optional[str]  = '2023-01-01'
    test_end: Optional[str]    = '2023-12-31'
    # standardization
    standardize_target: bool = True
    standardize_features: bool = True
    robust_feature_scaler: bool = False  # if True, use median/IQR for features
    # windowing
    L_days: int = 10          # number of days in the context window
    bars_per_day: Optional[int] = None  # T; if None, inferred as max count of unique time within a day
    min_valid_ratio: float = 0.7  # within a window, fraction of points with label mask=1
    # sampling
    K_per_symbol: int = 40    # samples per symbol per epoch (training)
    # misc
    assume_zero_trade_for_missing_bars: bool = False  # leave False by default; requires calendar to be safe

# -----------------------------
# Helper functions
# -----------------------------

def _detect_target_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a target column among candidates {candidates}. "
                     f"Please set PrepConfig.target_col_candidates accordingly.")

def _parse_datetime_columns(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    # date -> datetime64[ns]
    if not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col])
    # time could be already integer 1..T; otherwise, try to parse hh:mm into ordinal within-day
    t = out[time_col]
    if np.issubdtype(t.dtype, np.number):
        # assume already 1..T or 0..T-1; normalize to 1..T later
        pass
    else:
        # parse string like "09:35" -> minutes from open; here we just order lexicographically within date
        # We will map unique times to 1..T in sorted order per entire dataset
        pass
    return out

def _infer_bars_per_day(df: pd.DataFrame, date_col: str, time_col: str) -> int:
    # Use the mode (most frequent) of unique time counts per date
    counts = df.groupby(date_col)[time_col].nunique()
    return int(counts.mode().iloc[0])

def _ensure_time_index(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    """
    Ensure intraday time is an integer index 1..T consistent across the dataset.
    If `time_col` is numeric but not 1..T, we will remap by sorting unique times globally.
    If it's string-like, we map unique sorted values to 1..T as well.
    """
    out = df.copy()
    # Build a global ordering of time values:
    unique_times = out[time_col].dropna().unique()
    # Use pandas' natural sort (if string) or numeric sort
    try:
        order = np.sort(unique_times)
    except Exception:
        order = pd.Index(unique_times).astype(str).sort_values().values
    mapping = {v: i+1 for i, v in enumerate(order)}
    out['bar_idx'] = out[time_col].map(mapping).astype('int32')
    return out, mapping

def _build_global_calendar_map(df: pd.DataFrame, date_col: str, bar_col: str) -> pd.Series:
    """
    Create a monotonically increasing global index for each (date, bar_idx) pair,
    shared across all symbols. This allows contiguity tests even when some symbols
    miss bars.
    Returns a Series aligned to df rows: global_bar_id (int64).
    """
    # Unique sorted (date, bar_idx)
    cal = df[[date_col, bar_col]].drop_duplicates().sort_values([date_col, bar_col]).reset_index(drop=True)
    cal['global_bar_id'] = np.arange(len(cal), dtype=np.int64)
    # Merge back
    merged = df.merge(cal, on=[date_col, bar_col], how='left')
    return merged['global_bar_id']

def _standardize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / (std + 1e-8)

def _robust_scale(arr: np.ndarray, median: float, iqr: float) -> np.ndarray:
    return (arr - median) / (iqr + 1e-8)

# -----------------------------
# Public API
# -----------------------------

def prepare_uam_arrays(
    df: pd.DataFrame,
    config: PrepConfig,
    feature_cols: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Convert a tidy long DataFrame into arrays for UAM training and evaluation.
    Returns a dict with arrays and metadata:
        - X: [N_total, F] float32
        - Y: [N_total]    float32 (next-step standardized log target; NaN where label unavailable)
        - M: [N_total]    float32 (label mask, 0/1)
        - S: [N_total]    int32   (symbol_id)
        - D: [N_total]    datetime64[ns] (date)
        - B: [N_total]    int16   (bar_idx 1..T)
        - g: [N_total]    int64   (global_bar_id)
        - train_mask, val_mask, test_mask: boolean arrays aligned to rows
        - scalers: dict with feature/target stats (fit on training rows only)
        - time_mapping: dict mapping original time values -> bar_idx
    """
    cfg = config

    # 0) sanitize columns & parse
    df = df.copy()
    if cfg.date_col not in df.columns or cfg.time_col not in df.columns or cfg.symbol_col not in df.columns:
        raise ValueError(f"DataFrame must have columns {cfg.date_col}, {cfg.time_col}, {cfg.symbol_col}.")

    # 1) map symbol -> id
    # ensure stable order
    syms = pd.Index(df[cfg.symbol_col].unique()).sort_values()
    sym2id = {s: i for i, s in enumerate(syms)}
    df['symbol_id'] = df[cfg.symbol_col].map(sym2id).astype('int32')

    # 2) normalize intraday time to bar_idx 1..T
    df, time_mapping = _ensure_time_index(df, cfg.date_col, cfg.time_col)
    bar_col = 'bar_idx'

    # 3) detect target column
    target_col = _detect_target_col(df, cfg.target_col_candidates)
    if df[target_col].isna().all():
        raise ValueError(f"Target column '{target_col}' is all NaN.")

    # 4) log1p target
    df['y_raw'] = df[target_col].astype('float64')
    df['y_log'] = np.log1p(df['y_raw'].clip(lower=0))

    # 5) sort rows by (symbol, date, bar)
    df = df.sort_values(['symbol_id', cfg.date_col, bar_col]).reset_index(drop=True)

    # 6) label mask (bar-level) and next-step label
    # Here, we treat "row exists" as bar is present. If you want to treat missing bars as zero-trade,
    # build a full calendar before this step.
    df['mask_bar'] = 1  # present rows are evaluable by default
    # Next-step availability: current and next bar must both be present for label
    df['mask_next'] = df.groupby('symbol_id')['mask_bar'].shift(-1).fillna(0).astype(int)
    df['mask_label'] = (df['mask_bar'] * df['mask_next']).astype('int32')
    df['y_log_next'] = df.groupby('symbol_id')['y_log'].shift(-1)

    # 7) build global calendar id to enable contiguity checks
    df['global_bar_id'] = _build_global_calendar_map(df, cfg.date_col, bar_col).values

    # 8) temporal splits
    # ensure datetime
    if not np.issubdtype(df[cfg.date_col].dtype, np.datetime64):
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    date = df[cfg.date_col].values
    train_mask = (date >= np.datetime64(cfg.train_start)) & (date <= np.datetime64(cfg.train_end))
    if cfg.val_start and cfg.val_end:
        val_mask = (date >= np.datetime64(cfg.val_start)) & (date <= np.datetime64(cfg.val_end))
    else:
        # no explicit val: keep False; user can do rolling folds externally
        val_mask = np.zeros(len(df), dtype=bool)
    test_mask = (date >= np.datetime64(cfg.test_start)) & (date <= np.datetime64(cfg.test_end))

    # 9) feature columns
    if feature_cols is None:
        reserved = {cfg.date_col, cfg.time_col, cfg.symbol_col, 'symbol_id', 'y_raw', 'y_log',
                    'y_log_next', 'mask_bar', 'mask_next', 'mask_label', 'global_bar_id', 'bar_idx'}
        # numeric columns apart from reserved
        num_cols = [c for c in df.columns if c not in reserved and
                    (np.issubdtype(df[c].dtype, np.number))]
        feature_cols = num_cols
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns detected. Please pass feature_cols explicitly.")

    F = len(feature_cols)

    # 10) standardization (fit on training rows only)
    scalers = {'target': None, 'features': {}}
    y = df['y_log_next'].values.astype('float64')   # will standardize below
    M = df['mask_label'].values.astype('float32')

    # Target scaler
    if cfg.standardize_target:
        y_train = y[train_mask & (M == 1)]
        y_mu = float(np.nanmean(y_train))
        y_std = float(np.nanstd(y_train))
        scalers['target'] = {'mean': y_mu, 'std': y_std}
        y_std_all = (y - y_mu) / (y_std + 1e-8)
        y = y_std_all.astype('float32')
    else:
        scalers['target'] = None
        y = y.astype('float32')

    # Feature scaler
    X = df[feature_cols].values.astype('float64')
    if cfg.standardize_features:
        # Train rows only
        X_train = X[train_mask]
        if cfg.robust_feature_scaler:
            med = np.nanmedian(X_train, axis=0)
            iqr = np.nanpercentile(X_train, 75, axis=0) - np.nanpercentile(X_train, 25, axis=0)
            X = (X - med) / (iqr + 1e-8)
            scalers['features'] = {'type': 'robust', 'median': med.astype('float32'), 'iqr': iqr.astype('float32')}
        else:
            mu = np.nanmean(X_train, axis=0)
            std = np.nanstd(X_train, axis=0)
            X = (X - mu) / (std + 1e-8)
            scalers['features'] = {'type': 'zscore', 'mean': mu.astype('float32'), 'std': std.astype('float32')}
    X = X.astype('float32')

    # 11) assemble outputs
    S = df['symbol_id'].values.astype('int32')
    D = df[cfg.date_col].values
    B = df['bar_idx'].values.astype('int16')
    G = df['global_bar_id'].values.astype('int64')

    return {
        'X': X, 'Y': y, 'M': M, 'S': S, 'D': D, 'B': B, 'g': G,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
        'feature_cols': feature_cols, 'time_mapping': time_mapping,
        'scalers': scalers, 'symbol_mapping': {int(v): k for k, v in {k:v for k,v in zip(syms, range(len(syms)))}.items()}
    }

# -----------------------------
# Window utilities for UAM
# -----------------------------

def build_symbol_start_indices(
    G: np.ndarray,  # global_bar_id per row
    S: np.ndarray,  # symbol_id per row
    M: np.ndarray,  # label mask per row (0/1)
    subset_mask: np.ndarray,  # boolean mask to constrain to a temporal split (e.g., train rows)
    L: int,
    min_valid_ratio: float = 0.7
) -> Dict[int, np.ndarray]:
    """
    For each symbol, return valid start row-indices i such that:
      - rows i..i+L-1 belong to the same symbol
      - global_bar_id is contiguous: g[j+1] - g[j] == 1 for all window positions
      - within the window, the fraction of M==1 is >= min_valid_ratio
    Only start indices whose **entire window** lies within subset_mask are returned.
    """
    assert len(G) == len(S) == len(M) == len(subset_mask)
    n = len(G)
    out: Dict[int, List[int]] = {}
    # Pre-group by symbol
    order = np.argsort(S, kind='mergesort')
    S_sorted = S[order]
    G_sorted = G[order]
    M_sorted = M[order]
    subset_sorted = subset_mask[order]
    # find boundaries
    uniq, starts = np.unique(S_sorted, return_index=True)
    idx_boundaries = list(starts) + [len(S_sorted)]
    for k in range(len(uniq)):
        sym = int(uniq[k])
        a, b = idx_boundaries[k], idx_boundaries[k+1]
        g = G_sorted[a:b]
        m = M_sorted[a:b]
        sm = subset_sorted[a:b]
        # contiguity: diff == 1 => adjacency
        if len(g) < L:
            out[sym] = np.array([], dtype=np.int64); continue
        diff = g[1:] - g[:-1]
        adj = (diff == 1).astype(np.int8)  # length b-a-1
        # sliding sum over adj to check all L-1 adjacencies inside window
        # We'll build a window-valid mask of length (n_sym - L + 1)
        # adjacency sum over window = L-1 required
        # efficient via cumsum
        c = np.concatenate(([0], np.cumsum(adj)))
        # adjacency sum for window starting at j: sum = c[j+L-1] - c[j]
        # Also need: all rows within window belong to subset_mask
        # We'll cumulative-sum sm (boolean) to count valid rows
        cs = np.concatenate(([0], np.cumsum(sm.astype(np.int32))))
        cm = np.concatenate(([0], np.cumsum((m==1).astype(np.int32))))
        valid_starts = []
        for j in range(0, (b - a) - L + 1):
            adj_sum = c[j + (L-1)] - c[j]
            if adj_sum != (L - 1):
                continue
            # subset coverage: all rows in window must be in subset_mask
            count_subset = cs[j + L] - cs[j]
            if count_subset != L:
                continue
            # label coverage
            count_m = cm[j + L] - cm[j]
            if (count_m / float(L)) < min_valid_ratio:
                continue
            # Map back to original row-index (in unsorted order)
            # In sorted space, row position is (a + j) .. (a + j + L - 1)
            # We need the *first* row's original index, which is order[a + j]
            start_idx_orig = order[a + j]
            valid_starts.append(int(start_idx_orig))
        out[sym] = np.array(valid_starts, dtype=np.int64)
    return out

class UAMWindowDataset(Dataset):
    """
    Dataset that serves (L,F) windows aligned to next-step labels and masks.
    It assumes you will sample start indices with a Sampler or provide them explicitly.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray, L: int, start_indices: Sequence[int]):
        self.X = X; self.Y = Y; self.M = M
        self.L = int(L)
        self.starts = np.asarray(start_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        e = s + self.L
        x = torch.from_numpy(self.X[s:e])   # (L, F)
        # labels are aligned to each step; we use entire sequence with mask
        y = torch.from_numpy(self.Y[s:e])   # (L,)
        m = torch.from_numpy(self.M[s:e])   # (L,)
        return x, y, m

class BalancedPerSymbolSampler(Sampler[int]):
    """
    Per epoch, sample K start indices **per symbol** to balance UAM training.
    If a symbol has fewer than K candidates, sample with replacement.
    """
    def __init__(self, symbol_start_indices: Dict[int, np.ndarray], K_per_symbol: int, shuffle: bool = True, seed: int = 2025):
        self.symbol_start_indices = symbol_start_indices
        self.K = int(K_per_symbol)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.symbols = sorted(symbol_start_indices.keys())

    def __iter__(self):
        all_idx = []
        for sym in self.symbols:
            pool = self.symbol_start_indices[sym]
            if len(pool) == 0:
                continue
            if self.shuffle:
                pool = self.rng.permutation(pool)
            if len(pool) >= self.K:
                take = pool[:self.K]
            else:
                take = self.rng.choice(pool, size=self.K, replace=True)
            all_idx.append(take)
        if len(all_idx) == 0:
            return iter([])
        out = np.concatenate(all_idx)
        if self.shuffle:
            out = self.rng.permutation(out)
        return iter(out.tolist())

    def __len__(self) -> int:
        total = 0
        for pool in self.symbol_start_indices.values():
            total += max(len(pool), self.K if len(pool) > 0 else 0)
        return total

# Convenience function to compute T and L
def compute_L_from_config(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[int, int]:
    if cfg.bars_per_day is None:
        T = _infer_bars_per_day(df, cfg.date_col, cfg.time_col)
    else:
        T = int(cfg.bars_per_day)
    L = int(cfg.L_days) * int(T)
    return T, L
''')

deeplobv_code = dedent(r'''
# deeplobv_model.py
# -------------------------------------------
# DeepLOBv-style model: Causal Conv + Inception + LSTM + pointwise head
# Author: prepared by ChatGPT (GPT-5 Pro)
# -------------------------------------------

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _causal_pad(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1)

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__(in_ch, out_ch, k, padding=_causal_pad(k, dilation), dilation=dilation)
        self.k = k; self.d = dilation
    def forward(self, x):
        y = super().forward(x)
        trim = self.d * (self.k - 1)
        return y[..., :-trim] if trim > 0 else y

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(c_in, c_out, k, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out, kernels=(1,3,5,7), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(c_in, c_out, k),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ) for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(c_out*len(kernels), c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        outs = [b(x) for b in self.branches]
        y = torch.cat(outs, dim=1)
        return self.fuse(y)

class DeepLOBv(nn.Module):
    """
    Input:  (B, L, F)
    Output: (B, L)  # next-step regression aligned with each timestep (use mask in loss)
    """
    def __init__(self, in_features: int,
                 cnn_channels: int = 64,
                 cnn_layers: int = 2,
                 inception_channels: int = 64,
                 inception_layers: int = 1,
                 inception_kernels=(1,3,5,7),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, cnn_channels)

        convs = []
        for i in range(cnn_layers):
            convs.append(ConvBlock(cnn_channels, cnn_channels, k=5, dilation=2**i, dropout=dropout))
        self.conv_stack = nn.Sequential(*convs)

        incepts = []
        for _ in range(inception_layers):
            incepts.append(InceptionBlock(cnn_channels, inception_channels,
                                          kernels=inception_kernels, dropout=dropout))
        self.inception_stack = nn.Sequential(*incepts)

        self.lstm = nn.LSTM(input_size=inception_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden//2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None, h0=None):
        # x: (B, L, F)
        z = self.in_proj(x)         # (B, L, C)
        z = z.transpose(1, 2)       # (B, C, L)
        z = self.conv_stack(z)      # (B, C, L)
        z = self.inception_stack(z) # (B, C', L)
        z = z.transpose(1, 2)       # (B, L, C')
        z, _ = self.lstm(z, h0)     # (B, L, H)
        y = self.head(z).squeeze(-1)  # (B, L)
        return y if mask is None else y * mask
''')

# Write files
with open('/mnt/data/uam_data_prep.py', 'w', encoding='utf-8') as f:
    f.write(uam_prep_code)

with open('/mnt/data/deeplobv_model.py', 'w', encoding='utf-8') as f:
    f.write(deeplobv_code)

print("Files written:\n - /mnt/data/uam_data_prep.py\n - /mnt/data/deeplobv_model.py")
