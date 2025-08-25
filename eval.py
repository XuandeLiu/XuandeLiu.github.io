# src/eval.py
from __future__ import annotations
import os, argparse
import torch
import pandas as pd

from config import load_config, ensure_dirs
from utils.common import set_seed, device_auto, save_json
from models.deeplobv import DeepLOBv
from data.uam_threeway_prep import ThreeWayConfig, make_threeway_loaders

from utils.metrics import r2_last_global
#
# @torch.no_grad()
# def eval_loader_global(model: torch.nn.Module, loader, device: str):
#     """
#     全局聚合的一步前瞻指标（只看每个窗口最后一格）：
#       - R2(last)  = 1 - SS_res / SS_tot
#       - MSE(last) = SS_res / sum(mask)
#       - MAE(last) = sum(|err|*mask) / sum(mask)
#     其中 mask = 每个样本最后一步是否可评。
#     """
#     model.eval()
#     sum_m = 0.0
#     sum_y = 0.0
#     sum_y2 = 0.0
#     sum_res = 0.0
#     sum_abs = 0.0
#
#     for xb, yb, mb in loader:
#         xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
#         yh = model(xb)
#
#         y_last  = yb[:, -1]
#         yh_last = yh[:, -1]
#         m_last  = mb[:, -1]
#
#         # for R2
#         sum_m  += m_last.sum().item()
#         sum_y  += (y_last * m_last).sum().item()
#         sum_y2 += ((y_last ** 2) * m_last).sum().item()
#         sum_res += (((yh_last - y_last) ** 2) * m_last).sum().item()
#
#         # for MAE
#         sum_abs += (torch.abs(yh_last - y_last) * m_last).sum().item()
#
#     # 全局均值与方差（加权）
#     denom = max(sum_y2 - (sum_y ** 2) / (sum_m + 1e-8), 1e-8)
#     r2  = 1.0 - (sum_res / denom)
#     mse = sum_res / max(sum_m, 1e-8)
#     mae = sum_abs / max(sum_m, 1e-8)
#     return {"r2_last": float(r2), "mse_last": float(mse), "mae_last": float(mae)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--ckpt', type=str, default='outputs/checkpoints/best.pt')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg['seed'])
    device = device_auto()

    # ---- 构造 DataLoader（与训练保持同一配置）----
    dc = cfg['data']
    twc = ThreeWayConfig(
        date_col=dc['date_col'], time_col=dc['time_col'], symbol_col=dc['symbol_col'], y_col=dc['y_col'],
        inner_split_date=dc['inner_split_date'], L_days=dc['L_days'], bars_per_day=dc['bars_per_day'],
        min_valid_ratio=dc['min_valid_ratio'], use_sparse=dc['use_sparse'],
        add_delta_g_feature=dc['add_delta_g_feature'], delta_cap=dc['delta_cap'],
        K_per_symbol=dc['K_per_symbol'], batch_size=dc['batch_size'],
        num_workers=dc['num_workers'], pin_memory=dc['pin_memory']
    )

    training_df = pd.read_parquet('data/interim/training.parquet')
    testing_df  = pd.read_parquet('data/interim/testing.parquet')
    train_loader, inner_test_loader, outer_test_loader, meta = make_threeway_loaders(training_df, testing_df, twc)

    # ---- 构建模型并加载权重 ----
    model = DeepLOBv(in_features=meta['X'].shape[1], **cfg['model']).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- 评估（全局聚合口径）----
    # inner = eval_loader_global(model, inner_test_loader, device)
    # outer = eval_loader_global(model, outer_test_loader, device)
    #
    # print(
    #     f"Inner-split  | R2(last)={inner['r2_last']:.6f}  "
    #     f"MSE(last)={inner['mse_last']:.6f}  MAE(last)={inner['mae_last']:.6f}"
    # )
    # print(
    #     f"Outer-test   | R2(last)={outer['r2_last']:.6f}  "
    #     f"MSE(last)={outer['mse_last']:.6f}  MAE(last)={outer['mae_last']:.6f}"
    # )

    metrics_inner = r2_last_global(model, inner_test_loader, device)
    metrics_outer = r2_last_global(model, outer_test_loader, device)

    print(
        f"Inner-split  | R2(last)={metrics_inner['r2_last']:.6f}  "
        f"MSE(last)={metrics_inner['mse_last']:.6f}  MAE(last)={metrics_inner['mae_last']:.6f}"
    )
    print(
        f"Outer-test   | R2(last)={metrics_outer['r2_last']:.6f}  "
        f"MSE(last)={metrics_outer['mse_last']:.6f}  MAE(last)={metrics_outer['mae_last']:.6f}"
    )
    # ---- 落盘指标 ----
    save_json(
        {
            "in_features": int(meta['X'].shape[1]),
            "T": int(meta['T']),
            "L": int(meta['L']),
            "n_inner_windows": len(inner_test_loader.dataset),
            "n_outer_windows": len(outer_test_loader.dataset),
            "inner": inner,
            "outer": outer,
        },
        os.path.join(cfg['paths']['metrics_dir'], 'eval_summary.json')
    )


if __name__ == '__main__':
    main()
