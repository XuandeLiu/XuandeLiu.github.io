# src/train.py
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.amp import GradScaler
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from config import load_config, ensure_dirs
from utils.common import set_seed, device_auto, save_json
from utils.logging import get_logger
from utils.metrics import masked_mse, r2_last_global
from models.deeplobv import DeepLOBv
from data.uam_threeway_prep import ThreeWayConfig, make_threeway_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg['seed'])
    device = device_auto()

    # ---- 1) 读数据 ----
    training_df = pd.read_parquet('data/interim/training.parquet')
    testing_df  = pd.read_parquet('data/interim/testing.parquet')

    # ---- 2) 构造 DataLoader（一次，后面训练时每 epoch 重采样 train）----
    dc = cfg['data']
    twc = ThreeWayConfig(
        date_col=dc['date_col'], time_col=dc['time_col'], symbol_col=dc['symbol_col'], y_col=dc['y_col'],
        inner_split_date=dc['inner_split_date'], L_days=dc['L_days'], bars_per_day=dc['bars_per_day'],
        min_valid_ratio=dc['min_valid_ratio'], use_sparse=dc['use_sparse'],
        add_delta_g_feature=dc['add_delta_g_feature'], delta_cap=dc['delta_cap'],
        K_per_symbol=dc['K_per_symbol'], batch_size=dc['batch_size'],
        num_workers=dc['num_workers'], pin_memory=dc['pin_memory']
    )
    # 先构一次，拿 meta 与评估全集
    train_loader, inner_test_loader, outer_test_loader, meta = make_threeway_loaders(training_df, testing_df, twc)

    in_features, T, L = meta['X'].shape[1], meta['T'], meta['L']
    logger = get_logger('train', cfg['paths']['logs_dir'])
    logger.info(f"in_features={in_features}, T={T}, L={L}, "
                f"train_windows={len(train_loader.dataset)}, "
                f"inner_test_windows={len(inner_test_loader.dataset)}, "
                f"outer_test_windows={len(outer_test_loader.dataset)}")
    logger.info(f"device={device} | cuda={torch.cuda.is_available()}")

    # ---- 2.1) 构造固定 dev 子集 + 大 batch 的评估器（从 YAML 读取阈值）----
    ev = cfg.get('eval', {})
    eval_bs = max(int(ev.get('eval_batch_size', 256)), dc['batch_size'])
    max_dev = int(ev.get('dev_max_windows', 100_000))
    idx_all = np.arange(len(inner_test_loader.dataset))
    rng = np.random.default_rng(42)
    keep = idx_all if len(idx_all) <= max_dev else rng.choice(idx_all, size=max_dev, replace=False)
    inner_dev_loader = DataLoader(
        Subset(inner_test_loader.dataset, keep.tolist()),
        batch_size=eval_bs, shuffle=False,
        num_workers=dc['num_workers'], pin_memory=dc['pin_memory']
    )
    full_eval_every = int(ev.get('full_eval_every', 5))

    # ---- 3) 模型 / 优化器 / 调度 ----
    model = DeepLOBv(in_features=in_features, **cfg['model']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    sch_cfg = cfg['train']['scheduler']
    scheduler = None
    if sch_cfg['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sch_cfg['t_max'], eta_min=sch_cfg['min_lr'])
    elif sch_cfg['name'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                               factor=sch_cfg['factor'], patience=sch_cfg['patience'],
                                                               min_lr=sch_cfg['min_lr'])
    scaler = GradScaler(enabled=cfg['train']['amp'])

    # ---- 4) 评估函数（带进度条，全局聚合 R²）----
    @torch.no_grad()
    def eval_loader(loader) -> float:
        model.eval()
        sum_m = sum_y = sum_y2 = sum_res = 0.0
        for xb, yb, mb in tqdm(loader, desc='[eval]', leave=False):
            xb, yb, mb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            mb = torch.nan_to_num(mb, nan=0.0, posinf=0.0, neginf=0.0)
            yh = model(xb)
            yh = torch.nan_to_num(yh, nan=0.0, posinf=0.0, neginf=0.0)

            y_last, yh_last, m_last = yb[:, -1], yh[:, -1], mb[:, -1]
            sum_m  += m_last.sum().item()
            sum_y  += (y_last * m_last).sum().item()
            sum_y2 += ((y_last ** 2) * m_last).sum().item()
            sum_res += (((yh_last - y_last) ** 2) * m_last).sum().item()
        if sum_m <= 0:
            return 0.0
        denom = max(sum_y2 - (sum_y ** 2)/(sum_m + 1e-8), 1e-8)
        return 1.0 - (sum_res / denom)

    # ---- 5) 训练循环（每个 epoch 重采样 train，dev-subset 早停，全量偶尔评）----
    best_metric, best_state = -1e9, None
    patience = cfg['train']['early_stopping']['patience']
    bad = 0
    epochs = cfg['train']['epochs']

    for ep in range(1, epochs + 1):
        # ★ 重采样训练起点（最省事：重建 train_loader；inner/outer 测试不变）
        train_loader, _, _, _ = make_threeway_loaders(training_df, testing_df, twc)

        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {ep}/{epochs}')
        for xb, yb, mb in pbar:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            mb = torch.nan_to_num(mb, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=cfg['train']['amp']):
                yh = model(xb)
                loss = masked_mse(yh, yb, mb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))

        # dev 子集做早停评估
        val_r2 = eval_loader(inner_dev_loader)
        logger.info(f'Epoch {ep} | inner_dev R2(last) = {val_r2:.6f}')

        # 每 full_eval_every 轮做一次全量 inner 评估（仅日志）
        if ep % full_eval_every == 0:
            full_r2 = eval_loader(inner_test_loader)
            logger.info(f'Epoch {ep} | inner_full R2(last) = {full_r2:.6f}')

        # 调度器（首轮不 step，避免 Warning）
        if scheduler is not None and ep > 1:
            if sch_cfg['name'] == 'plateau':
                scheduler.step(val_r2)
            else:
                scheduler.step()

        # 早停
        if val_r2 > best_metric + cfg['train']['early_stopping']['min_delta']:
            best_metric, bad = val_r2, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(cfg['paths']['checkpoints_dir'], 'best.pt'))
        else:
            bad += 1
            if cfg['train']['early_stopping']['enabled'] and bad >= patience:
                logger.info('Early stopping triggered.')
                break

    save_json({'best_inner_dev_r2': best_metric}, os.path.join(cfg['paths']['metrics_dir'], 'train_summary.json'))
    logger.info(f'Best inner_dev R2(last): {best_metric:.6f} (checkpoint saved).')


if __name__ == '__main__':
    main()
