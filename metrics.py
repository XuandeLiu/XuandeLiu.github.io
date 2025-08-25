# src/utils/metrics.py
import torch

def _nan2num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def masked_mse(yh, y, m):
    yh = _nan2num(yh); y = _nan2num(y); m = _nan2num(m)
    denom = m.sum()
    if denom <= 0:
        return torch.zeros((), device=yh.device)
    num = ((yh - y)**2 * m).sum()
    if torch.isnan(num):
        return torch.zeros((), device=yh.device)
    return num / (denom + 1e-8)

def masked_mae(yh, y, m):
    # 与 masked_mse 一致的清洗与加权口径
    yh = _nan2num(yh); y = _nan2num(y); m = _nan2num(m)
    denom = m.sum()
    if denom <= 0:
        return torch.zeros((), device=yh.device)
    num = (torch.abs(yh - y) * m).sum()
    return num / (denom + 1e-8)

@torch.no_grad()
def r2_last_global(model, loader, device: str):
    model.eval()
    sum_m = 0.0; sum_y = 0.0; sum_y2 = 0.0; sum_res = 0.0; sum_abs = 0.0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        # 清洗 batch
        xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        mb = torch.nan_to_num(mb, nan=0.0, posinf=0.0, neginf=0.0)

        yh = model(xb)
        yh = torch.nan_to_num(yh, nan=0.0, posinf=0.0, neginf=0.0)
        y_last, yh_last, m_last = yb[:, -1], yh[:, -1], mb[:, -1]
        m_last = torch.nan_to_num(m_last, nan=0.0)

        sum_m  += m_last.sum().item()
        sum_y  += (y_last * m_last).sum().item()
        sum_y2 += ((y_last ** 2) * m_last).sum().item()
        sum_res += (((yh_last - y_last) ** 2) * m_last).sum().item()
        sum_abs += (torch.abs(yh_last - y_last) * m_last).sum().item()

    if sum_m <= 0:
        # 极端情况下整个验证集最后一格都不可评
        return {"r2_last": 0.0, "mse_last": float("nan"), "mae_last": float("nan")}

    denom = max(sum_y2 - (sum_y ** 2) / (sum_m + 1e-8), 1e-8)
    r2  = 1.0 - (sum_res / denom)
    mse = sum_res / sum_m
    mae = sum_abs / sum_m
    return {"r2_last": float(r2), "mse_last": float(mse), "mae_last": float(mae)}
