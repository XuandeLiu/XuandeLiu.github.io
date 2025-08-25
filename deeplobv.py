# -*- coding: utf-8 -*-
# src/models/deeplobv.py
# -----------------------------------------------------------------------------
# DeepLOBv (适配 UAM 的日内成交量/对数成交量预测)
#
# 结构：
#   (B, L, F)  --Linear-->  (B, L, C)
#                      └──>  Conv1d causal stack (空洞卷积, 保持长度)
#                      └──>  Inception-like 多尺度因果卷积 (融合)
#                      └──>  LSTM (batch_first)  → 逐刻回归头
#   输出： (B, L)    # 每个时间步一个预测值；评估通常只看最后一格的“一步前瞻”
#
# 备注：
#   - 卷积均为**因果卷积**（右侧不看未来），padding = dilation*(k-1) 并剪掉右端，保持长度 L 不变。
#   - Inception 分支用不同 kernel size（默认 1/3/5/7）并融合，强化多尺度时序模式提取。
#   - LSTM 默认单层，隐藏维=128（可在构造函数里自定义）。
#   - Dropout + WeightDecay + Grad Clip + EarlyStopping 建议在训练脚本中配合启用。
#   - 如果你使用“稀疏窗口 + delta_g/gap_flag 特征”，本模型无需改动；直接作为额外特征进入 F。
#
# 作者：GPT-5 Pro（适配你的项目骨架）
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import math
import torch
import torch.nn as nn

__all__ = ["DeepLOBv"]

# -----------------------------------------------------------------------------
# 工具层：因果卷积（右侧不看未来，保持长度）
# -----------------------------------------------------------------------------

def _causal_pad_amount(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1)

class CausalConv1d(nn.Conv1d):
    """
    因果卷积：在右端进行“有效剪裁”，保证输出长度与输入一致，并且不泄漏未来信息。
    输入/输出张量： (B, C, L)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=_causal_pad_amount(kernel_size, dilation),
            bias=bias,
        )
        self._k = kernel_size
        self._d = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)                  # (B, C, L + pad)
        trim = _causal_pad_amount(self._k, self._d)
        return y[..., :-trim] if trim > 0 else y  # 去掉“看未来”的右侧 padding


# -----------------------------------------------------------------------------
# 基础块：卷积块（含 BatchNorm / GELU / Dropout）+ 残差
# -----------------------------------------------------------------------------

class ResidualCausalConvBlock(nn.Module):
    """
    单层因果卷积块 + 残差连接（通道不匹配时用 1x1 卷积对齐）
    输入/输出：(B, C_in, L) -> (B, C_out, L)
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = CausalConv1d(c_in, c_out, kernel_size, dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm1d(c_out)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res  = (
            nn.Identity() if c_in == c_out
            else nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y + self.res(x)


# -----------------------------------------------------------------------------
# Inception-like 模块：多尺度因果卷积分支 + 1x1 融合 + 残差
# -----------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """
    多尺度因果卷积分支（不同 kernel_size），再用 1x1 Conv 融合。
    输入：(B, C_in, L) → 输出：(B, C_out, L)
    """
    def __init__(self, c_in: int, c_out: int, kernels: Sequence[int] = (1, 3, 5, 7), dropout: float = 0.1):
        super().__init__()
        assert len(kernels) > 0, "kernels 至少需要一个元素"
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(c_in, c_out, k, dilation=1, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            )
            for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(c_out * len(kernels), c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res = (
            nn.Identity() if c_in == c_out
            else nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]          # list of (B, c_out, L)
        y = torch.cat(feats, dim=1)                    # (B, c_out * n_br, L)
        y = self.fuse(y)                               # (B, c_out, L)
        return y + self.res(x)                         # 残差


# -----------------------------------------------------------------------------
# DeepLOBv 主干
# -----------------------------------------------------------------------------

class DeepLOBv(nn.Module):
    """
    DeepLOBv：因果卷积 + Inception 多尺度 + LSTM + 逐刻回归头
    适用于：输入 (B, L, F)，输出 (B, L)

    参数
    ----
    in_features : int
        输入特征维度 F（与 DataLoader 输出保持一致；含你新增的 delta_g/gap_flag 等）。
    cnn_channels : int
        卷积通道数（线性投影后通道 C）。
    cnn_layers : int
        因果卷积层数；第 i 层使用 dilation=2**i（指数空洞扩大感受野）。
    inception_channels : int
        Inception 模块的输出通道数（融合后通道）。
    inception_layers : int
        Inception 模块堆叠层数。
    inception_kernels : Iterable[int]
        Inception 分支的 kernel 尺寸列表（如 (1,3,5,7)）。
    lstm_hidden : int
        LSTM 隐藏维度 H。
    lstm_layers : int
        LSTM 层数。
    dropout : float
        Dropout 比例（用于卷积/融合/头部 MLP）。
    """
    def __init__(
        self,
        in_features: int,
        cnn_channels: int = 64,
        cnn_layers: int = 2,
        inception_channels: int = 64,
        inception_layers: int = 1,
        inception_kernels: Iterable[int] = (1, 3, 5, 7),
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert in_features > 0, "in_features 必须为正"

        # 线性投影：F -> C（便于卷积）
        self.in_proj = nn.Linear(in_features, cnn_channels)

        # 因果卷积堆叠（指数空洞，扩大感受野）
        convs = []
        for i in range(cnn_layers):
            convs.append(
                ResidualCausalConvBlock(
                    c_in=cnn_channels,
                    c_out=cnn_channels,
                    kernel_size=5,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
        self.conv_stack = nn.Sequential(*convs) if len(convs) > 0 else nn.Identity()

        # Inception 堆叠
        incepts = []
        c_in = cnn_channels
        for _ in range(inception_layers):
            incepts.append(InceptionBlock(c_in, inception_channels, kernels=tuple(inception_kernels), dropout=dropout))
            c_in = inception_channels
        self.inception_stack = nn.Sequential(*incepts) if len(incepts) > 0 else nn.Identity()
        c_after = inception_channels if len(incepts) > 0 else cnn_channels

        # LSTM
        self.lstm = nn.LSTM(
            input_size=c_after,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(0.0 if lstm_layers == 1 else dropout),
            bidirectional=False,
        )

        # 逐刻回归头（小 MLP）
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, max(8, lstm_hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, lstm_hidden // 2), 1),
        )

        # 参数初始化
        self.apply(self._init_weights)
        self._init_lstm_forget_bias()

    # -------------------------------------------------------------------------
    # 前向
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x : (B, L, F)  # 由 DataLoader 提供
        mask : (B, L) 或 None
            若提供，仅用于在中间阶段做可选的掩码（一般不需要；损失侧已有掩码）。
        return : (B, L)
        """
        assert x.dim() == 3, f"期望 (B, L, F)，得到 {tuple(x.shape)}"
        B, L, F = x.shape

        # 线性投影 + (B,L,C) -> (B,C,L) 适配 Conv1d
        z = self.in_proj(x)                  # (B,L,C)
        z = z.transpose(1, 2)                # (B,C,L)

        # 因果卷积堆叠
        z = self.conv_stack(z)               # (B,C,L)

        # Inception 堆叠
        z = self.inception_stack(z)          # (B,C',L)

        # 回到 (B,L,C')
        z = z.transpose(1, 2)                # (B,L,C')

        # （可选）在进入 LSTM 前应用 mask，将“无效步”置零；通常不需要。
        if mask is not None:
            z = z * mask.unsqueeze(-1).to(z.dtype)

        # LSTM
        z, _ = self.lstm(z)                  # (B,L,H)

        # 回归头
        y = self.head(z).squeeze(-1)         # (B,L)
        return y

    # -------------------------------------------------------------------------
    # 助手：初始化
    # -------------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_lstm_forget_bias(self):
        """
        LSTM 的 forget gate bias 设置为正，有助于早期训练稳定。
        """
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                # PyTorch LSTM bias 结构: [b_ii | b_if | b_ig | b_io]
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)  # forget gate bias = 1.0

    # -------------------------------------------------------------------------
    # 便捷：估算感受野（用于判断 L 是否足够覆盖历史）
    # -------------------------------------------------------------------------
    def receptive_field(self, inception_kernels: Sequence[int] | None = None) -> int:
        """
        估算模型“单点输出”所依赖的历史步数（感受野，单位=步）。
        - 卷积堆叠：sum_i ( (k-1) * dilation_i )
        - Inception：取各分支中最大的 kernel 对应的 (k-1)，每层叠加一次
        """
        rf = 1
        # 卷积堆叠（与构造保持一致）
        # 这里读取 conv_stack 内每层的 dilation & kernel
        for m in self.conv_stack.modules():
            if isinstance(m, CausalConv1d):
                rf += _causal_pad_amount(m._k, m._d)
        # Inception
        if inception_kernels is None:
            max_k = 7  # 与默认 (1,3,5,7) 一致
        else:
            max_k = max(inception_kernels)
        inc_layers = sum(isinstance(m, InceptionBlock) for m in self.inception_stack.modules())
        rf += (max_k - 1) * max(1, inc_layers)
        return int(rf)


# -----------------------------------------------------------------------------
# 自检：本文件直接运行时做一次 shape 检查（导入时不会执行）
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, L, F = 8, 670, 128
    x = torch.randn(B, L, F)
    model = DeepLOBv(in_features=F)
    y = model(x)
    print("input:", x.shape, "output:", y.shape, "RF≈", model.receptive_field())
