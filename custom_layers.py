"""
手写卷积和池化层，使用 torch 基础操作（unfold + matmul/max/mean）实现。
不调用 nn.Conv2d / F.conv2d / nn.MaxPool2d 等现成卷积/池化接口。
所有操作均为 torch tensor 运算，autograd 可自动追踪梯度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ManualConv2d(nn.Module):
    """
    用 im2col (unfold + matmul) 手写二维卷积。
    weight: (out_channels, in_channels, kH, kW)，nn.Parameter，autograd 追踪。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride       = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding      = padding if isinstance(padding, tuple) else (padding, padding)

        kH, kW = self.kernel_size
        # Kaiming 均匀初始化
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kH * kW
            bound  = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))
        else:
            self.bias = None

    def forward(self, x):
        # x: (N, C_in, H, W)
        N, C_in, H, W = x.shape
        kH, kW   = self.kernel_size
        sH, sW   = self.stride
        pH, pW   = self.padding

        # 1. zero padding（F.pad 是函数，不是卷积层，符合要求）
        if pH > 0 or pW > 0:
            x = F.pad(x, (pW, pW, pH, pH))

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        # 2. im2col：unfold 沿 H 和 W 方向展开 patch
        # x_unf: (N, C_in, H_out, W_out, kH, kW)
        x_unf = x.unfold(2, kH, sH).unfold(3, kW, sW)
        # permute -> (N, H_out, W_out, C_in, kH, kW)，再 reshape -> (N, H_out*W_out, C_in*kH*kW)
        x_col = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous().view(N, H_out * W_out, C_in * kH * kW)

        # 3. weight reshape: (out_channels, C_in * kH * kW)
        w = self.weight.view(self.out_channels, -1)

        # 4. matmul: (N, H_out*W_out, out_channels) -> permute -> (N, out_channels, H_out*W_out)
        out = (x_col @ w.T).permute(0, 2, 1)

        # 5. reshape 回 (N, out_channels, H_out, W_out)
        out = out.view(N, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}")


class ManualMaxPool2d(nn.Module):
    """
    用 unfold + max 手写最大池化。
    torch 的 max 操作可微，autograd 自动处理反向传播。
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if stride is None:
            stride = self.kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x):
        # x: (N, C, H, W)
        kH, kW = self.kernel_size
        sH, sW = self.stride

        # unfold: (N, C, H_out, W_out, kH, kW)
        x_unf = x.unfold(2, kH, sH).unfold(3, kW, sW)
        # reshape: (N, C, H_out, W_out, kH*kW)
        x_unf = x_unf.contiguous().view(*x_unf.shape[:4], kH * kW)
        # max over last dim
        return x_unf.max(dim=-1).values

    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}"


class ManualAvgPool2d(nn.Module):
    """
    用 unfold + mean 手写平均池化（含 Global Average Pooling）。
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if stride is None:
            stride = self.kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x):
        kH, kW = self.kernel_size
        sH, sW = self.stride
        x_unf  = x.unfold(2, kH, sH).unfold(3, kW, sW)
        x_unf  = x_unf.contiguous().view(*x_unf.shape[:4], kH * kW)
        return x_unf.mean(dim=-1)

    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}"


# ─────────────────────────── 单元测试 ───────────────────────────

if __name__ == '__main__':
    import torch.nn.functional as F_ref

    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)

    # 验证 ManualConv2d 与 F.conv2d 数值一致
    conv = ManualConv2d(3, 8, kernel_size=3, stride=1, padding=1)
    out_manual = conv(x)

    out_ref = F_ref.conv2d(x, conv.weight, conv.bias, stride=1, padding=1)
    diff = (out_manual - out_ref).abs().max().item()
    print(f"ManualConv2d vs F.conv2d max diff: {diff:.2e}  {'PASS' if diff < 1e-5 else 'FAIL'}")

    # 验证梯度可以反向传播
    out_manual.sum().backward()
    print(f"ManualConv2d gradient OK: weight.grad shape = {conv.weight.grad.shape}")

    # 验证 ManualMaxPool2d
    pool = ManualMaxPool2d(2, 2)
    xp   = torch.randn(2, 8, 8, 8, requires_grad=True)
    out_pool = pool(xp)
    out_pool.sum().backward()
    print(f"ManualMaxPool2d output shape: {out_pool.shape}, grad OK: {xp.grad is not None}")

    # 验证 ManualAvgPool2d（Global）
    gap = ManualAvgPool2d(8, 8)
    out_gap = gap(torch.randn(2, 64, 8, 8))
    print(f"ManualAvgPool2d (global) output shape: {out_gap.shape}")
