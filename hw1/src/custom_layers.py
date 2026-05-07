import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ManualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride       = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding      = padding if isinstance(padding, tuple) else (padding, padding)

        kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kH * kW
            bound  = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))
        else:
            self.bias = None

    def forward(self, x):
        N, C_in, H, W = x.shape
        kH, kW   = self.kernel_size
        sH, sW   = self.stride
        pH, pW   = self.padding

        if pH > 0 or pW > 0:
            x = F.pad(x, (pW, pW, pH, pH))

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        x_unf = x.unfold(2, kH, sH).unfold(3, kW, sW)
        x_col = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous().view(N, H_out * W_out, C_in * kH * kW)

        w = self.weight.view(self.out_channels, -1)
        out = (x_col @ w.T).permute(0, 2, 1)
        out = out.view(N, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}")


class ManualMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if stride is None:
            stride = self.kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x):
        kH, kW = self.kernel_size
        sH, sW = self.stride
        x_unf = x.unfold(2, kH, sH).unfold(3, kW, sW)
        x_unf = x_unf.contiguous().view(*x_unf.shape[:4], kH * kW)
        return x_unf.max(dim=-1).values

    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}"


class ManualAvgPool2d(nn.Module):
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


if __name__ == '__main__':
    import torch.nn.functional as F_ref

    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)

    conv = ManualConv2d(3, 8, kernel_size=3, stride=1, padding=1)
    out_manual = conv(x)

    out_ref = F_ref.conv2d(x, conv.weight, conv.bias, stride=1, padding=1)
    diff = (out_manual - out_ref).abs().max().item()
    print(f"ManualConv2d vs F.conv2d max diff: {diff:.2e}  {'PASS' if diff < 1e-5 else 'FAIL'}")

    out_manual.sum().backward()
    print(f"ManualConv2d gradient OK: weight.grad shape = {conv.weight.grad.shape}")

    pool = ManualMaxPool2d(2, 2)
    xp   = torch.randn(2, 8, 8, 8, requires_grad=True)
    out_pool = pool(xp)
    out_pool.sum().backward()
    print(f"ManualMaxPool2d output shape: {out_pool.shape}, grad OK: {xp.grad is not None}")

    gap = ManualAvgPool2d(8, 8)
    out_gap = gap(torch.randn(2, 64, 8, 8))
    print(f"ManualAvgPool2d (global) output shape: {out_gap.shape}")
