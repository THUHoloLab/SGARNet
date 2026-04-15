# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import math
import torch
from torch import nn as nn
from torch.nn import init as init



@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)




def hex_peak_mask(H, W, a_px, bandwidth=0.06, harmonics=(1,2), device=None):
    import math, torch
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fu = torch.fft.fftshift(torch.linspace(-0.5, 0.5 - 1.0/W, W, device=device))
    fv = torch.fft.fftshift(torch.linspace(-0.5, 0.5 - 1.0/H, H, device=device))
    U, V = torch.meshgrid(fu, fv, indexing='xy')  # [W,H]
    f0 = 1.0 / max(float(a_px), 1e-6)
    angles = [0, 60, 120, 180, 240, 300]
    M = torch.zeros((W, H), device=device)

    def g2d(U, V, uc, vc, s):
        return torch.exp(-((U-uc)**2 + (V-vc)**2) / (2*s*s))

    def wrap_nyq(t):  # wrap 到 [-0.5, 0.5)
        return ((t + 0.5) % 1.0) - 0.5

    s = bandwidth * 0.5
    for n in harmonics:
        fn = n * f0
        for ang in angles:
            rad = math.radians(ang)
            uc_raw, vc_raw = fn*math.cos(rad), fn*math.sin(rad)

            for sign in (1.0, -1.0):
                uc = wrap_nyq(sign * uc_raw)
                vc = wrap_nyq(sign * vc_raw)
                M += g2d(U, V, uc, vc, s)

    M = M / (M.max() + 1e-8)
    return torch.fft.ifftshift(M).T  # [H,W]



class SpectralGate2D(nn.Module):

    def __init__(self, channels, a_px, bandwidth=0.06, harmonics=(1,2),
                 alpha_max=0.7, init_alpha=0.0, per_channel=True):
        super().__init__()
        self.a_px = float(a_px)
        self.bandwidth = float(bandwidth)
        self.harmonics = harmonics
        self.alpha_max = float(alpha_max)
        self.per_channel = per_channel

        if per_channel:
            self.alpha_raw = nn.Parameter(torch.full((channels,1,1), float(init_alpha)))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor([[ [float(init_alpha)] ]]))  # [1,1,1]

        self._mask = None
        self._mask_shape = None
        self._mask_device = None

    def _get_mask(self, H, W, device):
        if (self._mask is None) or (self._mask_shape != (H, W)) or (self._mask_device != device):
            M = hex_peak_mask(H, W, a_px=self.a_px, bandwidth=self.bandwidth,
                              harmonics=self.harmonics, device=device)
            self._mask = M
            self._mask_shape = (H, W)
            self._mask_device = device
        return self._mask

    def forward(self, x):
        """
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape
        device = x.device
        M = self._get_mask(H, W, device)                  # [H,W]
        alpha = torch.sigmoid(self.alpha_raw) * self.alpha_max
        if alpha.shape[0] != C:
            alpha = alpha.expand(C, 1, 1)

        dtype_in = x.dtype
        X = torch.fft.fft2(x.to(torch.float32), norm='ortho')     # complex64
        gate = (1.0 - alpha * M[None, None, ...])                 # [C,H,W]
        Y = X * gate                                              # [B,C,H,W]
        Z = torch.fft.ifft2(Y, norm='ortho')  # complex
        y = Z.real.contiguous()  # real part
        return y.to(dtype_in)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


import time
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps