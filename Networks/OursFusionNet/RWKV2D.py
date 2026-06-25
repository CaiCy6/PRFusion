import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

T_MAX = 256*256

class AdaptiveDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(AdaptiveDepthwiseSeparableConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.kernel_size = self.select_kernel_size(in_channels)

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                   kernel_size=self.kernel_size, stride=stride, padding=self.kernel_size // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def select_kernel_size(self, in_channels):
        
        if in_channels <= 64:
            return 3
        elif in_channels <= 128:
            return 5
        else:
            return 7

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


import os as _os
from torch.utils.cpp_extension import load
_cuda_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'cuda')
wkv_cuda = load(name="wkv", sources=[_os.path.join(_cuda_dir, "wkv_op.cpp"), _os.path.join(_cuda_dir, "wkv_cuda.cu")],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

class VRWKV_SpatialMix_Tri_Eff_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.directions = 3 

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.fusion_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.directions, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.directions, self.n_embd)))

    def shift_2d(self, input, shift_pixel=1, gamma=1/4, patch_resolution=None):

        B, N, C = input.shape
        H, W = patch_resolution
        # (B, N, C) -> (B, C, H, W)
        input = input.transpose(1, 2).reshape(B, C, H, W)
        output = torch.zeros_like(input)
        C_gamma = int(C * gamma)

        output[:, 0:C_gamma, :, shift_pixel:W] = input[:, 0:C_gamma, :, 0:W-shift_pixel]
        output[:, C_gamma:2*C_gamma, :, 0:W-shift_pixel] = input[:, C_gamma:2*C_gamma, :, shift_pixel:W]
        output[:, 2*C_gamma:3*C_gamma, shift_pixel:H, :] = input[:, 2*C_gamma:3*C_gamma, 0:H-shift_pixel, :]
        output[:, 3*C_gamma:4*C_gamma, 0:H-shift_pixel, :] = input[:, 3*C_gamma:4*C_gamma, shift_pixel:H, :]

        return output.flatten(2).transpose(1, 2)

    def jit_func(self, x, resolution):
        h, w = resolution
        # print(x.shape)
        # x = self.shift_2d(input=x, patch_resolution=resolution)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        h, w = resolution

        v1 = RUN_CUDA(B, T, C, self.spatial_decay[0] / T, self.spatial_first[0] / T, k, v)
        x = v1
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x
    
class VRWKV_ChannelMix_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

    def forward(self, x, resolution):
        h, w = resolution
        k = self.key(x)
        k = torch.square(torch.relu(k)) 
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        x = r * kv
        return x

class Block_2D(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.AdConv1 = AdaptiveDepthwiseSeparableConv(n_embd, n_embd)  # 第一个自适应深度卷积
        self.AdConv2 = AdaptiveDepthwiseSeparableConv(n_embd, n_embd)  # 第二个自适应深度卷积
        self.att = VRWKV_SpatialMix_Tri_Eff_2D(n_embd, n_layer, layer_id, init_mode, key_norm=key_norm)
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = VRWKV_ChannelMix_2D(n_embd, n_layer, layer_id, hidden_rate, init_mode, key_norm=key_norm)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        
        self.dwconv_branch = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)

        x = self.AdConv1(x)
        x_residual = x 
        x = rearrange(x, 'b c h w -> b (h w) c') 
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = self.AdConv2(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        dwconv_out = self.dwconv_branch(x_residual) 
        
        x = x + dwconv_out 
        return x


if __name__ == "__main__":
    
    input_tensor = torch.rand(1, 64, 256, 256).cuda()  

    block = Block_2D(n_embd=64, n_layer=12, layer_id=0).cuda()

    output_tensor = block(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
