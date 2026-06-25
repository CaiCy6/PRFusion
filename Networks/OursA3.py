import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

T_MAX = 256*256
'''
w/o channel mix 

'''

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
_cuda_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'OursFusionNet', 'cuda')
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
        
        # x = self.AdConv2(x)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        dwconv_out = self.dwconv_branch(x_residual) 
        
        x = x + dwconv_out 
        return x

class UNetFusionModel(nn.Module):
    def __init__(self, in_channels=1, n_layers=4, hidden_rate=4, key_norm=False):
        super().__init__()
        self.n_layers = n_layers
        self.channel_sizes = [32, 64, 128, 256]  # 四层通道数
        self.hidden_rate = hidden_rate
        self.key_norm = key_norm

        # ------------------------------
        # 编码器（两条路径，分别处理输入1和输入2）
        # ------------------------------
        self.encoders_path1 = nn.ModuleList()  # 路径1编码器
        self.encoders_path2 = nn.ModuleList()  # 路径2编码器
        self.downsamplers = nn.ModuleList()   # 下采样层（共享结构，两条路径各用一套参数）

        # 输入投影层（将单通道输入映射到第一个通道数32）
        self.input_proj1 = nn.Conv2d(in_channels, self.channel_sizes[0], kernel_size=3, padding=1, bias=False)
        self.input_proj2 = nn.Conv2d(in_channels, self.channel_sizes[0], kernel_size=3, padding=1, bias=False)

        for i in range(n_layers):
            # 编码器Block：每个编码器包含2个Block_2D
            current_channels = self.channel_sizes[i]
            encoder_block1 = nn.Sequential(
                Block_2D(current_channels, n_layers, layer_id=i, hidden_rate=hidden_rate, key_norm=key_norm),
                Block_2D(current_channels, n_layers, layer_id=i+1, hidden_rate=hidden_rate, key_norm=key_norm)
            )
            encoder_block2 = nn.Sequential(
                Block_2D(current_channels, n_layers, layer_id=i, hidden_rate=hidden_rate, key_norm=key_norm),
                Block_2D(current_channels, n_layers, layer_id=i+1, hidden_rate=hidden_rate, key_norm=key_norm)
            )
            self.encoders_path1.append(encoder_block1)
            self.encoders_path2.append(encoder_block2)

            # 下采样层（除了最后一层不需要下采样）
            if i < n_layers - 1:
                downsampler = nn.Conv2d(current_channels, self.channel_sizes[i+1], 
                                       kernel_size=2, stride=2, padding=0, bias=False)
                self.downsamplers.append(downsampler)

        # ------------------------------
        # 最深层融合模块
        # ------------------------------
        self.fusion_block = Block_2D(self.channel_sizes[-1], n_layers, layer_id=n_layers, 
                                    hidden_rate=hidden_rate, key_norm=key_norm)

        # ------------------------------
        # 解码器（单路径，接收融合后的特征）
        # ------------------------------
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()  # 调整跳跃连接的通道数

        for i in range(n_layers-2, -1, -1):
            # 上采样层：将通道数从当前层恢复到上一层
            up_channels = self.channel_sizes[i+1]
            target_channels = self.channel_sizes[i]
            upsampler = nn.ConvTranspose2d(up_channels, target_channels, 
                                          kernel_size=2, stride=2, padding=0, bias=False)
            self.upsamplers.append(upsampler)

            # 跳跃连接通道调整：拼接后通道数翻倍，需要调整回目标通道数
            self.skip_convs.append(nn.Conv2d(target_channels * 2, target_channels, 
                                           kernel_size=1, padding=0, bias=False))

            # 解码器Block：每个解码器包含2个Block_2D
            decoder_block = nn.Sequential(
                Block_2D(target_channels, n_layers, layer_id=i+n_layers, hidden_rate=hidden_rate, key_norm=key_norm),
                Block_2D(target_channels, n_layers, layer_id=i+n_layers+1, hidden_rate=hidden_rate, key_norm=key_norm)
            )
            self.decoders.append(decoder_block)

        # ------------------------------
        # 输出层
        # ------------------------------
        self.output_proj = nn.Conv2d(self.channel_sizes[0], 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2):
        """
        x1: 输入图像1，shape (B, 1, H, W)
        x2: 输入图像2，shape (B, 1, H, W)
        返回：融合后的图像，shape (B, 1, H, W)
        """
        # ------------------------------
        # 编码器阶段：两条路径并行提取特征
        # ------------------------------
        skip_connections1 = []  # 路径1的跳跃连接特征
        skip_connections2 = []  # 路径2的跳跃连接特征

        # 输入投影
        feat1 = self.input_proj1(x1)  # (B, 32, H, W)
        feat2 = self.input_proj2(x2)  # (B, 32, H, W)

        for i in range(self.n_layers):
            # 经过编码器Block
            feat1 = self.encoders_path1[i](feat1)
            feat2 = self.encoders_path2[i](feat2)

            # 保存跳跃连接特征（除了最深层）
            if i < self.n_layers - 1:
                skip_connections1.append(feat1)
                skip_connections2.append(feat2)

                # 下采样
                feat1 = self.downsamplers[i](feat1)
                feat2 = self.downsamplers[i](feat2)

        # ------------------------------
        # 最深层融合（仅在这里融合两条路径）
        # ------------------------------
        fused_feat = self.fusion_block(feat1 + feat2)  # (B, 256, H/8, W/8)

        # ------------------------------
        # 解码器阶段：上采样 + 跳跃连接
        # ------------------------------
        for i in range(self.n_layers-1):
            # 上采样
            fused_feat = self.upsamplers[i](fused_feat)  # 恢复空间尺寸

            # 获取对应的跳跃连接特征（两条路径的特征相加）
            skip1 = skip_connections1[-(i+1)]
            skip2 = skip_connections2[-(i+1)]
            skip_feat = skip1 + skip2  # (B, C, H', W')

            # 调整尺寸对齐（处理整除问题）
            if fused_feat.shape[2:] != skip_feat.shape[2:]:
                fused_feat = F.interpolate(fused_feat, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)

            # 拼接 + 通道调整
            fused_feat = torch.cat([fused_feat, skip_feat], dim=1)  # (B, 2C, H', W')
            fused_feat = self.skip_convs[i](fused_feat)  # (B, C, H', W')

            # 经过解码器Block
            fused_feat = self.decoders[i](fused_feat)

        # ------------------------------
        # 输出投影
        # ------------------------------
        output = self.output_proj(fused_feat)  # (B, 1, H, W)

        return output

# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 创建模型
    model = UNetFusionModel(in_channels=1, n_layers=4).cuda()
    
    # 创建测试输入（batch_size=1, 1通道, 256x256图像）
    x1 = torch.randn(1, 1, 256, 256).cuda()
    x2 = torch.randn(1, 1, 256, 256).cuda()
    
    # 前向传播
    with torch.no_grad():
        output = model(x1, x2)
    
    # 打印形状信息
    print(f"输入1形状: {x1.shape}")
    print(f"输入2形状: {x2.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证模型结构
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 打印编码器/解码器结构信息
    print(f"\n编码器层数: {len(model.encoders_path1)}")
    print(f"解码器层数: {len(model.decoders)}")
    print(f"通道数序列: {model.channel_sizes}")