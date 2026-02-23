import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from Ablation import *


class DSConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out) 
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)

class GateLayer(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim

        self.net_depth = net_depth
        self.kernel_size = kernel_size
        self.reduction_dim = max(4, dim//8)
        
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect'),
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, self.reduction_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduction_dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )

        self.proj = DSConv(dim, dim, kernel_size, 1, kernel_size//2, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out


class LGRBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, conv_layer=GateLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
        super().__init__()
        self.norm = norm_layer(dim)
        self.conv = GateLayer(net_depth, dim, kernel_size, gate_act)
    
    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, net_depth, dim, depth, kernel_size=3, 
                 conv_layer=GateLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            LGRBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
            for i in range(depth)
        ])

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PAFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(PAFusion, self).__init__()
        self.dim = dim
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Independent PWConv (1x1 Conv) for Encoder and Decoder features
        self.pw_enc = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.pw_dec = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        
        self.act = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)

    def forward(self, in_feats):
        
        up_feat = in_feats[0]
        skip_feat = in_feats[1]
        
        gap_up = self.avg_pool(up_feat)     # (B, C, 1, 1)
        gap_skip = self.avg_pool(skip_feat) # (B, C, 1, 1)
        
        feat_up = self.pw_dec(gap_up)       # (B, C, 1, 1)
        feat_skip = self.pw_enc(gap_skip)   # (B, C, 1, 1)
        
        fused_stats = feat_up + feat_skip
        fused_stats = self.act(fused_stats)
        
        # affine_params shape: (B, 2C, 1, 1)
        affine_params = self.final_conv(fused_stats)
        
        gamma, beta = torch.chunk(affine_params, 2, dim=1)
        
        # Apply Affine Transformation to Skip Feature
        skip_aligned = skip_feat * gamma + beta
        out = up_feat + skip_aligned
        
        return out


class PGLNet(nn.Module):
    def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=GateLayer, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid, fusion_layer=PAFusion):
        super(PGLNet, self).__init__()
        # setting
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2**i*base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num

        # input convolution
        self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(self.stage_num):
            self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                                      conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num):
            self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.fusions.append(fusion_layer(embed_dims[i]))

        # output convolution
        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)


    def forward(self, x):
        feat = self.inconv(x)

        skips = []

        for i in range(self.half_num):
            feat = self.layers[i](feat)
            skips.append(self.skips[i](feat))
            feat = self.downs[i](feat)

        feat = self.layers[self.half_num](feat)

        for i in range(self.half_num-1, -1, -1):
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, skips[i]])
            feat = self.layers[self.stage_num-i-1](feat)

        x = self.outconv(feat) + x

        return x


__all__ = ['PGLNet', 'pglnet_t', 'pglnet_s', 'pglnet_b', 'pglnet_d']

def pglnet_t():
    return PGLNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=GateLayer,
                 norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=PAFusion)

def pglnet_s():
    return PGLNet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=GateLayer,
                 norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=PAFusion)

def pglnet_b():
    return PGLNet(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=GateLayer,
                 norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=PAFusion)

def pglnet_d():
    return PGLNet(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=GateLayer,
                 norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=PAFusion)
