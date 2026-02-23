import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionGate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

# =========================================================================
# Fusion module: PW_Fusion_Concat (Channel-wise Concatenation)
# =========================================================================
class ConcatFusion(nn.Module):
    """
    Fusion module using simple Channel-wise Concatenation.
    
    Structure: 
    1. Concatenate Up-stream (Decoder) and Skip-stream (Encoder) features along channel dimension.
    2. Apply 1x1 Conv to adjust channel dimension back to original.
    """
    def __init__(self, dim, height=2, reduction=8):
        # height 和 reduction 参数保留是为了兼容原接口调用
        super(ConcatFusion, self).__init__()
        self.dim = dim
        
        # 拼接后通道数为 2*dim，使用 1x1 卷积调整回 dim
        self.conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=True)

    def forward(self, in_feats):
        # gUNet 传进来的 in_feats 是 [feat(Up), skips[i](Skip)]
        up_feat = in_feats[0]
        skip_feat = in_feats[1]
        
        # 通道拼接
        fused = torch.cat([up_feat, skip_feat], dim=1)
        
        # 调整通道数回原来的维度
        out = self.conv(fused)
        
        return out

class SummationFusion(nn.Module):
    """
    Simple summation fusion for skip connections
    """
    def __init__(self, dim, *args, **kwargs):
        super(SummationFusion, self).__init__()
        # 保持接口兼容，但不使用任何参数
        self.dim = dim

    def forward(self, in_feats):
        # 简单地将两个特征相加
        # in_feats 是 [decoder_feat, skip_feat]
        decoder_feat = in_feats[0]
        skip_feat = in_feats[1]
        return decoder_feat + skip_feat


# =========================================================================
# Fusion module: PW_Fusion_Attention (Using AttentionGate for skip connection)
# =========================================================================
class AGFusion(nn.Module):
    """
    Fusion module using AttentionGate mechanism from Attention U-Net.
    
    Structure: 
    1. Use AttentionGate to generate attention weights for skip connection features.
    2. Apply attention weights to skip connection features.
    3. Concatenate the weighted skip features with decoder features.
    4. Use 1x1 Conv to adjust channel dimension back to original.
    """
    def __init__(self, dim, height=2, reduction=8):
        # height 和 reduction 参数保留是为了兼容原接口调用
        super(AGFusion, self).__init__()
        self.dim = dim
        
        # 使用 AttentionGate，F_g 和 F_l 都是 dim，F_int 设置为 dim//2 以减少计算量
        self.attention_gate = AttentionGate(dim, dim, dim//2)
        
        # 拼接后通道数为 2*dim，使用 1x1 卷积调整回 dim
        self.conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=True)

    def forward(self, in_feats):
        # gUNet 传进来的 in_feats 是 [feat(Up), skips[i](Skip)]
        up_feat = in_feats[0]  # Decoder features
        skip_feat = in_feats[1]  # Encoder skip features
        
        # 使用 AttentionGate 生成注意力加权后的跳跃连接特征
        # 注意：AttentionGate的forward方法参数是(g, x)，其中g是gate信号(来自decoder)，x是要加权的特征(来自encoder)
        att_weighted_skip = self.attention_gate(up_feat, skip_feat)
        
        # 将注意力加权后的跳跃连接特征与解码器特征进行通道拼接
        fused = torch.cat([up_feat, att_weighted_skip], dim=1)
        
        # 使用1x1卷积调整通道数回原来的维度
        out = self.conv(fused)
        
        return out
    
class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out

class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x