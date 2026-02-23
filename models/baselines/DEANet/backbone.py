import torch.nn as nn
import torch.nn.functional as F

# 尝试从modules导入必要的组件
try:
    from .modules import DEABlock, DEBlock, CGAFusion
except ImportError:
    # 当直接运行文件时，使用绝对导入
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from models.model.modules import DEABlock, DEBlock, CGAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Backbone(nn.Module):
    def __init__(self, base_dim=32):
        super(Backbone, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlock(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlock(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out

if __name__ == "__main__":
    import torch
    # 创建Backbone实例
    backbone = Backbone()
    
    # 计算和打印参数量
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    input_tensor = torch.randn(1, 3, 256, 256)
    output = backbone(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算和打印MACs
    try:
        from thop import profile
        macs, params = profile(backbone, inputs=(input_tensor,))
        print(f"MACs: {macs / 1e9:.2f} G")
    except ImportError:
        print("Warning: thop library not found. MACs calculation skipped.")
    
    print("Backbone test completed successfully!")