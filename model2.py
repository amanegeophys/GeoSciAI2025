import torch.nn as nn
import torch
import torch.nn.functional as F

class CBAM2(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ch_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * ch_attn

        # ----- Spatial Attention -----
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn

        return x

# === UNet2D Model ===
class UNet2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = self.block(in_channels, base_channels)
        self.enc2 = self.block(base_channels, base_channels)
        self.enc3 = self.block(base_channels, base_channels)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.block(base_channels, base_channels * 2)

        self.dec3 = self.block(base_channels * 2 + base_channels, base_channels * 2)  # 32 + 16
        self.dec2 = self.block(base_channels * 2 + base_channels, base_channels)      # 32 + 16
        self.dec1 = self.block(base_channels * 2, base_channels) 

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def block(self, in_c, out_c):
        num_groups = max(1, out_c // 4)
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(num_groups, out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(num_groups, out_c),
            nn.ReLU(inplace=True),
            CBAM2(out_c)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))