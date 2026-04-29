import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import math
from torch.fft import fftn, ifftn
from torchvision.ops import DeformConv2d
from .vmamba import CrossMambaFusion_SS2D_SSM
from einops import rearrange
# --- 导入新的 Backbone ---
from models.pvtv2 import pvt_v2_b5, pvt_v2_b2
import torch.nn.functional as F
import os
import onnx

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class DFMG(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(DFMG, self).__init__()

        # --- 修改 1: 将 backbone 替换为 pvt_v2_b5 ---
        self.backbone_rgb = pvt_v2_b5()
        self.backbone_depth = pvt_v2_b5()

        # --- PVT-v2-b5 的通道维度 (从浅到深) ---
        pvt_channels = [64, 128, 320, 512]

        # --- 修改 2: 调整 MGAF 和 FAME 的初始化维度以匹配 pvt_v2_b5 ---
        # MGAF(channels) -> channels 是 backbone 单个分支的输出通道数
        # 顺序与 pvt_channels 对应 (从浅到深)
        self.mgaf4 = MGAF(pvt_channels[0])  # 64
        self.mgaf3 = MGAF(pvt_channels[1])  # 128
        self.mgaf2 = MGAF(pvt_channels[2])  # 320
        self.mgaf1 = MGAF(pvt_channels[3])  # 512

        # FAME(channels) -> channels 是 MGAF 的输出通道数 (MGAF_channels * 2)
        mgaf_out_channels = [c * 2 for c in pvt_channels]  # -> [128, 256, 640, 1024]
        self.fame4 = FAME(mgaf_out_channels[0])  # 128
        self.fame3 = FAME(mgaf_out_channels[1])  # 256
        self.fame2 = FAME(mgaf_out_channels[2])  # 640
        self.fame1 = FAME(mgaf_out_channels[3])  # 1024

        # --- 修改 3: 使用原始的、未经修改的 IterativeDecoder ---
        # 因为 FAME 的输出维度 [1024, 640, 256, 128] 正好是它期望的输入
        self.decoder = IterativeDecoder()
        self.decoder_f = IterativeDecoder()

        # 后续层保持不变
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv256_32 = conv3x3_bn_relu(256, 32)
        self.conv64_32 = conv3x3_bn_relu(64, 32)
        self.conv32_1 = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def _load_pre(self, path):
        print(f"--- Loading pre-trained PVT-v2-b5 weights from: {path} ---")
        if not os.path.exists(path):
            print(f"Warning: Pre-trained model path does not exist!")
            return
        try:
            save_model = torch.load(path, map_location='cpu')

            model_dict_rgb = self.backbone_rgb.state_dict()
            state_dict_rgb = {k: v for k, v in save_model.items() if k in model_dict_rgb and 'head' not in k}
            model_dict_rgb.update(state_dict_rgb)
            self.backbone_rgb.load_state_dict(model_dict_rgb)

            model_dict_depth = self.backbone_depth.state_dict()
            state_dict_depth = {k: v for k, v in save_model.items() if k in model_dict_depth and 'head' not in k}
            model_dict_depth.update(state_dict_depth)
            self.backbone_depth.load_state_dict(model_dict_depth)
            print("Successfully loaded weights for both RGB and Depth PVT-v2-b5 backbones.")
        except Exception as e:
            print(f"Error loading pre-trained weights from {path}: {e}")

    def forward(self, x, d):
        # Backbone a
        pvt_rgb = self.backbone_rgb(x)
        pvt_depth = self.backbone_depth(d)

        # 遵循原始命名习惯 r4(浅) -> r1(深)
        r4 = pvt_rgb[0]  # [batch_size, 64, H/4, W/4]
        r3 = pvt_rgb[1]  # [batch_size, 128, H/8, W/8]
        r2 = pvt_rgb[2]  # [batch_size, 320, H/16, W/16]
        r1 = pvt_rgb[3]  # [batch_size, 512, H/32, W/32]

        d4 = pvt_depth[0]
        d3 = pvt_depth[1]
        d2 = pvt_depth[2]
        d1 = pvt_depth[3]

        # MGAF 融合 (从深到浅)
        fuse1 = self.mgaf1(r1, d1)  # 512 -> 1024
        fuse2 = self.mgaf2(r2, d2)  # 320 -> 640
        fuse3 = self.mgaf3(r3, d3)  # 128 -> 256
        fuse4 = self.mgaf4(r4, d4)  # 64  -> 128

        # FAME 频域增强
        ff1 = self.fame1(fuse1)
        ff2 = self.fame2(fuse2)
        ff3 = self.fame3(fuse3)
        ff4 = self.fame4(fuse4)

        # 双路解码器 (输入维度已完美匹配)
        end_fuse, _ = self.decoder(ff1, ff2, ff3, ff4)
        end_ff, _ = self.decoder_f(ff1, ff2, ff3, ff4)

        # 后续处理
        end_cam = self.conv256_32(end_fuse)
        edge = self.conv256_32(end_ff)
        fus = self.conv64_32(torch.cat([end_cam, edge], dim=1))
        out = self.up4(fus)
        edge_out = self.up_edge(edge)
        cam_out = self.conv32_1(out)

        return cam_out, self.sigmoid(cam_out), edge_out


class MGAF(nn.Module):
    # --- MGAF 完整代码 (保持不变) ---
    def __init__(self, channels, d_state=4, dt_rank="auto", ssm_ratio=2.0):
        super().__init__();
        self.fusion = CrossMambaFusion_SS2D_SSM(d_model=channels, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
                                                d_conv=3, dropout=0.1, softmax_version=True);
        self.channel_conv = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=1), nn.BatchNorm2d(channels),
                                          nn.GELU());
        self.channel_attn = ChannelAttention(in_planes=channels, ratio=16);
        self.spatial_att = SpatialAttention(kernel_size=7);
        self.final_conv = conv3x3_bn_relu(channels * 2, channels * 2)

    def forward(self, rgb, depth):
        fused = self.fusion(rgb.permute(0, 2, 3, 1), depth.permute(0, 2, 3, 1));
        mamba_out = fused.permute(0, 3, 1, 2);
        concat_features = self.channel_conv(torch.cat([rgb, depth], dim=1));
        channel_weights = self.channel_attn(mamba_out + concat_features);
        channel_out = channel_weights * mamba_out + (1 - channel_weights) * concat_features;
        spatial_weights = self.spatial_att(mamba_out * concat_features);
        spatial_out = spatial_weights * mamba_out + (1 - spatial_weights) * concat_features;
        combined1 = channel_out + spatial_out;
        combined2 = channel_out * spatial_out;
        final_combined = torch.cat([combined1, combined2], dim=1);
        return self.final_conv(final_combined)


class ChannelAttention(nn.Module):
    # --- ChannelAttention 完整代码 (保持不变) ---
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__();
        self.max_pool = nn.AdaptiveMaxPool2d(1);
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False);
        self.relu1 = nn.ReLU();
        self.fc2 = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False);
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))));
        out = max_out;
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # --- SpatialAttention 完整代码 (保持不变) ---
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__();
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7';
        padding = 3 if kernel_size == 7 else 1;
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False);
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True);
        x = max_out;
        x = self.conv1(x);
        return self.sigmoid(x)


class FreqTransformer(nn.Module):
    # --- FreqTransformer 完整代码 (保持不变) ---
    def __init__(self, channels, num_heads=4):
        super().__init__();
        self.num_heads = num_heads;
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False);
        self.proj = nn.Conv2d(channels, channels, 1, bias=False);
        self.scale = (channels // self.num_heads) ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape;
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w).permute(1, 0, 2, 3, 4);
        q, k, v = qkv[0], qkv[1], qkv[2];
        attn = (q @ k.transpose(-2, -1)) * self.scale;
        attn = attn.softmax(dim=-1);
        x = (attn @ v).transpose(2, 3).reshape(b, c, h, w);
        return self.proj(x)


class FAME(nn.Module):
    # --- FAME 完整代码 (保持不变) ---
    def __init__(self, channels, num_heads=2):
        super().__init__();
        self.ratio_param_low = nn.Parameter(torch.tensor(0.2));
        self.ratio_param_high = nn.Parameter(torch.tensor(0.6));
        self.phase_weight_high = nn.Parameter(torch.tensor(1.0));
        self.high_scale = nn.Parameter(torch.ones(1, channels, 1, 1));
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False);
        self.edge_refine = nn.Sequential(nn.Conv2d(channels, channels // 4, 1), nn.ReLU(),
                                         nn.Conv2d(channels // 4, channels, 1), nn.Tanh());
        self.freq_transformer = FreqTransformer(channels, num_heads);
        self.small_enhance = nn.Conv2d(channels, channels, 1, bias=False);
        self.fusion_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        enhanced_feat = self.frequency_enhance(x);
        return x + enhanced_feat

    def frequency_enhance(self, x):
        fft = torch.fft.rfftn(x, dim=(-2, -1));
        fft_amplitude = torch.abs(fft);
        fft_phase = torch.angle(fft);
        low_mask, high_mask = self.create_masks(x.shape[-2:], x.device);
        high_fft = fft_amplitude * high_mask * self.high_scale * (1 + self.phase_weight_high * fft_phase);
        high_feat = torch.fft.irfftn(high_fft, s=x.shape[-2:], dim=(-2, -1));
        high_feat = self.freq_transformer(high_feat);
        edge_feat = self.edge_conv(high_feat);
        edge_feat = edge_feat * self.edge_refine(edge_feat);
        small_edge = self.small_enhance(edge_feat * (torch.mean(edge_feat, dim=[2, 3], keepdim=True) > 0.5).float());
        final_feat = edge_feat + small_edge;
        return final_feat

    def create_masks(self, size, device):
        h, w = size;
        freq_w = w // 2 + 1;
        Y, X = torch.meshgrid(torch.arange(h, device=device), torch.arange(freq_w, device=device), indexing='ij');
        center_y = h // 2;
        dist = ((X) ** 2 + (Y - center_y) ** 2).sqrt();
        max_dist = dist.max();
        cutoff_low = max_dist * (0.1 + 0.3 * torch.sigmoid(self.ratio_param_low));
        cutoff_high = max_dist * (0.1 + 0.3 * torch.sigmoid(self.ratio_param_high));
        low_mask = (dist <= cutoff_low).float().unsqueeze(0).unsqueeze(0);
        high_mask = (dist > cutoff_high).float().unsqueeze(0).unsqueeze(0);
        total = low_mask + high_mask + 1e-6;
        return low_mask / total, high_mask / total


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UCBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(UCBlock, self).__init__()
        self.conv = nn.Sequential(
            # 1x1 降维瓶颈
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 3x3 特征融合
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, left_feat, top_feat):
        if top_feat.shape[2:] != left_feat.shape[2:]:
            top_feat = F.interpolate(top_feat, size=left_feat.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([left_feat, top_feat], dim=1)
        return self.conv(out)


class IterativeDecoder(nn.Module):
    def __init__(self):
        super(IterativeDecoder, self).__init__()

        c_i1, c_i2, c_i3, c_i4 = 128, 256, 640, 1024
        out_dim = 256

        # 左侧 3个 UC 模块
        self.uc1 = UCBlock(c_i1 + c_i2, out_dim)  # I1 + I2
        self.uc2 = UCBlock(c_i1 + c_i3, out_dim)  # I1 + I3
        self.uc3 = UCBlock(c_i1 + c_i4, out_dim)  # I1 + I4

        # 通道注意力模块
        self.channel_att = SELayer(out_dim)

        # 右侧 3个 UC 模块
        self.uc4 = UCBlock(out_dim + c_i1, out_dim)  # Left: uc1 输出, Top: I1
        self.uc5 = UCBlock(out_dim + out_dim, out_dim)  # Left: uc2 输出, Top: uc4 输出
        self.uc6 = UCBlock(out_dim + out_dim, out_dim)  # Left: mult_out, Top: uc5 输出

    def forward(self, f1, f2, f3, f4):
        i1 = f4  # 128
        i2 = f3  # 256
        i3 = f2  # 640
        i4 = f1  # 1024

        # === 左侧并行融合 ===
        uc1_out = self.uc1(i1, i2)
        uc2_out = self.uc2(i1, i3)
        uc3_out = self.uc3(i1, i4)

        # === 通道注意力 ===
        mult_out = self.channel_att(uc3_out)

        # === 右侧级联融合 ===
        uc4_out = self.uc4(uc1_out, i1)
        uc5_out = self.uc5(uc2_out, uc4_out)
        uc6_out = self.uc6(mult_out, uc5_out)

        return uc6_out, uc5_out