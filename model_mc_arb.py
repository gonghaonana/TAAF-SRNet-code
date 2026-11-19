import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# ---------- 梯度强度（Sobel）先验 ----------
class GradMag(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.to1 = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        kx = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=torch.float32).view(1,1,3,3)/8.0
        ky = torch.tensor([[1, 2,  1],[0, 0,  0],[-1,-2, -1]], dtype=torch.float32).view(1,1,3,3)/8.0
        with torch.no_grad():
            self.sobel_x.weight.copy_(kx)
            self.sobel_y.weight.copy_(ky)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):  # x: [B,C,H,W]
        x1 = self.to1(x)
        gx = self.sobel_x(x1)
        gy = self.sobel_y(x1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-12)
        g_min = g.amin(dim=(2,3), keepdim=True)
        g_max = g.amax(dim=(2,3), keepdim=True)
        g = (g - g_min) / (g_max - g_min + 1e-12)
        return g  # [B,1,H,W]


# ---------- 自适应融合：输出 out + 可选的权重图 ----------
class AdaptiveMerger(nn.Module):
    def __init__(self, in_ch, mid_ch=64, use_prior=True, temperature=1.5):
        super().__init__()
        self.use_prior = use_prior
        self.temperature = temperature  # >1 更“硬”，<1 更“软”

        self.reduce = nn.Conv2d(in_ch * 3, mid_ch, kernel_size=1, bias=True)
        in_gate = mid_ch + (1 if use_prior else 0)
        self.gate = nn.Sequential(
            nn.Conv2d(in_gate, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.prior = GradMag(in_ch) if use_prior else None

    def forward(self, x, x_ercsr, x_vit):
        feat = torch.cat([x, x_ercsr, x_vit], dim=1)      # [B,3C,H,W]
        feat = self.reduce(feat)                           # [B,mid,H,W]
        if self.use_prior:
            p = self.prior(x).detach()                     # [B,1,H,W] 引导，不回传梯度
            gate_in = torch.cat([feat, p], dim=1)
        else:
            p = None
            gate_in = feat

        logits = self.gate(gate_in)                        # [B,1,H,W]
        w_ercsr = self.sigmoid(logits * self.temperature)  # [B,1,H,W]
        w_ercsr = torch.clamp(w_ercsr, 1e-4, 1-1e-4)
        w_vit = 1.0 - w_ercsr

        out = x + w_ercsr * x_ercsr + w_vit * x_vit
        return out, (w_ercsr, w_vit, p)
# -------------------------------
# 2D Residual Block
# -------------------------------
class TwoUint(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

# -------------------------------
# Helper functions for 3D-2D conversion
# -------------------------------
def _to_4d_tensor(x, depth_stride=None):
    x = x.transpose(0, 2)
    if depth_stride:
        x = x[::depth_stride]
    depth = x.size(0)
    x = x.permute(2, 0, 1, 3, 4)
    x = torch.split(x, 1, dim=0)
    x = torch.cat(x, dim=1)
    x = x.squeeze(0)
    return x, depth

def _to_5d_tensor(x, depth):
    x = torch.split(x, depth)
    x = torch.stack(x, dim=0)
    x = x.transpose(1, 2)
    return x

# -------------------------------
# ERCSR Block
# -------------------------------
class ERCSRBlock(nn.Module):
    def __init__(self, n_feats, n_twounit=2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.conv3 = nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 3), padding=(1, 0, 1))
        self.twoD_body = nn.Sequential(*[TwoUint(n_feats) for _ in range(n_twounit)])

    def forward(self, x, n_scale=2):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) + self.conv3(out)
        shortcut = out
        out, depth = _to_4d_tensor(out)
        out = self.twoD_body(out)
        out = _to_5d_tensor(out, depth)
        return out + shortcut + residual

# -------------------------------
# Hybrid FEM = VITFEM + ERCSR
# -------------------------------
class FEM(nn.Module):
    def __init__(self, in_channels=64, embed_dim=64, patch_size=4, num_heads=8, depth=6, mlp_ratio=4.0, n_ercsr_blocks=2, n_twounit=2):
        super().__init__()
        # ----- ERCSR 分支 -----
        self.ercsr_blocks = nn.ModuleList([ERCSRBlock(in_channels, n_twounit) for _ in range(n_ercsr_blocks)])
        self.gamma = nn.Parameter(torch.ones(n_ercsr_blocks))
        self.reduction_conv = nn.Conv3d(in_channels * n_ercsr_blocks, in_channels, kernel_size=1)

        # ----- ViT 分支 -----
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 可插值的 2D 可学习位置编码，避免固定尺寸导致形状不匹配
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 16, 16))  # base grid 16x16，可按需改大
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

        # ----- 自适应融合 -----
        self.merger = AdaptiveMerger(in_ch=in_channels, mid_ch=in_channels, use_prior=True, temperature=1.5)

    def forward(self, x, scale=2, return_weights: bool = False):
        # ---- ERCSR 分支 ----
        x3d = x.unsqueeze(2)                               # [B,C,1,H,W]
        outs = []
        for i, blk in enumerate(self.ercsr_blocks):
            x3d = blk(x3d, scale)
            outs.append(self.gamma[i] * x3d)
        x_ercsr = torch.cat(outs, dim=1)                   # [B, C*n, 1, H, W]
        x_ercsr = self.reduction_conv(x_ercsr).squeeze(2)  # [B,C,H,W]

        # ---- ViT 分支 ----
        B, C, H, W = x.shape
        x_patch = self.patch_embed(x)                      # [B,E,h,w]
        h, w = x_patch.shape[-2:]
        pe2d = F.interpolate(self.pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        x_patch = x_patch + pe2d
        tokens = x_patch.flatten(2).transpose(1, 2)        # [B, N, E]
        x_enc = self.transformer(tokens)                   # [B, N, E]
        x_enc = x_enc.transpose(1, 2).view(B, -1, h, w)
        x_vit = self.proj(x_enc)                           # [B,C,H,W]

        # ---- 自适应融合 ----
        out, (w_ercsr, w_vit, prior) = self.merger(x, x_ercsr, x_vit)

        if return_weights:
            return out, {"w_ercsr": w_ercsr, "w_vit": w_vit, "prior": prior}
        return out

#
# # MCNet
# class FEM(nn.Module):
#     def __init__(self, in_channels=64, num_blocks=4, num_groups=3):
#         super(FEM, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.num_groups = num_groups
#         self.gamma = nn.Parameter(torch.ones(num_groups))
#
#         # 三种卷积感受野：3x3, 5x5, 7x7
#         self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
#         self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
#
#         # 残差分支模块
#         def make_block():
#             layers = []
#             for _ in range(num_blocks):
#                 layers.append(nn.Sequential(
#                     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#                 ))
#             return nn.Sequential(*layers)
#
#         self.group1 = make_block()
#         self.group2 = make_block()
#         self.group3 = make_block()
#
#         # 通道融合卷积
#         self.fusion = nn.Conv2d(in_channels * num_groups, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         residual = x
#
#         # 多尺度感受野输入（非共享）
#         x1 = self.relu(self.conv3(x))
#         x2 = self.relu(self.conv5(x))
#         x3 = self.relu(self.conv7(x))
#
#         # 三个 group 并行残差堆叠
#         x1 = self.group1(x1) + x1
#         x2 = self.group2(x2) + x2
#         x3 = self.group3(x3) + x3
#
#         # 按 gamma 权重拼接融合
#         out = torch.cat([self.gamma[0]*x1, self.gamma[1]*x2, self.gamma[2]*x3], dim=1)
#         out = self.relu(self.fusion(out))
#         out = out + residual  # 残差连接
#         return out

# -------------------------------
# 2D Residual Block
# -------------------------------
# class TwoUint(nn.Module):
#     def __init__(self, n_feats):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.conv2(out)
#         return x + out
#
#
# # Convert 5D to 4D tensor
#
# def _to_4d_tensor(x, depth_stride=None):
#     x = x.transpose(0, 2)
#     if depth_stride:
#         x = x[::depth_stride]
#     depth = x.size(0)
#     x = x.permute(2, 0, 1, 3, 4)
#     x = torch.split(x, 1, dim=0)
#     x = torch.cat(x, dim=1)
#     x = x.squeeze(0)
#     return x, depth
#
#
# def _to_5d_tensor(x, depth):
#     x = torch.split(x, depth)
#     x = torch.stack(x, dim=0)
#     x = x.transpose(1, 2)
#     return x
#
#
# # -------------------------------
# # ERCSR Block
# # -------------------------------
# class ERCSRBlock(nn.Module):
#     def __init__(self, n_feats, n_twounit=2):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1))
#         self.conv2 = nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 1), padding=(1, 1, 0))
#         self.conv3 = nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 3), padding=(1, 0, 1))
#         self.twoD_body = nn.Sequential(*[TwoUint(n_feats) for _ in range(n_twounit)])
#
#     def forward(self, x, n_scale=2):
#         residual = x
#         out = self.relu(self.conv1(x))
#         out = self.conv2(out) + self.conv3(out)
#         shortcut = out
#         out, depth = _to_4d_tensor(out)
#         out = self.twoD_body(out)
#         out = _to_5d_tensor(out, depth)
#         return out + shortcut + residual
#
#
# # -------------------------------
# # FEM with ERCSR Blocks
# # -------------------------------
# class FEM(nn.Module):
#     def __init__(self, in_channels=64, n_blocks=4, n_twounit=2):
#         super().__init__()
#         self.n_blocks = n_blocks
#         self.blocks = nn.ModuleList([
#             ERCSRBlock(in_channels, n_twounit) for _ in range(n_blocks)
#         ])
#         self.gamma = nn.Parameter(torch.ones(n_blocks))
#         self.reduction_conv = nn.Conv3d(in_channels * n_blocks, in_channels, kernel_size=1)
#
#     def forward(self, x, scale=2):
#         x_in = x
#         x = x.unsqueeze(2)
#         outs = []
#         for i, blk in enumerate(self.blocks):
#             x = blk(x, scale)
#             outs.append(self.gamma[i] * x)
#         x = torch.cat(outs, dim=1)
#         x = self.reduction_conv(x)
#         x = x.squeeze(2)
#         return x + x_in

#VITFEM
# class FEM(nn.Module):
#     def __init__(self, in_channels=64, embed_dim=64, patch_size=4, num_heads=8, depth=6, mlp_ratio=4.0):
#         super(FEM, self).__init__()
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#
#         # Patch embedding
#         self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         num_patches = (64 // patch_size) * (64 // patch_size)  # 64 为 DEM patch 默认尺寸，动态可改
#
#         # 可学习的位置编码
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#
#         # Transformer 编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
#             dropout=0.1, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#
#         # 输出还原卷积（类似 unpatchify）
#         self.proj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_patch = self.patch_embed(x)                       # [B, C, H/4, W/4]
#         x_flat = x_patch.flatten(2).transpose(1, 2)         # [B, N, C]
#         x_flat = x_flat + self.pos_embed[:, :x_flat.size(1), :]
#         x_enc = self.transformer(x_flat)                    # [B, N, C]
#         x_enc = x_enc.transpose(1, 2).view(B, -1, H // 4, W // 4)
#         out = self.proj(x_enc)                              # [B, C, H, W]
#         return out + x  # 残差连接


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # 路由网络：根据比例 1/r 输出 expert 权重
        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

        # 多个专家卷积核
        weight_pool = []
        for i in range(num_experts):
            w = nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            weight_pool.append(w)
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, dim=0))  # (N, Cout, Cin, K, K)

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale):
        B, C, H, W = x.shape

        # 输入比例 → routing 权重（softmax）
        scale_tensor = torch.full((1, 1), 1.0 / scale, device=x.device)
        routing_weights = self.routing(scale_tensor).view(self.num_experts, 1, 1)

        # 融合专家核
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(self.channels_out, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.matmul(routing_weights.view(1, -1), self.bias_pool).view(-1)
        else:
            fused_bias = None

        out = F.conv2d(x, fused_weight, bias=fused_bias, stride=self.stride, padding=self.padding)
        return out

# -----------------------------------------
# SAFAM 模块（适用于 DEM）
# -----------------------------------------
class SAFAM(nn.Module):
    def __init__(self, channels=64, num_experts=4):
        super(SAFAM, self).__init__()

        # mask：局部控制是否调整（类似软注意力）
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.adapt = SA_conv(channels, channels, kernel_size=3, padding=1, num_experts=num_experts)

    def forward(self, x, scale):
        mask = self.mask(x)                    # 输出 (B, 1, H, W)
        adapted = self.adapt(x, scale)         # 输出 (B, C, H, W)
        return x + adapted * mask              # 局部位置调整特征
# -------------------------------
# 升级版 MetaUpsample3D（修正 unfold 匹配逻辑）
# -------------------------------

class MetaUpsample3D(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(MetaUpsample3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.in_channels = in_channels

        self.weight_predictor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * kernel_size * kernel_size)
        )

    def forward(self, x, scale):
        B, C, H, W = x.shape
        out_H, out_W = int(H * scale), int(W * scale)

        device = x.device
        i = torch.arange(out_H, device=device).float()
        j = torch.arange(out_W, device=device).float()
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')

        i_f = grid_i / scale
        j_f = grid_j / scale
        i0 = i_f.floor().long().clamp(0, H - 1)
        j0 = j_f.floor().long().clamp(0, W - 1)
        di = (i_f - i0).unsqueeze(0).repeat(B, 1, 1)
        dj = (j_f - j0).unsqueeze(0).repeat(B, 1, 1)
        dr = torch.full_like(di, 1.0 / scale)

        meta_input = torch.stack([di, dj, dr], dim=1).permute(0, 2, 3, 1).reshape(-1, 3)
        kernel_pred = self.weight_predictor(meta_input)
        kernel_pred = kernel_pred.view(B, out_H, out_W, C, self.kernel_size, self.kernel_size).permute(0, 3, 4, 5, 1, 2)

        x_pad = F.pad(x, [self.padding]*4, mode='reflect')
        unfold = F.unfold(x_pad, kernel_size=self.kernel_size).view(B, C, self.kernel_size, self.kernel_size, H, W)

        patch = unfold[:, :, :, :, i0, j0]
        out = (patch * kernel_pred).sum(dim=[2, 3])
        return out


# -------------------------------
# DSPPM 模块
# -------------------------------
class DSPPM(nn.Module):
    def __init__(self, in_dim=64, out_dim=64):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.pool8 = nn.AdaptiveAvgPool2d(8)

        self.conv1 = nn.Conv2d(in_dim, in_dim, 1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 1)
        self.conv3 = nn.Conv2d(in_dim, in_dim, 1)
        self.conv4 = nn.Conv2d(in_dim, in_dim, 1)

        self.out_conv = nn.Linear(in_dim * 5, out_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p1 = F.interpolate(self.conv1(self.pool1(x)), size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv2(self.pool2(x)), size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv3(self.pool4(x)), size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.conv4(self.pool8(x)), size=(H, W), mode='bilinear', align_corners=False)
        concat = torch.cat([x, p1, p2, p3, p4], dim=1)
        gap = F.adaptive_avg_pool2d(concat, 1).view(B, -1)
        return self.out_conv(gap)


# -------------------------------
# 特征融合模块
# -------------------------------
class ScaleFusion(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.fuse = nn.Conv2d(n_feats * 3, n_feats, 1)

    def forward(self, f2, f4, f8):
        f4_up = F.interpolate(f4, size=f2.shape[-2:], mode='bilinear')
        f8_up = F.interpolate(f8, size=f2.shape[-2:], mode='bilinear')
        concat = torch.cat([f2, f4_up, f8_up], dim=1)
        return self.fuse(concat)


# -------------------------------
# 完整的 MCArbNet Full 模型（含独立 SAFAM）
# -------------------------------

class MCArbNet_Full(nn.Module):
    def __init__(self, n_feats=64, n_blocks=4):
        super(MCArbNet_Full, self).__init__()
        self.n_blocks = n_blocks

        # 编码器：每个输入尺度一个
        self.encoders = nn.ModuleDict({
            "x2": nn.Conv2d(1, n_feats, kernel_size=3, padding=1),
            "x4": nn.Conv2d(1, n_feats, kernel_size=3, padding=1),
            "x8": nn.Conv2d(1, n_feats, kernel_size=3, padding=1),
        })

        # FEM：交替结构，每层一个
        self.fem_blocks = nn.ModuleList([FEM(in_channels=n_feats) for _ in range(n_blocks)])

        # SAFAM：每个尺度一个独立分支，每层一个
        self.safams = nn.ModuleDict({
            "x2": nn.ModuleList([SAFAM(n_feats) for _ in range(n_blocks)]),
            "x4": nn.ModuleList([SAFAM(n_feats) for _ in range(n_blocks)]),
            "x8": nn.ModuleList([SAFAM(n_feats) for _ in range(n_blocks)]),
        })

        # Meta 上采样模块（共享）
        self.upsample = MetaUpsample3D()

        # 特征融合向量生成（共享）
        self.dsppm = DSPPM()

        # 每个尺度的最终输出卷积头
        self.output_convs = nn.ModuleDict({
            "x2": nn.Conv2d(n_feats, 1, kernel_size=3, padding=1),
            "x4": nn.Conv2d(n_feats, 1, kernel_size=3, padding=1),
            "x8": nn.Conv2d(n_feats, 1, kernel_size=3, padding=1),
        })

        # 三尺度特征融合
        self.fusion = ScaleFusion(n_feats)

    def forward_branch(self, x, scale, key):
        x = self.encoders[key](x)
        feat = x
        for i in range(self.n_blocks):
            feat = self.fem_blocks[i](feat)
            feat = self.safams[key][i](feat, scale)
            feat = feat / scale  # 每层都做尺度归一化
        out = self.upsample(feat, scale)
        out = self.output_convs[key](out)
        vec = self.dsppm(feat)
        return out, feat, vec

    def forward(self, x2, x4, x8):
        out2, f2, v2 = self.forward_branch(x2, 2, "x2")
        out4, f4, v4 = self.forward_branch(x4, 4, "x4")
        out8, f8, v8 = self.forward_branch(x8, 8, "x8")

        fusion_feat = self.fusion(f2, f4, f8)

        return {
            "x2": out2,
            "x4": out4,
            "x8": out8
        }, {
            "feat": [f2, f4, f8],
            "vec": [v2, v4, v8],
            "fused": fusion_feat
        }
    def forward_single(self, x, scale):
        """
        用于测试/部署阶段的任意尺度推理。
        输入:
            x: [B, 1, H, W]，单尺度低分辨率图像
            scale: float，任意放大倍数（如 2.5）
        输出:
            out: [B, 1, H*scale, W*scale]，重建的高分图像
        """
        # 默认使用 x2 分支的 encoder 和 output conv
        # 因为它们结构一致，只是输入不同
        x = self.encoders["x2"](x)
        feat = x
        for i in range(self.n_blocks):
            feat = self.fem_blocks[i](feat)
            feat = self.safams["x2"][i](feat, scale)
            feat = feat / scale
        out = self.upsample(feat, scale)
        out = self.output_convs["x2"](out)
        return out
