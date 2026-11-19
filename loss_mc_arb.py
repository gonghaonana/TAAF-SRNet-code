import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Slope_net  # 确保 utils.py 中包含此类

# -------------------------------
# 不确定性加权模块（四个任务：x2, x4, x8, triplet）
# -------------------------------
class UncertaintyWeight(nn.Module):
    def __init__(self, num_tasks=4):
        super(UncertaintyWeight, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # λ1~λ4

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total

# -------------------------------
# TV Loss（总变分损失）
# -------------------------------
def tv_loss(x):
    b, c, h, w = x.shape
    tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean()
    tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean()
    return (tv_h + tv_w) / 2

# -------------------------------
# 三尺度特征一致性损失
# -------------------------------
def triplet_consistency_loss(f2, f4, f8, fused=None):
    base_loss = F.l1_loss(f2, F.interpolate(f4, size=f2.shape[-2:], mode='bilinear', align_corners=False)) + \
                F.l1_loss(f4, F.interpolate(f8, size=f4.shape[-2:], mode='bilinear', align_corners=False))

    fused_loss = torch.tensor(0.0, device=f2.device)
    if fused is not None:
        if fused.dim() == 2:
            fused = fused.view(fused.size(0), fused.size(1), 1, 1).expand(-1, -1, f2.shape[2], f2.shape[3])
        elif fused.shape[-2:] != f2.shape[-2:]:
            fused = F.interpolate(fused, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        avg_feat = (
            f2 +
            F.interpolate(f4, size=f2.shape[-2:], mode='bilinear', align_corners=False) +
            F.interpolate(f8, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        ) / 3

        fused_loss = F.l1_loss(fused, avg_feat)

    return base_loss + fused_loss, base_loss, fused_loss

# -------------------------------
# 多尺度总损失函数（含不确定性 + TV + Triplet + Slope）
# -------------------------------
class MultiScaleLoss(nn.Module):
    def __init__(self, loss_type='l1', use_uncertainty=True, use_triplet=True, use_fused=True, use_tv=True, use_slope=True):
        super(MultiScaleLoss, self).__init__()
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss
        self.use_triplet = use_triplet
        self.use_fused = use_fused
        self.use_tv = use_tv
        self.use_slope = use_slope
        self.uncertainty = UncertaintyWeight(num_tasks=4) if use_uncertainty else None
        self.slope_net = Slope_net() if use_slope else None
        self.slope_loss_fn = nn.MSELoss()

    def forward(self, preds, hr, features):
        x2_out, x4_out, x8_out = preds["x2"], preds["x4"], preds["x8"]
        f2, f4, f8 = features["feat"]
        fused = features.get("fused", None) if self.use_fused else None

        # 上量样 GT 到各尺寸
        hr2 = F.interpolate(hr, size=x2_out.shape[-2:], mode='bicubic', align_corners=False)
        hr4 = F.interpolate(hr, size=x4_out.shape[-2:], mode='bicubic', align_corners=False)
        hr8 = F.interpolate(hr, size=x8_out.shape[-2:], mode='bicubic', align_corners=False)

        # 重建损失
        loss2 = self.loss_fn(x2_out[:, :1, ...], hr2)
        loss4 = self.loss_fn(x4_out[:, :1, ...], hr4)
        loss8 = self.loss_fn(x8_out[:, :1, ...], hr8)

        # TV Loss
        tv = tv_loss(x2_out) + tv_loss(x4_out) + tv_loss(x8_out) if self.use_tv else torch.tensor(0.0, device=hr.device)

        # 一致性损失
        if self.use_triplet:
            total_triplet, base_triplet, fused_loss = triplet_consistency_loss(f2, f4, f8, fused)
        else:
            total_triplet = base_triplet = fused_loss = torch.tensor(0.0, device=hr.device)

        # 坡度损失
        if self.use_slope:
            slope_gt = self.slope_net(hr)
            slope_pred = self.slope_net(x8_out[:, :1])
            slope_gt = F.interpolate(slope_gt, size=slope_pred.shape[-2:], mode='bilinear', align_corners=False)
            slope_loss = self.slope_loss_fn(slope_pred, slope_gt)
        else:
            slope_loss = torch.tensor(0.0, device=hr.device)

        # 总损失
        if self.uncertainty:
            total_loss = self.uncertainty([loss2, loss4, loss8, total_triplet]) + 1e-6 * tv + 0.05 * slope_loss
        else:
            total_loss = (loss2 + loss4 + loss8 + total_triplet) / 4 + 1e-6 * tv + 0.05 * slope_loss

        return total_loss, [loss2, loss4, loss8], base_triplet, fused_loss, tv, slope_loss

