# -*- coding: utf-8 -*-
"""
Standalone training script for *single‑branch* MCArbNet (x2 / x4 / x8).
=======================================================================

Usage (example):
    python MCArbNet_Single.py \
        --train_root E:/jjq/train/land \
        --val_root   E:/jjq/test/land  \
        --branch x2  \
        --epochs 200 \
        --batch_size 16 \
        --lr 1e-4 \
        --update_fem            # ⬅︎ optional, also fine‑tune shared FEM

This script assumes the following project structure already exists::

    ├── model_mc_arb.py          # defines MCArbNet_Full
    ├── dataset.py               # defines DEMDataset (returns HR patches)
    ├── utils.py                 # defines Slope_net, tv_loss (or you can use the local fallback)
    └── this file (MCArbNet_Single.py)

Key Points
----------
* **Only one scale branch is optimised at a time.**  No triplet or fused‑feature losses.
* **Shared modules** (FEM, MetaUpsample3D, DSPPM, ScaleFusion) can be
  *frozen* or *fine‑tuned* via `--update_fem`.
* Validation computes PSNR/SSIM on the chosen scale.
* Automatically saves the best model (highest PSNR) to
  `checkpoints_single_{branch}/best_model.pth`.
"""

import os
import math
import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_mc_arb import MCArbNet_Full
from dataset import DEMDataset
from utils import Slope_net
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

try:
    from utils import tv_loss
except ImportError:
    def tv_loss(x: torch.Tensor) -> torch.Tensor:
        tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return 0.5 * (tv_h + tv_w)

def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

try:
    from skimage.metrics import structural_similarity as sk_ssim
    def calc_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
        sr_np = sr.squeeze(1).cpu().numpy()
        hr_np = hr.squeeze(1).cpu().numpy()
        ssim = 0.0
        for i in range(sr_np.shape[0]):
            ssim += sk_ssim(hr_np[i], sr_np[i], data_range=1.0)
        return ssim / sr_np.shape[0]
except ImportError:
    def calc_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
        return float('nan')

class MCArbNet_Single(nn.Module):
    def __init__(self, base: MCArbNet_Full, branch: str):
        super().__init__()
        assert branch in {"x2", "x4", "x8"}
        self.branch = branch
        self.fem_blocks = base.fem_blocks
        self.upsample = base.upsample
        self.encoder = base.encoders[branch]
        self.safams = base.safams[branch]
        self.output_conv = base.output_convs[branch]

    def forward(self, x: torch.Tensor, scale: int):
        x = self.encoder(x)
        feat = x
        for i in range(len(self.safams)):
            feat = self.fem_blocks[i](feat)
            feat = self.safams[i](feat, scale)
            feat = feat / scale
        out = self.upsample(feat, scale)
        out = self.output_conv(out)
        return out, feat

class SingleScaleLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1', tv_w: float = 1e-6, slope_w: float = 0.05, use_slope: bool = True):
        super().__init__()
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss
        self.tv_w = tv_w
        self.slope_w = slope_w
        self.use_slope = use_slope
        self.slope_net = Slope_net() if use_slope else None
        self.slope_loss_fn = nn.MSELoss()

    def forward(self, pred: torch.Tensor, hr: torch.Tensor):
        rec = self.loss_fn(pred, hr)
        tv = tv_loss(pred)
        if self.use_slope:
            with torch.no_grad():
                slope_gt = self.slope_net(hr)
            slope_pd = self.slope_net(pred.detach())
            slope_loss = self.slope_loss_fn(slope_pd, slope_gt)
        else:
            slope_loss = pred.new_tensor(0.)
        total = rec + self.tv_w * tv + self.slope_w * slope_loss
        return total, rec, tv, slope_loss

def get_branch_params(model: MCArbNet_Single, update_fem: bool) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    params += list(model.encoder.parameters())
    params += list(model.safams.parameters())
    params += list(model.output_conv.parameters())
    if update_fem:
        params += list(model.fem_blocks.parameters())
        params += list(model.upsample.parameters())
    return params

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_root', type=str, default='E:/jjq/data/ocean/train')
    parser.add_argument('--val_root',   type=str, default='E:/jjq/data/ocean/test')
    parser.add_argument('--branch',     type=str, choices=['x2', 'x4', 'x8'], required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs',     type=int, default=300)
    parser.add_argument('--lr',         type=float, default=1e-5)
    parser.add_argument('--update_fem', action='store_true')
    parser.add_argument('--save_dir',   type=str, default='checkpoints_single')
    parser.add_argument('--resume',     type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(DEMDataset(args.train_root), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(DEMDataset(args.val_root), batch_size=1, shuffle=False, num_workers=2)

    scale = {'x2': 2, 'x4': 4, 'x8': 8}[args.branch]

    base_model = MCArbNet_Full().to(device)
    model = MCArbNet_Single(base_model, args.branch).to(device)

    best_psnr = 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        base_model.load_state_dict(ckpt['model'])
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"🔄  Resumed from {args.resume} (best_psnr={best_psnr:.2f} dB)")

    criterion = SingleScaleLoss().to(device)
    optimizer = torch.optim.Adam(get_branch_params(model, args.update_fem), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    ckpt_dir = Path(args.save_dir) / f'{args.branch}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / 'train_log.txt'
    log_file = open(log_path, 'a')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', ncols=120)
        for hr in pbar:
            if isinstance(hr, (list, tuple)):
                hr = hr[0]
            hr = hr.to(device)

            lr = F.interpolate(hr, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            pred, _ = model(lr, scale)
            hr_crop = F.interpolate(hr, size=pred.shape[-2:], mode='bicubic', align_corners=False)
            loss, rec, tv, slope = criterion(pred, hr_crop)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * hr.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss /= len(train_loader.dataset)

        # ---------------- Validation ----------------
        model.eval()
        psnr_sum, ssim_sum, mae_sum, rmse_sum = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for hr in val_loader:
                if isinstance(hr, (list, tuple)):
                    hr = hr[0]
                hr = hr.to(device)

                lr = F.interpolate(hr, scale_factor=1 / scale, mode='bicubic', align_corners=False)
                pred, _ = model(lr, scale)
                hr_crop = F.interpolate(hr, size=pred.shape[-2:], mode='bicubic', align_corners=False)

                pred = pred.clamp(0, 1)
                hr_crop = hr_crop.clamp(0, 1)

                psnr_sum += calc_psnr(pred, hr_crop)
                ssim_sum += calc_ssim(pred, hr_crop)
                mae_sum += F.l1_loss(pred, hr_crop).item()
                rmse_sum += F.mse_loss(pred, hr_crop).sqrt().item()

        # 计算平均指标
        psnr = psnr_sum / len(val_loader)
        ssim = ssim_sum / len(val_loader)
        mae = mae_sum / len(val_loader)
        rmse = rmse_sum / len(val_loader)

        scheduler.step(psnr)

        log_line = f"{epoch}\t{epoch_loss:.6f}\t{psnr:.4f}\t{ssim:.4f}\t{mae:.6f}\t{rmse:.6f}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        if psnr > best_psnr:
            best_psnr = psnr
            save_path = ckpt_dir / 'best_model.pth'
            torch.save({
                'model': base_model.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
            }, save_path)
            print(f"✅  Saved better model @ {best_psnr:.2f} dB → {save_path}")

    log_file.close()

if __name__ == '__main__':
    main()
