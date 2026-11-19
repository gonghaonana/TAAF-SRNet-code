import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for lr2, lr4, lr8, hr, _ in tqdm(dataloader, desc="Training", leave=False):
        lr2, lr4, lr8, hr = lr2.to(device), lr4.to(device), lr8.to(device), hr.to(device)

        optimizer.zero_grad()

        outputs = model(lr2, lr4, lr8)
        x2_out, x4_out, x8_out, *_ = outputs

        # 上采样到 HR 尺寸
        x2_up = nn.functional.interpolate(x2_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        x4_up = nn.functional.interpolate(x4_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        x8_up = nn.functional.interpolate(x8_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # 三尺度平均损失
        loss = (criterion(x2_up, hr) + criterion(x4_up, hr) + criterion(x8_up, hr)) / 3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def _to_np(x):
    return x.squeeze().detach().cpu().numpy().astype(np.float32)


def _unpack_batch(batch):
    """
    兼容 Dataset 返回：
    - (lr2, lr4, lr8, hr, range_info)
    - (lr2, lr4, lr8, hr, path)
    - (lr2, lr4, lr8, hr, path, range_info)
    - (lr2, lr4, lr8, hr, name, path, range_info, ...)
    - dict: {'lr2','lr4','lr8','hr', 'range_info'(optional), ...}

    返回：
    lr2, lr4, lr8, hr, range_info_or_None, extra
    """
    if isinstance(batch, dict):
        lr2 = batch["lr2"]
        lr4 = batch["lr4"]
        lr8 = batch["lr8"]
        hr = batch["hr"]
        range_info = batch.get("range_info", None)
        extra = {k: v for k, v in batch.items() if k not in ["lr2", "lr4", "lr8", "hr", "range_info"]}
        return lr2, lr4, lr8, hr, range_info, extra

    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if len(batch) < 4:
        raise ValueError(f"Batch length={len(batch)} < 4, cannot unpack lr2/lr4/lr8/hr.")

    lr2, lr4, lr8, hr = batch[0], batch[1], batch[2], batch[3]
    rest = list(batch[4:])

    range_info = None
    range_idx = None

    def _is_number(x):
        if isinstance(x, (int, float)):
            return True
        if torch.is_tensor(x) and x.numel() == 1:
            return True
        return False

    for i, item in enumerate(rest):
        # 1) dict 且含 min/max
        if isinstance(item, dict) and (("min" in item and "max" in item) or ("min_val" in item and "max_val" in item)):
            range_info = item
            range_idx = i
            break

        # 2) Tensor: [2] 或 [B,2]
        if torch.is_tensor(item):
            if item.numel() == 2:
                range_info = item
                range_idx = i
                break
            if item.ndim == 2 and item.shape[-1] == 2:
                range_info = item
                range_idx = i
                break

        # 3) tuple/list 长度2 且是数字（min,max）
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = item[0], item[1]
            if _is_number(a) and _is_number(b):
                range_info = item
                range_idx = i
                break

        # 4) ((min,max),) / [(min,max)]
        if isinstance(item, (list, tuple)) and len(item) == 1:
            inner = item[0]
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                a, b = inner[0], inner[1]
                if _is_number(a) and _is_number(b):
                    range_info = inner
                    range_idx = i
                    break

    if range_idx is not None:
        extra = rest[:range_idx] + rest[range_idx + 1:]
    else:
        extra = rest

    return lr2, lr4, lr8, hr, range_info, extra


def _rangeinfo_to_tensor(range_info, batch_size, device):
    """
    统一输出 Tensor [B,2] -> (min,max)
    兼容：
    - (min, max)
    - ((min,max),) / [(min,max)]
    - [(min,max), (min,max), ...]
    - Tensor [2] / [B,2]
    - dict {'min':..., 'max':...} / {'min_val':..., 'max_val':...}
    """
    if range_info is None:
        raise ValueError("range_info is None, but denormalization is requested.")

    # 剥掉多余单层包装
    while isinstance(range_info, (list, tuple)) and len(range_info) == 1:
        range_info = range_info[0]

    # dict
    if isinstance(range_info, dict):
        min_v = range_info.get("min", range_info.get("min_val", None))
        max_v = range_info.get("max", range_info.get("max_val", None))
        if min_v is None or max_v is None:
            raise ValueError(f"range_info dict must contain min/max keys, got keys={list(range_info.keys())}")

        min_v = torch.as_tensor(min_v, dtype=torch.float32, device=device).view(-1)
        max_v = torch.as_tensor(max_v, dtype=torch.float32, device=device).view(-1)

        if min_v.numel() == 1 and batch_size > 1:
            min_v = min_v.repeat(batch_size)
        if max_v.numel() == 1 and batch_size > 1:
            max_v = max_v.repeat(batch_size)

        if min_v.numel() != batch_size or max_v.numel() != batch_size:
            raise ValueError(f"range_info dict size mismatch: min={min_v.numel()}, max={max_v.numel()}, B={batch_size}")

        return torch.stack([min_v, max_v], dim=1)

    # Tensor
    if torch.is_tensor(range_info):
        t = range_info.to(device).float()
        if t.ndim == 2 and t.shape[1] == 2:
            return t
        if t.numel() == 2:
            return t.view(1, 2).repeat(batch_size, 1)
        raise ValueError(f"Unsupported range_info tensor shape: {tuple(t.shape)}")

    # list / tuple
    if isinstance(range_info, (list, tuple)):
        # (min,max)
        if len(range_info) == 2:
            min_v, max_v = range_info[0], range_info[1]
            min_v = torch.as_tensor(min_v, dtype=torch.float32, device=device).view(-1)
            max_v = torch.as_tensor(max_v, dtype=torch.float32, device=device).view(-1)

            if min_v.numel() == 1 and batch_size > 1:
                min_v = min_v.repeat(batch_size)
            if max_v.numel() == 1 and batch_size > 1:
                max_v = max_v.repeat(batch_size)

            if min_v.numel() != batch_size or max_v.numel() != batch_size:
                raise ValueError(f"(min,max) size mismatch: min={min_v.numel()}, max={max_v.numel()}, B={batch_size}")

            return torch.stack([min_v, max_v], dim=1)

        # [(min,max), ...] len == B
        if len(range_info) == batch_size:
            mins, maxs = [], []
            for item in range_info:
                while isinstance(item, (list, tuple)) and len(item) == 1:
                    item = item[0]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    mn, mx = item[0], item[1]
                    mn = float(mn) if not torch.is_tensor(mn) else float(mn.detach().cpu().item())
                    mx = float(mx) if not torch.is_tensor(mx) else float(mx.detach().cpu().item())
                    mins.append(mn)
                    maxs.append(mx)
                else:
                    raise ValueError(f"Unsupported range_info item: {item} (type={type(item)})")

            min_v = torch.tensor(mins, dtype=torch.float32, device=device)
            max_v = torch.tensor(maxs, dtype=torch.float32, device=device)
            return torch.stack([min_v, max_v], dim=1)

        raise ValueError(f"Unsupported range_info tuple/list length={len(range_info)} (B={batch_size}).")

    raise TypeError(f"Unsupported range_info type: {type(range_info)}")


def _parse_model_outputs(model_out):
    """
    兼容模型输出：
    - dict: {'x2','x4','x8'}
    - tuple/list: (x2, x4, x8, ...)
    - tuple/list: ({'x2','x4','x8'}, ...)
    """
    if isinstance(model_out, dict):
        return model_out["x2"], model_out["x4"], model_out["x8"]

    if isinstance(model_out, (list, tuple)):
        first = model_out[0]
        if isinstance(first, dict):
            return first["x2"], first["x4"], first["x8"]
        if len(model_out) >= 3:
            return model_out[0], model_out[1], model_out[2]

    raise TypeError(f"Unsupported model output type: {type(model_out)}")


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    评估三尺度输出，并在拿到 range_info 时反归一化到真实高程范围后再计算指标。
    返回：
    psnr_list, ssim_list, mae_list, rmse_list
    """
    model.eval()

    psnr_sum = [0.0, 0.0, 0.0]
    ssim_sum = [0.0, 0.0, 0.0]
    mae_sum = [0.0, 0.0, 0.0]
    rmse_sum = [0.0, 0.0, 0.0]
    count = 0

    for batch in loader:
        lr2, lr4, lr8, hr, range_info, _extra = _unpack_batch(batch)

        lr2 = lr2.to(device, non_blocking=True)
        lr4 = lr4.to(device, non_blocking=True)
        lr8 = lr8.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        B = hr.shape[0]

        if range_info is not None:
            range_info = _rangeinfo_to_tensor(range_info, batch_size=B, device=device)

        model_out = model(lr2, lr4, lr8)
        sr_x2, sr_x4, sr_x8 = _parse_model_outputs(model_out)

        # 强制对齐到 HR 尺寸
        sr_x2 = F.interpolate(sr_x2, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        sr_x4 = F.interpolate(sr_x4, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        sr_x8 = F.interpolate(sr_x8, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # 反归一化
        if range_info is not None:
            min_val = range_info[:, 0].view(-1, 1, 1, 1)
            max_val = range_info[:, 1].view(-1, 1, 1, 1)
            value_range = (max_val - min_val).clamp(min=1e-12)

            sr_x2 = sr_x2 * value_range + min_val
            sr_x4 = sr_x4 * value_range + min_val
            sr_x8 = sr_x8 * value_range + min_val
            hr_dn = hr * value_range + min_val
        else:
            hr_dn = hr

        # 逐样本算指标
        for b in range(B):
            hr_np = hr_dn[b, 0].detach().cpu().numpy().astype(np.float32)
            sr2_np = sr_x2[b, 0].detach().cpu().numpy().astype(np.float32)
            sr4_np = sr_x4[b, 0].detach().cpu().numpy().astype(np.float32)
            sr8_np = sr_x8[b, 0].detach().cpu().numpy().astype(np.float32)

            dr = float(hr_np.max() - hr_np.min())
            if dr < 1e-12:
                dr = 1.0

            for i, pred_np in enumerate([sr2_np, sr4_np, sr8_np]):
                psnr_sum[i] += psnr(hr_np, pred_np, data_range=dr)
                ssim_sum[i] += ssim(hr_np, pred_np, data_range=dr)
                mae_sum[i] += float(np.mean(np.abs(pred_np - hr_np)))
                rmse_sum[i] += float(np.sqrt(np.mean((pred_np - hr_np) ** 2)))

            count += 1

    if count == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    return (
        [v / count for v in psnr_sum],
        [v / count for v in ssim_sum],
        [v / count for v in mae_sum],
        [v / count for v in rmse_sum],
    )


def write_log(filepath, message):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def calc_psnr(pred, gt, data_range=1.0):
    return psnr(gt, pred, data_range=data_range)


def calc_ssim(pred, gt):
    dr = gt.max() - gt.min()
    if dr <= 1e-12:
        dr = 1.0
    return ssim(gt, pred, data_range=dr)


def calc_mae(pred, gt):
    return mean_absolute_error(gt.flatten(), pred.flatten())


def calc_rmse(pred, gt):
    return math.sqrt(mean_squared_error(gt.flatten(), pred.flatten()))


class Slope_net(nn.Module):
    def __init__(self):
        super(Slope_net, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 注册为 buffer，支持自动跟随模型转移到 CUDA 或 CPU
        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

    def forward(self, x):
        weight_x = self.weight_x.to(x.device)
        weight_y = self.weight_y.to(x.device)

        grad_x = F.conv2d(x, weight_x, padding=1)
        grad_y = F.conv2d(x, weight_y, padding=1)
        slope = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)
        return slope