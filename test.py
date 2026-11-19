import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model_mc_arb import MCArbNet_Full
from dataset import DEMDataset
from utils import evaluate_model
import warnings
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# 配置路径
model_path = 'checkpoints_ocean_Adapt/best_model.pth'
test_root = 'E:/jjq/data/ocean/2/test'


# -----------------------------
# 坡度/坡向工具（只在 test.py 用）
# -----------------------------
def slope_aspect_np(dem: np.ndarray, cellsize: float = 1.0):
    """
    dem: 2D numpy array (H,W), 已反归一化到真实高程
    返回:
      slope_deg: 坡度(度)
      aspect_deg: 坡向(度, 0~360, 0=北,90=东)
    """
    dem = dem.astype(np.float64)
    dzdy, dzdx = np.gradient(dem, cellsize, cellsize)  # y,x

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg = np.mod(aspect_deg, 360.0)

    return slope_deg, aspect_deg


def angle_diff_deg(a: np.ndarray, b: np.ndarray):
    """最小环形角度差 [0,180]"""
    d = np.abs(a - b)
    d = np.minimum(d, 360.0 - d)
    return d


def find_range_info_in_batch(rest, device, B):
    """
    在 batch 的剩余元素中自动寻找 range_info：
    - Tensor(2) 或 Tensor(B,2)
    - tuple/list(min,max) 或 ((min,max),)
    - dict(min/max)
    找不到返回 None
    """
    def is_number(x):
        if isinstance(x, (int, float)):
            return True
        if torch.is_tensor(x) and x.numel() == 1:
            return True
        return False

    # rest 可能是 list/tuple
    for item in rest:
        # 剥掉单元素包装
        tmp = item
        while isinstance(tmp, (list, tuple)) and len(tmp) == 1:
            tmp = tmp[0]

        # dict
        if isinstance(tmp, dict) and (("min" in tmp and "max" in tmp) or ("min_val" in tmp and "max_val" in tmp)):
            mn = tmp.get("min", tmp.get("min_val"))
            mx = tmp.get("max", tmp.get("max_val"))
            mn = torch.as_tensor(mn, dtype=torch.float32, device=device).view(-1)
            mx = torch.as_tensor(mx, dtype=torch.float32, device=device).view(-1)
            if mn.numel() == 1 and B > 1: mn = mn.repeat(B)
            if mx.numel() == 1 and B > 1: mx = mx.repeat(B)
            if mn.numel() == B and mx.numel() == B:
                return torch.stack([mn, mx], dim=1)  # [B,2]

        # Tensor
        if torch.is_tensor(tmp):
            t = tmp.to(device).float()
            if t.ndim == 2 and t.shape[-1] == 2:
                return t
            if t.numel() == 2:
                return t.view(1, 2).repeat(B, 1)

        # (min,max)
        if isinstance(tmp, (list, tuple)) and len(tmp) == 2 and is_number(tmp[0]) and is_number(tmp[1]):
            mn = tmp[0]; mx = tmp[1]
            mn = torch.as_tensor(mn, dtype=torch.float32, device=device).view(-1)
            mx = torch.as_tensor(mx, dtype=torch.float32, device=device).view(-1)
            if mn.numel() == 1 and B > 1: mn = mn.repeat(B)
            if mx.numel() == 1 and B > 1: mx = mx.repeat(B)
            if mn.numel() == B and mx.numel() == B:
                return torch.stack([mn, mx], dim=1)

    return None


def evaluate_slope_aspect(model, dataloader, device, cellsize=1.0):
    """
    额外计算坡度/坡向指标：
    返回 6 个列表（x2,x4,x8）：
      slope_mae, slope_rmse, aspect_mae, aspect_rmse
    """
    model.eval()
    slope_mae_sum  = [0.0, 0.0, 0.0]
    slope_rmse_sum = [0.0, 0.0, 0.0]
    aspect_mae_sum  = [0.0, 0.0, 0.0]
    aspect_rmse_sum = [0.0, 0.0, 0.0]
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            # batch 可能是 tuple/list 或 dict
            if isinstance(batch, dict):
                lr2, lr4, lr8, hr = batch["lr2"], batch["lr4"], batch["lr8"], batch["hr"]
                rest = [batch.get("range_info", None)]
            else:
                lr2, lr4, lr8, hr = batch[0], batch[1], batch[2], batch[3]
                rest = list(batch[4:])

            lr2 = lr2.to(device)
            lr4 = lr4.to(device)
            lr8 = lr8.to(device)
            hr  = hr.to(device)
            B = hr.shape[0]

            # 找 range_info（可能没有）
            range_tensor = find_range_info_in_batch(rest, device=device, B=B)

            # forward 取三个尺度输出（兼容 tuple/list/dict）
            out = model(lr2, lr4, lr8)
            if isinstance(out, dict):
                sr_x2, sr_x4, sr_x8 = out["x2"], out["x4"], out["x8"]
            elif isinstance(out, (list, tuple)):
                # 有些模型第一个是 dict
                if isinstance(out[0], dict):
                    sr_x2, sr_x4, sr_x8 = out[0]["x2"], out[0]["x4"], out[0]["x8"]
                else:
                    sr_x2, sr_x4, sr_x8 = out[0], out[1], out[2]
            else:
                raise TypeError(f"Unsupported model output type: {type(out)}")

            # 对齐 HR 尺寸
            sr_x2 = torch.nn.functional.interpolate(sr_x2, size=hr.shape[-2:], mode='bicubic', align_corners=False)
            sr_x4 = torch.nn.functional.interpolate(sr_x4, size=hr.shape[-2:], mode='bicubic', align_corners=False)
            sr_x8 = torch.nn.functional.interpolate(sr_x8, size=hr.shape[-2:], mode='bicubic', align_corners=False)

            # 反归一化（如果拿得到 min/max）
            if range_tensor is not None:
                mn = range_tensor[:, 0].view(-1, 1, 1, 1)
                mx = range_tensor[:, 1].view(-1, 1, 1, 1)
                vr = (mx - mn).clamp(min=1e-12)
                sr_x2 = sr_x2 * vr + mn
                sr_x4 = sr_x4 * vr + mn
                sr_x8 = sr_x8 * vr + mn
                hr_dn = hr    * vr + mn
            else:
                hr_dn = hr

            # 逐样本
            for b in range(B):
                hr_np  = hr_dn[b, 0].detach().cpu().numpy()
                sr2_np = sr_x2[b, 0].detach().cpu().numpy()
                sr4_np = sr_x4[b, 0].detach().cpu().numpy()
                sr8_np = sr_x8[b, 0].detach().cpu().numpy()

                hr_slope, hr_aspect = slope_aspect_np(hr_np, cellsize=cellsize)

                for i, sr_np in enumerate([sr2_np, sr4_np, sr8_np]):
                    sr_slope, sr_aspect = slope_aspect_np(sr_np, cellsize=cellsize)

                    # slope
                    e = (sr_slope - hr_slope).reshape(-1)
                    slope_mae_sum[i]  += float(np.mean(np.abs(e)))
                    slope_rmse_sum[i] += float(np.sqrt(np.mean(e**2)))

                    # aspect（环形角度误差）
                    aerr = angle_diff_deg(sr_aspect, hr_aspect).reshape(-1)
                    aspect_mae_sum[i]  += float(np.mean(aerr))
                    aspect_rmse_sum[i] += float(np.sqrt(np.mean(aerr**2)))

                count += 1

    if count == 0:
        z = [0.0, 0.0, 0.0]
        return z, z, z, z

    return (
        [v / count for v in slope_mae_sum],
        [v / count for v in slope_rmse_sum],
        [v / count for v in aspect_mae_sum],
        [v / count for v in aspect_rmse_sum],
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试集
    test_set = DEMDataset(test_root)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    print(f"测试集样本数: {len(test_set)}")

    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']

    # 自动判断 SAFAM 结构
    if any(k.startswith('shared_safam') for k in state_dict.keys()):
        print("检测到使用共享 SAFAM 结构")
        model = MCArbNet_Full(shared_safam=True).to(device)
    else:
        print("检测到使用独立 SAFAM 结构")
        model = MCArbNet_Full().to(device)

    # 获取当前模型结构的 state_dict 以判断 shape
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape == model_state_dict[k].shape:
                new_state_dict[k] = state_dict[k]
            elif 'pos_embed' in k:
                print(f"[调整] {k}: {state_dict[k].shape} -> {model_state_dict[k].shape}")
                old = state_dict[k]
                new = model_state_dict[k].clone()
                new[:, :old.shape[1], :] = old
                new_state_dict[k] = new
            else:
                print(f"[跳过] {k}: {state_dict[k].shape} != {model_state_dict[k].shape}")
        else:
            print(f"[忽略] {k} 不在模型中")

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("开始评估模型 ...")
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating", ncols=100)

    # 1) 原有指标
    psnr_val, ssim_val, mae_val, rmse_val = evaluate_model(model, test_loader_tqdm, device)

    print("\n测试集评估结果：")
    print(f"×2  PSNR: {psnr_val[0]:.4f}, SSIM: {ssim_val[0]:.4f}, MAE: {mae_val[0]:.4f}, RMSE: {rmse_val[0]:.4f}")
    print(f"×4  PSNR: {psnr_val[1]:.4f}, SSIM: {ssim_val[1]:.4f}, MAE: {mae_val[1]:.4f}, RMSE: {rmse_val[1]:.4f}")
    print(f"×8  PSNR: {psnr_val[2]:.4f}, SSIM: {ssim_val[2]:.4f}, MAE: {mae_val[2]:.4f}, RMSE: {rmse_val[2]:.4f}")

    # 2) 新增坡度/坡向指标（注意：这里重新走一遍 loader，所以用原 test_loader）
    print("\n开始评估坡度/坡向 ...")
    test_loader_tqdm2 = tqdm(test_loader, desc="Terrain", ncols=100)
    slope_mae, slope_rmse, aspect_mae, aspect_rmse = evaluate_slope_aspect(
        model, test_loader_tqdm2, device, cellsize=1.0  # 如果知道像元大小，比如 2m/5m，替换这里
    )

    print("\n坡度/坡向评估结果（单位：degree）:")
    print(f"×2  Slope MAE: {slope_mae[0]:.4f}, Slope RMSE: {slope_rmse[0]:.4f}, "
          f"Aspect MAE: {aspect_mae[0]:.4f}, Aspect RMSE: {aspect_rmse[0]:.4f}")
    print(f"×4  Slope MAE: {slope_mae[1]:.4f}, Slope RMSE: {slope_rmse[1]:.4f}, "
          f"Aspect MAE: {aspect_mae[1]:.4f}, Aspect RMSE: {aspect_rmse[1]:.4f}")
    print(f"×8  Slope MAE: {slope_mae[2]:.4f}, Slope RMSE: {slope_rmse[2]:.4f}, "
          f"Aspect MAE: {aspect_mae[2]:.4f}, Aspect RMSE: {aspect_rmse[2]:.4f}")


if __name__ == '__main__':
    main()
