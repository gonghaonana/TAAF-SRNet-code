import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from MCArbNet_Single import MCArbNet_Single
from dataset import DEMDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# -------------------------------
# 配置
# -------------------------------
scale = 4
test_root = 'E:/jjq/data/ocean/test'
ckpt_path = f'checkpoints_single_ocean_x{scale}/best_model.pth'
BATCH_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 加载模型
# -------------------------------
model = MCArbNet_Single(scale=scale).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"✅ 加载模型成功: {ckpt_path}")

# -------------------------------
# 加载数据
# -------------------------------
test_set = DEMDataset(test_root)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"测试集大小: {len(test_set)}")

# -------------------------------
# 初始化指标
# -------------------------------
total_psnr, total_ssim, total_mae, total_rmse = 0.0, 0.0, 0.0, 0.0

# -------------------------------
# 推理并评估
# -------------------------------
with torch.no_grad():
    for data in tqdm(test_loader, desc="Evaluating"):
        lr, hr = data[0].to(device), data[1]
        sr = model(lr)
        sr = F.interpolate(sr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        sr_np = sr.cpu().numpy()
        hr_np = hr.numpy()

        for i in range(hr_np.shape[0]):
            sr_img = np.clip(np.squeeze(sr_np[i]), 0, 1)
            hr_img = np.clip(np.squeeze(hr_np[i]), 0, 1)

            total_psnr += psnr(hr_img, sr_img, data_range=1.0)
            total_ssim += ssim(hr_img, sr_img, data_range=1.0)
            total_mae += mean_absolute_error(hr_img, sr_img)
            total_rmse += mean_squared_error(hr_img, sr_img) ** 0.5



# -------------------------------
# 输出平均指标
# -------------------------------
n = len(test_set)
print(f"\n✅ 测试集指标 (x{scale}):")
print(f"  PSNR : {total_psnr / n:.4f}")
print(f"  SSIM : {total_ssim / n:.4f}")
print(f"  MAE  : {total_mae / n:.4f}")
print(f"  RMSE : {total_rmse / n:.4f}")
