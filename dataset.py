import os
import torch
from torch.utils.data import Dataset
import numpy as np
from rasterio import open as rasterio_open
import torch.nn.functional as F

HR_SIZE = 64

class DEMDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # ✅ 支持 .tif 和 .TIF 等所有大小写形式
        self.file_list = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith('.tif')
        ]

        print(f"找到 {len(self.file_list)} 个 TIF 文件在 {data_dir}")

    def __len__(self):
        return len(self.file_list)

    # def __getitem__(self, idx):
    #     file_path = self.file_list[idx]
    #     with rasterio_open(file_path) as src:
    #         hr_img = src.read(1).astype(np.float32)
    #
    #     if hr_img.shape != (HR_SIZE, HR_SIZE):
    #         raise ValueError(f"图像尺寸应为 {HR_SIZE}x{HR_SIZE}, 但为 {hr_img.shape}")
    #
    #     min_val, max_val = hr_img.min(), hr_img.max()
    #     if max_val > min_val:
    #         hr_img = (hr_img - min_val) / (max_val - min_val)
    #
    #     hr_tensor = torch.from_numpy(hr_img).unsqueeze(0)  # (1, H, W)
    #
    #     def downsample(t, scale):
    #         size = HR_SIZE // scale
    #         return F.interpolate(t.unsqueeze(0), size=(size, size), mode='bicubic', align_corners=False).squeeze(0)
    #
    #     lr_x2 = downsample(hr_tensor, 2)
    #     lr_x4 = downsample(hr_tensor, 4)
    #     lr_x8 = downsample(hr_tensor, 8)
    #
    #     range_info = torch.tensor([min_val, max_val], dtype=torch.float32)
    #     return lr_x2, lr_x4, lr_x8, hr_tensor, range_info


    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with rasterio_open(file_path) as src:
            hr_img = src.read(1).astype(np.float32)

        if hr_img.shape != (HR_SIZE, HR_SIZE):
            raise ValueError(f"图像尺寸应为 {HR_SIZE}x{HR_SIZE}, 但为 {hr_img.shape}")

        # 获取高程范围
        min_val, max_val = hr_img.min(), hr_img.max()
        range_info = torch.tensor([min_val, max_val], dtype=torch.float32)

        # 归一化到 0-1
        if max_val > min_val:
            hr_img = (hr_img - min_val) / (max_val - min_val)

        hr_tensor = torch.from_numpy(hr_img).unsqueeze(0)  # (1, H, W)

        def downsample(t, scale):
            size = HR_SIZE // scale
            return F.interpolate(t.unsqueeze(0), size=(size, size), mode='bicubic', align_corners=False).squeeze(0)

        lr_x2 = downsample(hr_tensor, 2)
        lr_x4 = downsample(hr_tensor, 4)
        lr_x8 = downsample(hr_tensor, 8)

        return lr_x2, lr_x4, lr_x8, hr_tensor, file_path, range_info
