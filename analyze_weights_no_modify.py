import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio

from model_mc_arb import MCArbNet_Full


# =========================================================
# 固定路径：直接用你的
# =========================================================
DEM_PATH = r"C:\Users\lenovo\Desktop\pic\ocean1\HR\869_inp.tif"
CKPT_PATH = r"E:\jjq\meta_srcnn\checkpoints_ocean_Adapt\best_model.pth"
SAVE_DIR = r"C:\Users\lenovo\Desktop\pic\ocean1\HR\weight_mean_results"


# =========================================================
# 读取 DEM
# =========================================================
def read_dem(path):
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
    return dem


# =========================================================
# 单图归一化
# =========================================================
def normalize_dem(dem):
    dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
    dmin = float(np.min(dem))
    dmax = float(np.max(dem))
    dem_norm = (dem - dmin) / (dmax - dmin + 1e-8)
    return dem_norm


# =========================================================
# 生成三尺度输入
# =========================================================
def make_lr_tensor(dem_norm, scale):
    x = torch.from_numpy(dem_norm).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    H, W = x.shape[-2:]
    h = max(1, H // scale)
    w = max(1, W // scale)
    lr = F.interpolate(x, size=(h, w), mode="nearest")
    return lr


# =========================================================
# Hook：抓 AdaptiveMerger 输出
# =========================================================
class MergerHook:
    def __init__(self):
        self.records = []

    def hook_fn(self, module, inputs, outputs):
        out, aux = outputs
        w_ercsr, w_vit, prior = aux
        self.records.append({
            "w_ercsr": w_ercsr.detach().cpu(),
            "w_vit": w_vit.detach().cpu()
        })


def register_hooks(model):
    hook_obj = MergerHook()
    handles = []

    for name, module in model.named_modules():
        if module.__class__.__name__ == "AdaptiveMerger":
            h = module.register_forward_hook(hook_obj.hook_fn)
            handles.append(h)
            print(f"Registered hook on: {name}")

    return hook_obj, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =========================================================
# 加载模型
# =========================================================
def load_model(ckpt_path, device):
    model = MCArbNet_Full(n_feats=64, n_blocks=4).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys()))
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\nLoad model result:")
    print("Missing keys count:", len(missing))
    print("Unexpected keys count:", len(unexpected))

    model.eval()
    return model


# =========================================================
# 主函数
# =========================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读取并归一化 DEM
    dem = read_dem(DEM_PATH)
    dem_norm = normalize_dem(dem)

    # 2. 三尺度输入
    x2 = make_lr_tensor(dem_norm, 2).to(device)
    x4 = make_lr_tensor(dem_norm, 4).to(device)
    x8 = make_lr_tensor(dem_norm, 8).to(device)

    print("Input shapes:")
    print("x2:", tuple(x2.shape))
    print("x4:", tuple(x4.shape))
    print("x8:", tuple(x8.shape))

    # 3. 加载模型
    model = load_model(CKPT_PATH, device)

    # 4. 注册 hook
    hook_obj, handles = register_hooks(model)

    # 5. 推理
    with torch.no_grad():
        _ = model(x2, x4, x8)

    # 6. 移除 hook
    remove_hooks(handles)

    total = len(hook_obj.records)
    print("\nCaptured merger outputs:", total)

    if total == 0:
        print("没有抓到任何融合权重，请检查模型和权重是否匹配。")
        return

    branches = ["x2", "x4", "x8"]
    n_blocks = total // 3

    save_txt = os.path.join(SAVE_DIR, "weight_mean_summary.txt")

    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("Fusion Weight Mean Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"DEM_PATH  : {DEM_PATH}\n")
        f.write(f"CKPT_PATH : {CKPT_PATH}\n")
        f.write(f"TOTAL_CAPTURED : {total}\n\n")

        for bi, branch in enumerate(branches):
            start_idx = bi * n_blocks
            end_idx = (bi + 1) * n_blocks

            cnn_branch_means = []
            vit_branch_means = []

            print(f"\n===== {branch} =====")
            f.write(f"===== {branch} =====\n")

            for local_i, idx in enumerate(range(start_idx, end_idx), start=1):
                rec = hook_obj.records[idx]

                w_ercsr = rec["w_ercsr"]
                w_vit = rec["w_vit"]

                cnn_mean = float(w_ercsr.mean().item())
                vit_mean = float(w_vit.mean().item())

                cnn_branch_means.append(cnn_mean)
                vit_branch_means.append(vit_mean)

                print(f"Block {local_i}: CNN mean = {cnn_mean:.6f}, ViT mean = {vit_mean:.6f}")
                f.write(f"Block {local_i}: CNN mean = {cnn_mean:.6f}, ViT mean = {vit_mean:.6f}\n")

            cnn_branch_avg = float(np.mean(cnn_branch_means))
            vit_branch_avg = float(np.mean(vit_branch_means))

            print(f"{branch} overall mean -> CNN = {cnn_branch_avg:.6f}, ViT = {vit_branch_avg:.6f}")
            f.write(f"{branch} overall mean -> CNN = {cnn_branch_avg:.6f}, ViT = {vit_branch_avg:.6f}\n\n")

    print("\n完成。平均值结果已保存到：")
    print(save_txt)


if __name__ == "__main__":
    main()