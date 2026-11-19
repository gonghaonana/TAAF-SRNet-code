import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_mc_arb import MCArbNet_Full as MCArbNet
from loss_mc_arb import MultiScaleLoss
from dataset import DEMDataset

# -------------------------------
# 配置
# -------------------------------
train_root = 'E:/jjq/data/coast/1/train'
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-4
SAVE_PATH = 'checkpoints_coast_Adapt_1e-4'
LOG_FILE = 'train_log_coast_Adapt_1e-4.txt'
SAVE_INTERVAL = 1
RESUME = False
USE_SLOPE = True  # 是否启用 Slope loss
TENSORBOARD_LOGDIR = os.path.join(SAVE_PATH, 'tensorboard')

# -------------------------------
# 准备目录 & TensorBoard
# -------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

def log_print(msg: str):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def main():
    # ------------ 设备 ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ------------ 数据 ------------
    train_set = DEMDataset(train_root)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,         # 你原来的设置
        pin_memory=True,       # 小优化：CUDA 传输更快（CPU→GPU）
        drop_last=True         # 建议：保证每个 batch 尺寸一致
    )
    print(f"训练数据集大小: {len(train_set)}")

    # ------------ 模型 ------------
    model = MCArbNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ------------ 损失 & 优化器 ------------
    criterion = MultiScaleLoss(
        loss_type='l1',
        use_uncertainty=True,
        use_triplet=True,
        use_fused=True,
        use_tv=True,
        use_slope=USE_SLOPE,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)  # ✅ 固定学习率，无 scheduler

    start_epoch = 1
    best_loss_x8 = float('inf')
    last_ckpt = os.path.join(SAVE_PATH, 'epoch-last.pth')

    # ------------ 恢复训练（可选） ------------
    if RESUME and os.path.exists(last_ckpt):
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 从 epoch {checkpoint['epoch']} 恢复训练")

    # ------------ 初始化日志文件 ------------
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('Epoch\tLoss\tTri\tFused\tSlope\tX2\tX4\tX8\n')

    # ====================================================================
    #                               训练循环
    # ====================================================================
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        recon_loss_total = [0.0, 0.0, 0.0]
        tri_loss_total = 0.0
        fused_loss_total = 0.0
        slope_loss_total = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", ncols=120)

        for batch in loop:
            # ------- 兼容 tuple/list/dict 返回，安全吸收多余项 -------
            if isinstance(batch, dict):
                # 假定常见键名；如你的 Dataset 用了别名（如 'x2'），可在这里补充候选键
                def pick(d, keys):
                    for k in keys:
                        if k in d:
                            return d[k]
                    raise KeyError(f"Missing keys {keys} in batch dict. Got keys: {list(d.keys())}")

                lr2 = pick(batch, ['lr2', 'x2', 'lr_x2']).to(device, non_blocking=True)
                lr4 = pick(batch, ['lr4', 'x4', 'lr_x4']).to(device, non_blocking=True)
                lr8 = pick(batch, ['lr8', 'x8', 'lr_x8']).to(device, non_blocking=True)
                hr  = pick(batch, ['hr', 'gt', 'target']).to(device, non_blocking=True)
                # 其他返回项（如 slope/fname/mask）可从 batch 继续取
            else:
                # tuple/list: 取前四项为 (lr2, lr4, lr8, hr)，其余全部忽略
                if not isinstance(batch, (list, tuple)) or len(batch) < 4:
                    raise ValueError(f"Dataset batch should provide at least 4 items (lr2, lr4, lr8, hr), got: {type(batch)} with len={len(batch) if isinstance(batch,(list,tuple)) else 'NA'}")
                lr2, lr4, lr8, hr, *rest = batch
                lr2 = lr2.to(device, non_blocking=True)
                lr4 = lr4.to(device, non_blocking=True)
                lr8 = lr8.to(device, non_blocking=True)
                hr  = hr.to(device,  non_blocking=True)
                # rest 中包含的额外返回（如 slope/fname/mask）此处不使用

            preds, features = model(lr2, lr4, lr8)
            loss, recon_list, triplet_loss, fused_loss, tv, slope_loss = criterion(preds, hr, features)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ------ 统计 ------
            epoch_loss += loss.item()
            for i in range(3):
                recon_loss_total[i] += recon_list[i].item()
            tri_loss_total += triplet_loss.item()
            fused_loss_total += fused_loss.item()
            slope_loss_total += (slope_loss.item() if USE_SLOPE else 0.0)

            loop.set_postfix(
                avg=f"{epoch_loss / (loop.n + 1):.3f}",
                now=f"{loss.item():.3f}",
                tri=f"{triplet_loss.item():.6f}",
                fused=f"{fused_loss.item():.6f}",
                slope=f"{slope_loss.item() if USE_SLOPE else 0:.6f}",
                x2=f"{recon_list[0].item():.4f}",
                x4=f"{recon_list[1].item():.4f}",
                x8=f"{recon_list[2].item():.4f}"
            )

        # ------ 日志 & TensorBoard ------
        mean_loss = epoch_loss / len(train_loader)
        mean_tri = tri_loss_total / len(train_loader)
        mean_fused = fused_loss_total / len(train_loader)
        mean_slope = slope_loss_total / len(train_loader)
        mean_x2 = recon_loss_total[0] / len(train_loader)
        mean_x4 = recon_loss_total[1] / len(train_loader)
        mean_x8 = recon_loss_total[2] / len(train_loader)

        log_print(
            f"{epoch}\t{mean_loss:.6f}\t{mean_tri:.6f}\t{mean_fused:.6f}\t{mean_slope:.6f}\t{mean_x2:.6f}\t{mean_x4:.6f}\t{mean_x8:.6f}"
        )

        writer.add_scalar('Loss/Total', mean_loss, epoch)
        writer.add_scalar('Loss/Triplet', mean_tri, epoch)
        writer.add_scalar('Loss/Fused', mean_fused, epoch)
        writer.add_scalar('Loss/Slope', mean_slope, epoch)
        writer.add_scalar('Loss/Reconstruction/x2', mean_x2, epoch)
        writer.add_scalar('Loss/Reconstruction/x4', mean_x4, epoch)
        writer.add_scalar('Loss/Reconstruction/x8', mean_x8, epoch)

        # ------ 保存 ckpt ------
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(SAVE_PATH, 'epoch-last.pth'))
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            torch.save(ckpt, os.path.join(SAVE_PATH, f'epoch-{epoch}.pth'))
            log_print(f"✅ 保存模型 epoch-{epoch}.pth")

        # ------ 保存最优 (x8) ------
        if mean_x8 < best_loss_x8:
            best_loss_x8 = mean_x8
            torch.save(ckpt, os.path.join(SAVE_PATH, 'best_model.pth'))
            log_print(f"🌟 保存最优模型 best_model.pth（x8 Loss: {best_loss_x8:.6f}）")

    writer.close()


if __name__ == '__main__':
    main()
