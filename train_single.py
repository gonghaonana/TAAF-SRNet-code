import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from MCArbNet_Single import MCArbNet_Single as MCArbNet
from loss_mc_arb import MultiScaleLoss
from dataset import DEMDataset

# -------------------------------
# 配置
# -------------------------------
scale = 8
train_root = 'E:/jjq/data/ocean/train'
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-5
SAVE_PATH = f'checkpoints_single_ocean_x{scale}'
LOG_FILE = f'train_log_single_ocean_x{scale}.txt'
TENSORBOARD_LOGDIR = os.path.join(SAVE_PATH, 'tensorboard')
SAVE_INTERVAL = 1
RESUME = False
USE_SLOPE = True

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

def log_print(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_set = DEMDataset(train_root)
    print(f"找到 {len(train_set)} 个 TIF 文件在 {train_root}")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = MCArbNet(scale=scale).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = MultiScaleLoss(
        loss_type='l1',
        use_uncertainty=True,
        use_triplet=False,   # ❌ 去除协同特征一致性
        use_fused=False,     # ❌ 去除 fused loss
        use_tv=True,
        use_slope=USE_SLOPE,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1
    best_loss = float('inf')
    last_ckpt = os.path.join(SAVE_PATH, 'epoch-last.pth')

    if RESUME and os.path.exists(last_ckpt):
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 从 epoch {checkpoint['epoch']} 恢复训练")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('Epoch\tLoss\tTV\tSlope\tRecon\n')

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tv = 0.0
        epoch_slope = 0.0
        epoch_recon = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", ncols=120)

        for lr, hr, *_ in loop:
            lr, hr = lr.to(device), hr.to(device)
            preds = model(lr)

            # 构造与 MultiScaleLoss 接口兼容的输入
            fake_preds = {"x2": preds, "x4": preds, "x8": preds}
            fake_feats = {"feat": [preds]*3, "vec": [preds]*3, "fused": preds}

            loss, recon_list, _, _, tv, slope = criterion(fake_preds, hr, fake_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_list[0].item()
            epoch_tv += tv.item()
            epoch_slope += slope.item() if USE_SLOPE else 0.0

            loop.set_postfix(loss=loss.item(), recon=recon_list[0].item())

        n = len(train_loader)
        mean_loss = epoch_loss / n
        mean_recon = epoch_recon / n
        mean_tv = epoch_tv / n
        mean_slope = epoch_slope / n

        log_print(f"{epoch}\t{mean_loss:.6f}\t{mean_tv:.6f}\t{mean_slope:.6f}\t{mean_recon:.6f}")
        writer.add_scalar('Loss/Total', mean_loss, epoch)
        writer.add_scalar('Loss/TV', mean_tv, epoch)
        writer.add_scalar('Loss/Slope', mean_slope, epoch)
        writer.add_scalar('Loss/Reconstruction', mean_recon, epoch)

        # 保存当前模型
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(ckpt, os.path.join(SAVE_PATH, 'epoch-last.pth'))

        # ✅ 只保留最优权重 best_model.pth
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt, os.path.join(SAVE_PATH, 'best_model.pth'))
            log_print(f"🌟 保存最优模型 best_model.pth（Loss: {best_loss:.6f}）")

    writer.close()

if __name__ == '__main__':
    main()
