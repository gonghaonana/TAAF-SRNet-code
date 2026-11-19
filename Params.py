import time
import torch
from thop import profile
from model_mc_arb import MCArbNet_Full   # 改成你的文件名

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 实例化模型
model = MCArbNet_Full(n_feats=64, n_blocks=4).to(device)
model.eval()

# 2. 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")
print(f"Params (M): {total_params / 1e6:.4f} M")

# 3. 构造假的输入（batch size = 1）
lr2 = torch.randn(1, 1, 32, 32).to(device)
lr4 = torch.randn(1, 1, 16, 16).to(device)
lr8 = torch.randn(1, 1, 8, 8).to(device)

# 4. 统计 FLOPs
flops, params = profile(model, inputs=(lr2, lr4, lr8), verbose=False)

print(f"FLOPs: {flops}")
print(f"FLOPs (G): {flops / 1e9:.4f} G")
print(f"Params from THOP (M): {params / 1e6:.4f} M")

# 5. 统计 Inference Time
# warm-up
with torch.no_grad():
    for _ in range(20):
        _ = model(lr2, lr4, lr8)

# timing
times = []
with torch.no_grad():
    for _ in range(100):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        _ = model(lr2, lr4, lr8)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

avg_time_ms = sum(times) / len(times) * 1000
print(f"Average Inference Time: {avg_time_ms:.4f} ms")