import os
import random
import shutil
from glob import glob
from tqdm import tqdm

def split_dataset(
    source_dir,
    output_dir,
    exts=(".tif", ".tiff"),
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    move=False  # True 表示移动，False 表示复制
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例必须加起来为1"

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # 获取所有 tif 文件
    file_list = []
    for ext in exts:
        file_list.extend(glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
    print(f"共找到 {len(file_list)} 个文件")

    # 随机打乱
    random.seed(seed)
    random.shuffle(file_list)

    n_total = len(file_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    datasets = {
        train_dir: file_list[:n_train],
        val_dir: file_list[n_train:n_train + n_val],
        test_dir: file_list[n_train + n_val:]
    }

    # 拷贝或移动
    for target_dir, files in datasets.items():
        for f in tqdm(files, desc=f"Processing {os.path.basename(target_dir)}"):
            basename = os.path.basename(f)
            dst = os.path.join(target_dir, basename)
            if move:
                shutil.move(f, dst)
            else:
                shutil.copy2(f, dst)

    print("✅ 数据集划分完成：")
    print(f"  Train: {n_train} files")
    print(f"  Val:   {n_val} files")
    print(f"  Test:  {n_test} files")


# ================================
# 使用示例（修改下面路径后直接运行）
# ================================
if __name__ == "__main__":
    split_dataset(
        source_dir="E:/jjq/my_data",        # 👈 修改为你的数据目录
        output_dir="E:/jjq/data3",  # 👈 输出划分后的目录
        exts=(".tif",),                      # 👈 文件后缀名
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=2024,
        move=False                           # True: 移动 | False: 复制
    )
