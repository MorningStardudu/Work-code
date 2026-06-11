#!/usr/bin/env python
"""
步骤1: 生成 VOC2012 消融实验数据集划分
输出:
  ablation/output/splits/train.txt   — 训练集图像ID
  ablation/output/splits/val.txt     — 验证集图像ID
  ablation/output/splits/test.txt    — 测试集图像ID
  ablation/output/annotations/{train,val,test}/ — 重映射后的标注PNG
"""

import argparse
import os
import random
from PIL import Image
import numpy as np

# 像素值重映射: VOC原始 → 训练用ID
# 原始: 0=bg, 1=aeroplane, 2=bicycle, ..., 20=tv, 255=ignore
# 目标: 0=aeroplane, 1=bicycle, ..., 19=tv, 255=ignore (背景归为ignore)
REMAP = {
    0: 255,
    1: 0,   2: 1,   3: 2,   4: 3,   5: 4,
    6: 5,   7: 6,   8: 7,   9: 8,   10: 9,
    11: 10, 12: 11, 13: 12, 14: 13, 15: 14,
    16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
    255: 255,
}

# VOC 20类名称 (与 register_voc.py 一致)
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VOC splits for ablation")
    parser.add_argument("--data-root", default="../datasets/VOC2012",
                        help="Path to VOC2012 dataset")
    parser.add_argument("--output-dir", default="ablation/output",
                        help="Output directory for splits and annotations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio of training data to use as validation")
    return parser.parse_args()


def load_image_ids(filepath):
    """从ImageSets/Segmentation的txt文件读取图像ID列表"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_image_ids(filepath, ids):
    with open(filepath, 'w') as f:
        for img_id in sorted(ids):
            f.write(img_id + '\n')
    print(f"  Saved {len(ids)} IDs → {filepath}")


def remap_annotation(src_path, dst_path):
    """读取原始标注PNG, 重映射像素值, 保存到目标路径"""
    mask = np.array(Image.open(src_path))
    # 使用向量化映射
    remapped = np.full_like(mask, 255, dtype=np.uint8)
    for src_val, dst_val in REMAP.items():
        remapped[mask == src_val] = dst_val

    # 跳过全忽略的标注
    if len(np.unique(remapped)) == 1 and np.unique(remapped)[0] == 255:
        return False

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    Image.fromarray(remapped).save(dst_path, "PNG")
    return True


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)

    # 路径定义
    train_txt = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    val_txt = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")
    seg_dir = os.path.join(data_root, "SegmentationClassAug")
    img_dir = os.path.join(data_root, "JPEGImages")

    # 验证路径
    for p in [train_txt, val_txt, seg_dir, img_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # 创建输出目录
    splits_dir = os.path.join(output_dir, "splits")
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(splits_dir, exist_ok=True)

    # === 1. 加载原始划分 ===
    print("=" * 60)
    print("步骤 1: 加载 VOC2012 原始划分")
    train_ids = load_image_ids(train_txt)
    val_ids = load_image_ids(val_txt)
    print(f"  原始 train: {len(train_ids)} 张")
    print(f"  原始 val:   {len(val_ids)} 张")

    # === 2. 划分 train/val/test ===
    print("\n步骤 2: 生成三集合划分")
    random.seed(args.seed)
    train_ids_shuffled = train_ids.copy()
    random.shuffle(train_ids_shuffled)

    n_val = max(1, int(len(train_ids_shuffled) * args.val_ratio))
    ablation_train = train_ids_shuffled[n_val:]    # ~90%
    ablation_val = train_ids_shuffled[:n_val]       # ~10%
    ablation_test = val_ids                         # 官方val → test

    save_image_ids(os.path.join(splits_dir, "train.txt"), ablation_train)
    save_image_ids(os.path.join(splits_dir, "val.txt"), ablation_val)
    save_image_ids(os.path.join(splits_dir, "test.txt"), ablation_test)

    # === 3. 重映射标注 ===
    print("\n步骤 3: 重映射标注PNG (0-20 → 0-19, bg归为ignore)")
    for split_name, split_ids in [
        ("train", ablation_train),
        ("val", ablation_val),
        ("test", ablation_test),
    ]:
        split_ann_dir = os.path.join(ann_dir, split_name)
        os.makedirs(split_ann_dir, exist_ok=True)
        skipped = 0
        for img_id in split_ids:
            src = os.path.join(seg_dir, img_id + ".png")
            dst = os.path.join(split_ann_dir, img_id + ".png")
            if not remap_annotation(src, dst):
                skipped += 1
        print(f"  [{split_name}] {len(split_ids) - skipped} 张标注已保存"
              + (f" (跳过 {skipped} 张全忽略)" if skipped else ""))

    # === 4. 统计 ===
    print("\n" + "=" * 60)
    print("数据集划分完成!")
    print(f"  Train: {len(ablation_train)} 张")
    print(f"  Val:   {len(ablation_val)} 张")
    print(f"  Test:  {len(ablation_test)} 张")
    # 验证无重叠
    overlap_tv = set(ablation_train) & set(ablation_val)
    overlap_tt = set(ablation_train) & set(ablation_test)
    overlap_vt = set(ablation_val) & set(ablation_test)
    assert not overlap_tv, f"Train-Val overlap: {overlap_tv}"
    assert not overlap_tt, f"Train-Test overlap: {overlap_tt}"
    assert not overlap_vt, f"Val-Test overlap: {overlap_vt}"
    print("  [OK] 三集合无重叠")
    print(f"\n操作完成。数据准备文件位于: {output_dir}")


if __name__ == "__main__":
    main()
