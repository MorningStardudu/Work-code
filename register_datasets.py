"""
步骤2: 注册 VOC2012 消融实验的 Detectron2 数据集
在训练脚本中 import 此模块即可自动注册三个数据集:
  - voc_ablation_train
  - voc_ablation_val
  - voc_ablation_test
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# VOC 20类名称
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tv",
)


def _register():
    """注册 VOC 消融实验数据集"""
    # 自动定位 annotation 目录
    ablation_dir = os.path.dirname(os.path.abspath(__file__))
    ann_root = os.path.join(ablation_dir, "output", "annotations")

    # 查找 VOC2012 JPEGImages 目录
    # 从 ablation/ → SAN-main/ → Project1/ → datasets/VOC2012/
    project_root = os.path.normpath(os.path.join(ablation_dir, "..", ".."))
    voc_img_dir = os.path.join(project_root, "datasets", "VOC2012", "JPEGImages")

    if not os.path.exists(voc_img_dir):
        # 尝试环境变量
        det_root = os.getenv("DETECTRON2_DATASETS", "datasets")
        voc_img_dir = os.path.join(det_root, "VOC2012", "JPEGImages")

    meta = {
        "stuff_classes": list(CLASS_NAMES),
    }

    for name in ["train", "val", "test"]:
        dataset_name = f"voc_ablation_{name}"
        image_dir = voc_img_dir
        gt_dir = os.path.join(ann_root, name)

        if not os.path.exists(gt_dir):
            print(f"[WARNING] Annotation directory not found: {gt_dir}")
            print(f"  Run 'python ablation/1_prepare_splits.py' first.")
            continue

        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )
        n = len(DatasetCatalog.get(dataset_name))
        print(f"[Dataset] {dataset_name}: {n} images")


_register()
