#!/usr/bin/env python
"""
4-fusion vs 6-fusion 单图推理对比
用法: python ablation/compare_predict.py \
        --img-path /path/to/image.jpg \
        --output-dir ablation/output/comparison
"""

import os, sys
import numpy as np
from PIL import Image
import torch

# 注册 ablation 数据集（避免后续 import 报错）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ablation  # noqa

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color

from san import add_san_config

# VOC 20类
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tv",
]


def setup_cfg(config_file: str, model_path: str, device: str = "cuda"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.SAN.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.SAN.CLIP_PRETRAINED_NAME = "openai"
    cfg.freeze()
    return cfg


def load_predictor(config_file: str, model_path: str, device: str = "cuda"):
    """加载 SAN 模型"""
    cfg = setup_cfg(config_file, model_path, device)

    model = DefaultTrainer.build_model(cfg)
    print(f"Loading: {model_path}")
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_path)
    model.eval()
    if device != "cpu" and torch.cuda.is_available():
        model = model.cuda()
    return model, cfg


def preprocess(image: Image.Image):
    """预处理: RGB + short side resize to 640"""
    image = image.convert("RGB")
    w, h = image.size
    if w < h:
        image = image.resize((640, int(h * 640 / w)))
    else:
        image = image.resize((int(w * 640 / h), 640))
    tensor = torch.from_numpy(np.asarray(image)).float().permute(2, 0, 1)
    return tensor, h, w


def infer(model, image_tensor, h, w, vocabulary):
    """推理, 返回语义分割图 (H,W)"""
    with torch.no_grad():
        result = model([
            {
                "image": image_tensor,
                "height": h,
                "width": w,
                "vocabulary": vocabulary,
            }
        ])[0]["sem_seg"]
    seg_map = result.argmax(dim=0).cpu().numpy()
    seg_map[seg_map >= len(vocabulary)] = len(vocabulary) - 1
    return seg_map


def visualize_seg(image: Image.Image, seg_map: np.ndarray, vocabulary: list,
                  title: str, output_path: str):
    """可视化分割结果"""
    np.random.seed(42)
    colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
    MetadataCatalog.get("_compare").set(
        stuff_classes=vocabulary, stuff_colors=colors
    )
    metadata = MetadataCatalog.get("_compare")

    v = Visualizer(image, metadata)
    v = v.draw_sem_seg(seg_map, area_threshold=0).get_image()
    v = Image.fromarray(v)

    # 不在这里加标题，用拼接后统一标注的方式
    MetadataCatalog.remove("_compare")
    v.save(output_path)
    print(f"  Saved: {output_path}")
    return v


def add_title_to_image(img: Image.Image, title: str, height: int = 40) -> Image.Image:
    """在图片上方添加标题栏"""
    title_img = Image.new("RGB", (img.width, height), (30, 30, 30))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(title_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((img.width - tw) // 2, (height - th) // 2), title,
              fill=(255, 255, 255), font=font)

    result = Image.new("RGB", (img.width, height + img.height))
    result.paste(title_img, (0, 0))
    result.paste(img, (0, height))
    return result


def compare_heatmap(seg4: np.ndarray, seg6: np.ndarray, h: int, w: int) -> Image.Image:
    """生成差异热力图: 红色=4胜, 蓝色=6胜, 灰色=一致"""
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    # 原始分辨率可能和 seg_map 不同, 需要 resize
    seg4_pil = Image.fromarray(seg4.astype(np.uint8))
    seg6_pil = Image.fromarray(seg6.astype(np.uint8))

    seg4_arr = np.array(seg4_pil.resize((w, h), Image.NEAREST))
    seg6_arr = np.array(seg6_pil.resize((w, h), Image.NEAREST))

    agree = (seg4_arr == seg6_arr)
    diff[agree] = [128, 128, 128]  # 灰色 = 一致
    diff[~agree] = [255, 100, 100]  # 红色 = 不一致

    return Image.fromarray(diff)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", default=None, help="输入图片路径 (默认: testImage/test1.png)")
    parser.add_argument("--output-dir", default="ablation/output/comparison",
                        help="输出目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # 自动定位 SAN-main 根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    san_root = os.path.dirname(script_dir)  # ablation/ -> SAN-main/

    # 默认图片: SAN-main/testImage/test1.png
    if args.img_path is None:
        args.img_path = os.path.join(san_root, "testImage", "test1.png")


    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(san_root, args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    ablation_dir = os.path.dirname(os.path.abspath(__file__))

    # 模型配置
    models = {
        "4fusions": {
            "config": os.path.join(ablation_dir, "configs", "san_vit_b16_voc_4fusions.yaml"),
            "weights": os.path.join(ablation_dir, "output", "4fusions_seed42", "model_final.pth"),
        },
        "6fusions": {
            "config": os.path.join(ablation_dir, "configs", "san_vit_b16_voc_6fusions.yaml"),
            "weights": os.path.join(ablation_dir, "output", "6fusions_seed42", "model_final.pth"),
        },
    }

    # 检查权重文件
    for name, m in models.items():
        if not os.path.exists(m["weights"]):
            print(f"ERROR: {name} checkpoint not found: {m['weights']}")
            sys.exit(1)


    # 加载图片
    image = Image.open(args.img_path)
    w_orig, h_orig = image.size
    print(f"Input image: {args.img_path} ({w_orig}x{h_orig})")

    # 预处理
    image_tensor, h_in, w_in = preprocess(image)

    vocab = list(VOC_CLASSES)

    results = {}
    for name, m in models.items():
        print(f"\n{'=' * 50}")
        print(f"  {name}: loading & inferring...")
        model, cfg = load_predictor(m["config"], m["weights"], args.device)

        seg_map = infer(model, image_tensor, h_in, w_in, vocab)
        print(f"  seg_map shape: {seg_map.shape}, unique classes: {len(np.unique(seg_map))}")

        # 可视化
        vis_path = os.path.join(args.output_dir, f"{name}_seg.png")
        vis = visualize_seg(image, seg_map, vocab, name, vis_path)
        results[name] = {"seg_map": seg_map, "vis": vis}

    # 并排对比图: 原图 | 4-fusion | 6-fusion | 差异热力图
    print(f"\n{'=' * 50}")
    print("  Generating comparison panel...")

    diff_map = compare_heatmap(
        results["4fusions"]["seg_map"],
        results["6fusions"]["seg_map"],
        h_in, w_in,
    )
    # resize diff map to match original
    diff_map = diff_map.resize((w_orig, h_orig), Image.NEAREST)

    # 原图
    orig = image.resize((w_orig, h_orig))

    # 拼图: 2x2 grid
    panel_w = w_orig * 2
    panel_h = h_orig * 2
    panel = Image.new("RGB", (panel_w, panel_h))

    # Row 1: 原图 | 4-fusion
    panel.paste(orig, (0, 0))
    panel.paste(results["4fusions"]["vis"].resize((w_orig, h_orig)), (w_orig, 0))

    # Row 2: 6-fusion | 差异图
    panel.paste(results["6fusions"]["vis"].resize((w_orig, h_orig)), (0, h_orig))
    panel.paste(diff_map, (w_orig, h_orig))

    # 添加标题
    panel_with_title = add_title_to_image(panel,
        "Top-Left: Original  |  Top-Right: 4-fusion  |  Bottom-Left: 6-fusion  |  Bottom-Right: Diff (red=disagree)",
        height=36)

    panel_path = os.path.join(args.output_dir, "comparison_panel.png")
    panel_with_title.save(panel_path)
    print(f"\n  Comparison saved to: {panel_path}")

    # 统计差异
    seg4 = results["4fusions"]["seg_map"]
    seg6 = results["6fusions"]["seg_map"]
    total = seg4.size
    disagree = (seg4 != seg6).sum()
    print(f"\n  Statistics:")
    print(f"    Total pixels (inference size): {total}")
    print(f"    Disagree pixels: {disagree} ({100*disagree/total:.2f}%)")
    print(f"    Agree pixels: {total - disagree} ({100*(total-disagree)/total:.2f}%)")


if __name__ == "__main__":
    main()
