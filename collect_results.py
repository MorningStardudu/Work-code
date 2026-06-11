#!/usr/bin/env python
"""
步骤6: 汇总所有实验结果 — 准确率 + 效率 → 最终对比表
用法: python ablation/6_collect_results.py
"""

import os
import re
import json
import sys
from collections import defaultdict

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABLATION_DIR, "output")

EXPERIMENTS = ["4fusions", "6fusions"]
SEEDS = [42, 123, 999]
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tv",
]


def parse_miou_from_log(log_path):
    """从 eval_log.txt 解析 mIoU"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 查找 SemSegEvaluator 输出的 mIoU
    # detectron2 输出格式: 'mIoU': 88.6609 (OrderedDict repr, 单引号key)
    patterns = [
        r"'mIoU'[\s:]*(\d+\.\d+)",
        r"\"mIoU\"\s*:\s*(\d+\.\d+)",
        r"mIoU['\":\s]+(\d+\.\d+)",
        r"mean IoU['\":\s]+(\d+\.\d+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, content)
        if matches:
            return float(matches[-1])  # 取最后一个 (最终结果)

    return None


def parse_per_class_iou(log_path):
    """从 eval_log.txt 解析 per-class IoU"""
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Detectron2 OrderedDict 格式: 'IoU-aeroplane': 98.9483
    per_class = {}
    for cls_name in CLASS_NAMES:
        # 匹配 'IoU-{cls_name}': 98.9483
        pat = rf"IoU-{cls_name}['\":\s]+(\d+\.\d+)"
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            per_class[cls_name] = float(matches[-1])

    return per_class


def parse_train_time(time_log_path):
    """读取训练时间"""
    if not os.path.exists(time_log_path):
        return None
    with open(time_log_path, "r") as f:
        content = f.read()
    m = re.search(r"train_time_seconds=(\d+)", content)
    if m:
        return int(m.group(1))
    return None


def parse_eval_fps(eval_time_path):
    """读取推理速度"""
    if not os.path.exists(eval_time_path):
        return None
    with open(eval_time_path, "r") as f:
        content = f.read()
    m = re.search(r"eval_fps=(\d+\.?\d*)", content)
    if m:
        return float(m.group(1))
    return None


def calc_mean_std(values):
    """计算均值和标准差"""
    if not values:
        return None, None
    import statistics
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    print("=" * 80)
    print("  SAN 消融实验 — 结果汇总")
    print("=" * 80)

    # 收集准确率结果
    accuracy = {}
    for exp_name in EXPERIMENTS:
        mious = []
        per_class_all = defaultdict(list)

        for seed in SEEDS:
            log_path = os.path.join(OUTPUT_DIR, f"{exp_name}_seed{seed}", "test_eval", "eval_log.txt")
            miou = parse_miou_from_log(log_path)
            if miou is not None:
                mious.append(miou)

            per_class = parse_per_class_iou(log_path)
            for cls_name, iou in per_class.items():
                per_class_all[cls_name].append(iou)

        mean_miou, std_miou = calc_mean_std(mious)
        accuracy[exp_name] = {
            "mious": mious,
            "mean_miou": mean_miou,
            "std_miou": std_miou,
            "per_class": {cls: calc_mean_std(vals) for cls, vals in per_class_all.items()},
        }

    # 收集效率结果
    efficiency = {}
    efficiency_file = os.path.join(OUTPUT_DIR, "results", "efficiency.json")
    if os.path.exists(efficiency_file):
        with open(efficiency_file, "r") as f:
            efficiency_raw = json.load(f)

        # 映射名称
        name_map = {
            "4-fusion (对照组)": "4fusions",
            "6-fusion (实验组)": "6fusions",
        }
        for raw_name, exp_name in name_map.items():
            if raw_name in efficiency_raw:
                efficiency[exp_name] = efficiency_raw[raw_name]

    # 收集训练时间
    train_times = {}
    for exp_name in EXPERIMENTS:
        times = []
        for seed in SEEDS:
            time_log = os.path.join(OUTPUT_DIR, f"{exp_name}_seed{seed}", "train_time.txt")
            t = parse_train_time(time_log)
            if t is not None:
                times.append(t)
        if times:
            mean_t, std_t = calc_mean_std(times)
            train_times[exp_name] = {"times": times, "mean_s": mean_t, "std_s": std_t}

    # 收集推理速度
    eval_fps = {}
    for exp_name in EXPERIMENTS:
        fps_vals = []
        for seed in SEEDS:
            fps_file = os.path.join(OUTPUT_DIR, f"{exp_name}_seed{seed}", "test_eval", "eval_time.txt")
            fps = parse_eval_fps(fps_file)
            if fps is not None:
                fps_vals.append(fps)
        if fps_vals:
            mean_fps, std_fps = calc_mean_std(fps_vals)
            eval_fps[exp_name] = {"fps_vals": fps_vals, "mean_fps": mean_fps, "std_fps": std_fps}

    # ============================================================
    # 输出汇总表
    # ============================================================

    # 表1: 准确率 + 效率汇总
    print(f"\n{'=' * 80}")
    print(f"  表1: 综合对比")
    print(f"{'=' * 80}")

    # 表头
    header = f"{'指标':<28s}"
    for exp in EXPERIMENTS:
        header += f"  {exp:<24s}"
    header += f"  {'Δ':<14s}"
    print(header)
    print("-" * 90)

    def print_row(metric_name, val4, val6, unit="", higher_is_better=True):
        row = f"{metric_name:<28s}"
        row += f"  {val4:<24s}"
        row += f"  {val6:<24s}"

        # 计算差值
        try:
            v4 = float(val4.split("±")[0].strip()) if "±" in val4 else float(val4)
            v6 = float(val6.split("±")[0].strip()) if "±" in val6 else float(val6)
            diff = v6 - v4
            arrow = "↑" if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better) else "↓" if diff != 0 else "="
            row += f"  {diff:+.2f} {arrow:<4s}"
        except (ValueError, AttributeError):
            row += f"  {'N/A':<14s}"

        print(row)

    # mIoU
    for exp in EXPERIMENTS:
        if accuracy[exp]["mean_miou"] is not None:
            a4 = f"{accuracy['4fusions']['mean_miou']:.2f} ± {accuracy['4fusions']['std_miou']:.2f}"
            a6 = f"{accuracy['6fusions']['mean_miou']:.2f} ± {accuracy['6fusions']['std_miou']:.2f}"
            print_row("mIoU (Test)", a4, a6, higher_is_better=True)

    # 参数量
    if efficiency:
        p4 = efficiency.get("4fusions", {}).get("total_params")
        p6 = efficiency.get("6fusions", {}).get("total_params")
        if p4 and p6:
            print_row("总参数", f"{p4/1e6:.2f}M", f"{p6/1e6:.2f}M", higher_is_better=False)

        tr4 = efficiency.get("4fusions", {}).get("trainable_params")
        tr6 = efficiency.get("6fusions", {}).get("trainable_params")
        if tr4 and tr6:
            print_row("可训练参数", f"{tr4/1e6:.1f}M", f"{tr6/1e6:.1f}M", higher_is_better=False)

    # FLOPs
    if efficiency:
        f4 = efficiency.get("4fusions", {}).get("flops")
        f6 = efficiency.get("6fusions", {}).get("flops")
        if f4 and f6:
            print_row("FLOPs", f"{f4/1e9:.2f}G", f"{f6/1e9:.2f}G", higher_is_better=False)

    # 训练时间
    if train_times:
        t4 = train_times.get("4fusions", {})
        t6 = train_times.get("6fusions", {})
        if t4 and t6:
            h4 = t4["mean_s"] / 3600
            h6 = t6["mean_s"] / 3600
            print_row("训练时间 (20K iter)", f"{h4:.1f}h", f"{h6:.1f}h", higher_is_better=False)

    # 推理FPS
    if eval_fps:
        fps4 = eval_fps.get("4fusions", {})
        fps6 = eval_fps.get("6fusions", {})
        if fps4 and fps6:
            print_row("推理速度 (FPS)", f"{fps4['mean_fps']:.1f}", f"{fps6['mean_fps']:.1f}", higher_is_better=True)

    # GPU显存
    if efficiency:
        m4 = efficiency.get("4fusions", {}).get("gpu_memory_gb")
        m6 = efficiency.get("6fusions", {}).get("gpu_memory_gb")
        if m4 and m6:
            print_row("GPU显存 (推理)", f"{m4:.2f} GB", f"{m6:.2f} GB", higher_is_better=False)

    # 表2: Per-class IoU 对比
    print(f"\n{'=' * 80}")
    print(f"  表2: Per-Class IoU 对比")
    print(f"{'=' * 80}")
    print(f"{'类别':<18s}  {'4-fusion':<16s}  {'6-fusion':<16s}  {'Δ':<10s}")
    print("-" * 70)
    for cls_name in CLASS_NAMES:
        pc4 = accuracy["4fusions"]["per_class"].get(cls_name, (None, None))
        pc6 = accuracy["6fusions"]["per_class"].get(cls_name, (None, None))
        if pc4[0] is not None and pc6[0] is not None:
            diff = pc6[0] - pc4[0]
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{cls_name:<18s}  {pc4[0]:>5.1f} ± {pc4[1]:>4.1f}   {pc6[0]:>5.1f} ± {pc6[1]:>4.1f}   {diff:+.1f} {arrow}")

    # 保存JSON结果
    results_json = {
        "accuracy": {},
        "efficiency": {},
        "train_time": {},
        "eval_fps": {},
    }
    for exp in EXPERIMENTS:
        if accuracy[exp]["mean_miou"] is not None:
            results_json["accuracy"][exp] = {
                "mean_miou": accuracy[exp]["mean_miou"],
                "std_miou": accuracy[exp]["std_miou"],
                "individual_mious": accuracy[exp]["mious"],
                "per_class": {
                    cls: {"mean": v[0], "std": v[1]}
                    for cls, v in accuracy[exp]["per_class"].items()
                    if v[0] is not None
                },
            }
    if efficiency:
        results_json["efficiency"] = efficiency
    if train_times:
        results_json["train_time"] = {
            exp: {
                "times_seconds": data["times"],
                "mean_hours": data["mean_s"] / 3600,
                "std_hours": data["std_s"] / 3600 if data["std_s"] else 0,
            }
            for exp, data in train_times.items()
        }
    if eval_fps:
        results_json["eval_fps"] = {
            exp: {
                "fps_values": data["fps_vals"],
                "mean_fps": data["mean_fps"],
                "std_fps": data["std_fps"],
            }
            for exp, data in eval_fps.items()
        }

    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "results", "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存至: {json_path}")


if __name__ == "__main__":
    main()
