#!/usr/bin/env python
"""
步骤5: 效率评估 — 参数量 / FLOPs / 显存使用
在训练之前独立运行, 快速比较两个配置的效率指标
用法: python ablation/5_efficiency.py
"""

import os
import sys
import json
import time

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

from san.config import add_san_config

# 自动注册数据集 (文件名含数字前缀，通过 importlib 导入)
import importlib.util
_reg_spec = importlib.util.spec_from_file_location(
    "register_dataset",
    os.path.join(os.path.dirname(__file__), "register_datasets.py")
)
_reg_mod = importlib.util.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg_mod)


def setup_cfg(config_path):
    """从YAML配置构建CfgNode"""
    cfg = get_cfg()
    add_san_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def measure_params(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, input_tensor):
    """使用 fvcore 测量 FLOPs"""
    # SAN model.forward 需要 batched_inputs 格式
    # 我们使用更简单的方式: 对子模块分别统计
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()
    except Exception as e:
        print(f"  [WARNING] FLOPs measurement failed: {e}")
        print(f"  Falling back to manual estimation...")
        return None


def measure_memory(model, input_dict, device="cuda"):
    """测量推理峰值显存"""
    if device != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        _ = model([input_dict])

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    return peak_mem


def create_dummy_input(cfg):
    """创建虚拟输入用于效率测量"""
    h, w = 640, 640
    image = torch.randn(3, h, w)
    return {
        "image": image,
        "height": h,
        "width": w,
        "image_id": 0,
        "meta": {"dataset_name": "voc_ablation_train"},
    }


def main():
    print("=" * 70)
    print("  SAN 消融实验 — 效率评估")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    experiments = {
        "4-fusion (对照组)": "ablation/configs/san_vit_b16_voc_4fusions.yaml",
        "6-fusion (实验组)": "ablation/configs/san_vit_b16_voc_6fusions.yaml",
    }

    results = {}

    for name, config_path in experiments.items():
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"  Config: {config_path}")
        print(f"{'─' * 70}")

        cfg = setup_cfg(config_path)

        # 构建模型
        print("  构建模型...")
        model = build_model(cfg)
        model.to(device)
        model.eval()

        # 1. 参数量
        total_p, trainable_p = measure_params(model)
        print(f"  参数量:")
        print(f"    总计:     {total_p:,} ({total_p/1e6:.2f}M)")
        print(f"    可训练:   {trainable_p:,} ({trainable_p/1e6:.2f}M)")
        print(f"    冻结:     {total_p - trainable_p:,} ({(total_p - trainable_p)/1e6:.2f}M)")

        # 统计各子模块参数
        module_params = {}
        for module_name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                module_params[module_name] = n

        print(f"  子模块参数分布:")
        for mn, mp in sorted(module_params.items(), key=lambda x: -x[1]):
            bar = "█" * int(30 * mp / total_p)
            print(f"    {mn:30s}: {mp:>12,} ({mp/total_p*100:5.1f}%) {bar}")

        # 2. FLOPs
        print(f"  测量 FLOPs...")
        dummy_input = create_dummy_input(cfg)
        dummy_input["image"] = dummy_input["image"].to(device)

        # fvcore FlopCountAnalysis 对复杂模型可能失败
        # 使用更稳健的方法: 只测 side_adapter_network 的 FLOPs
        flops_total = None
        try:
            # 尝试用 fvcore 对完整模型测FLOPs
            # 需要包装输入
            from fvcore.nn import FlopCountAnalysis

            # 创建 batch of 1
            batched = [dummy_input]
            # 对完整模型可能失败, 先尝试
            flops_total = measure_flops(model, batched)
        except Exception as e:
            print(f"    完整模型FLOPs测量失败: {e}")

        # 测量 side_adapter 子模块
        if hasattr(model, 'side_adapter_network'):
            sa = model.side_adapter_network
            try:
                # 创建适合side adapter的dummy输入
                dummy_img = torch.randn(1, 3, 640, 640).to(device)
                # clip_features 字典
                clip_feat = {}
                for i in range(10):
                    clip_feat[i] = torch.randn(1, 768, 40, 40).to(device)
                    clip_feat[f"{i}_cls_token"] = torch.randn(1, 1, 768).to(device)

                sa_flops = FlopCountAnalysis(sa, (dummy_img, clip_feat))
                sa_total = sa_flops.total()
                print(f"  Side Adapter FLOPs: {sa_total/1e9:.2f}G")
                flops_total = sa_total
            except Exception as e:
                print(f"    Side Adapter FLOPs测量失败: {e}")

        # 3. 显存
        gpu_mem = None
        if device == "cuda":
            print(f"  测量 GPU 显存...")
            try:
                gpu_mem = measure_memory(model, dummy_input, device)
                print(f"  推理峰值显存: {gpu_mem:.2f} GB")
            except Exception as e:
                print(f"    显存测量失败: {e}")

        results[name] = {
            "total_params": total_p,
            "trainable_params": trainable_p,
            "frozen_params": total_p - trainable_p,
            "flops": flops_total,
            "gpu_memory_gb": gpu_mem,
            "module_params": module_params,
        }

    # 汇总对比
    print(f"\n{'=' * 70}")
    print(f"  效率对比汇总")
    print(f"{'=' * 70}")
    header = f"{'指标':<25s}"
    for name in experiments:
        header += f"  {name:<25s}"
    header += f"  {'Δ':<15s}"
    print(header)
    print("-" * 95)

    # 参数量
    t4 = results["4-fusion (对照组)"]["total_params"]
    t6 = results["6-fusion (实验组)"]["total_params"]
    print(f"{'总参数':<25s}  {t4:>12,} ({t4/1e6:.1f}M)    {t6:>12,} ({t6/1e6:.1f}M)    {t6-t4:>+12,} ({(t6-t4)/1e6:+.1f}M)")

    tr4 = results["4-fusion (对照组)"]["trainable_params"]
    tr6 = results["6-fusion (实验组)"]["trainable_params"]
    print(f"{'可训练参数':<25s}  {tr4:>12,} ({tr4/1e6:.1f}M)    {tr6:>12,} ({tr6/1e6:.1f}M)    {tr6-tr4:>+12,} ({(tr6-tr4)/1e6:+.1f}M)")

    f4 = results["4-fusion (对照组)"].get("flops")
    f6 = results["6-fusion (实验组)"].get("flops")
    if f4 and f6:
        print(f"{'FLOPs':<25s}  {f4:>12,.0f} ({f4/1e9:.2f}G)   {f6:>12,.0f} ({f6/1e9:.2f}G)   {f6-f4:>+12,.0f} ({(f6-f4)/1e9:+.2f}G)")

    m4 = results["4-fusion (对照组)"].get("gpu_memory_gb")
    m6 = results["6-fusion (实验组)"].get("gpu_memory_gb")
    if m4 and m6:
        print(f"{'GPU显存(推理)':<25s}  {m4:>17.2f} GB      {m6:>17.2f} GB      {m6-m4:>+17.2f} GB")

    # 保存结果
    os.makedirs("ablation/output/results", exist_ok=True)
    # 转换结果以便JSON序列化
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            "total_params": v["total_params"],
            "trainable_params": v["trainable_params"],
            "frozen_params": v["frozen_params"],
            "flops": v["flops"],
            "gpu_memory_gb": v["gpu_memory_gb"],
            "module_params": {mk: mv for mk, mv in v["module_params"].items()},
        }

    with open("ablation/output/results/efficiency.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n结果已保存至: ablation/output/results/efficiency.json")


if __name__ == "__main__":
    main()
