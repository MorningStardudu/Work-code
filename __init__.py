"""
SAN 融合层消融实验包
导入此包会自动注册 VOC 消融数据集
"""
import importlib.util
import os

# 加载数据集注册模块 (文件名含数字前缀, 不能用标准import)
_reg_path = os.path.join(os.path.dirname(__file__), "register_datasets.py")
_reg_spec = importlib.util.spec_from_file_location("ablation_register_dataset", _reg_path)
_reg_mod = importlib.util.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg_mod)
