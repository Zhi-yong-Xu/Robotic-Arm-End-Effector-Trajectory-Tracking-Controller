"""YAML 配置加载与路径解析。"""
import os
import yaml


def load_config(path: str) -> dict:
    """加载 YAML 并将相对模型路径解析为项目根目录下的绝对路径。"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(path))
    mp = cfg["simulation"]["model_path"]
    if not os.path.isabs(mp):
        cfg["simulation"]["model_path"] = os.path.join(root, mp)
    return cfg
