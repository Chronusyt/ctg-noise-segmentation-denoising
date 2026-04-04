"""
CTG_test 数据 IO 模块 - 自包含副本，不依赖 CTG 外部路径
"""
from .fetal_reader import read_fetal, FetalData

__all__ = ["read_fetal", "FetalData"]
