"""IRSAM2_Benchmark 包入口。

Author: Egor Izmaylov

这里只暴露最常用的配置加载接口，方便外部脚本和测试统一导入。
"""

from .config import AppConfig, load_app_config

__all__ = ["AppConfig", "load_app_config"]
