"""Benchmark 平台入口。

这个文件只做一件事：读取环境变量配置，并把控制权交给统一的实验调度器。
之所以保持极简，是为了让运行入口在单卡、多卡、服务器脚本中都保持一致。
"""

from experiment_core import load_config, run_experiment


if __name__ == "__main__":
    # 入口层不掺杂任何业务逻辑，便于脚本直接复用同一启动方式。
    run_experiment(load_config())
