"""IRSAM2_Benchmark 顶层启动入口。

Author: Egor Izmaylov

这个文件的职责很单纯：
1. 在直接执行仓库根目录下的 ``main.py`` 时，确保 ``src/`` 被加入导入路径。
2. 把控制权交给真正的 CLI 入口 ``irsam2_benchmark.cli.main``。

保留这个薄入口的意义是：
- 便于本地直接 ``python main.py ...`` 调试；
- 也便于在服务器或脚本环境中以仓库根目录为工作目录直接启动。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 这里固定取仓库根目录，保证从任何工作目录启动时都能定位到 src。
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# 只在尚未注入时添加，避免重复修改 sys.path。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irsam2_benchmark.cli import main


if __name__ == "__main__":
    # 顶层文件只负责把参数和控制权转发给真正的 CLI 实现。
    main()
