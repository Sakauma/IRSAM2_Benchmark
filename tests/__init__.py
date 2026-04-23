"""测试目录初始化。

Author: Egor Izmaylov

测试运行时需要确保 `src/` 在导入路径上，这样可以直接从仓库根目录运行 unittest。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# 在测试入口处补一层导入路径注入，避免依赖外部 PYTHONPATH。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
