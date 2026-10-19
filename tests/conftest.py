"""pytest 公共夹具。

把仓库根加进 sys.path，这样测试可以直接 import app.*，无需先安装包。
评测集只跑一次（会话级），其余用例复用结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def eval_result():
    """跑一遍完整 gold 集，返回 (summary, rows)。约 0.2 秒。"""
    from eval.run_eval import evaluate

    return evaluate()
