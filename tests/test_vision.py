"""图像分析链路测试。

样例图来自 NEU-Seg 公开数据集的测试集（tests/fixtures/sample_defect.jpg），
含 5 处条带状夹杂。没有 torch 或权重文件时整组跳过，不影响纯文本链路的测试。
"""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample_defect.jpg"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="缺少样例图")


@pytest.fixture(scope="module")
def detection():
    torch = pytest.importorskip("torch", reason="未安装 torch，跳过图像链路测试")
    from PIL import Image

    from app.vision.detector import get_detector

    det = get_detector()
    assert det is not None
    return det.detect(Image.open(SAMPLE), save_id="pytest_sample")


def test_detection_schema(detection):
    d = detection.to_dict()
    for key in ("model", "defect_type", "level", "area_ratio", "boxes", "annotated_url"):
        assert key in d, f"输出缺少字段 {key}"
    assert d["level"] in {"A", "B", "C"}
    assert 0.0 <= d["area_ratio"] <= 1.0


def test_detection_finds_defect(detection):
    """样例图确实含缺陷，模型不应判成无缺陷。"""
    assert detection.defect_type != "无缺陷"
    assert len(detection.boxes) >= 1
    for b in detection.boxes:
        assert b["x2"] > b["x1"] and b["y2"] > b["y1"], f"外接框坐标异常：{b}"
        assert b["level"] in {"A", "B", "C"}
        assert "class_id" in b, "外接框缺少类别，标注图无法配色"


def test_level_thresholds_are_calibrated(detection):
    """等级阈值应来自训练集标定，而不是拍脑袋的默认值。"""
    assert detection.level_source.startswith("calibrated"), f"阈值来源：{detection.level_source}"
    assert detection.a_max > 0 and detection.b_max > detection.a_max


def test_annotated_image_written(detection):
    path = Path(detection.annotated_url.replace("/files/", ""))
    full = Path(__file__).resolve().parent.parent / "app" / "data" / "uploads" / path.name
    assert full.exists() and full.stat().st_size > 1000, "标注图没有生成"


def test_detection_is_deterministic(detection):
    """同一张图两次推理，等级与框数应完全一致（评估可复现的前提）。"""
    from PIL import Image

    from app.vision.detector import get_detector

    again = get_detector().detect(Image.open(SAMPLE), save_id="pytest_sample2")
    assert again.level == detection.level
    assert len(again.boxes) == len(detection.boxes)
    assert abs(again.area_ratio - detection.area_ratio) < 1e-6
