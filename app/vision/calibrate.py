"""局部视场等级阈值的标定脚本。

为什么需要标定：
    《表面缺陷判定标准》里的阈值（如划痕 1.5%）是**整卷板面口径**，
    而分割模型输出的是 200×200 局部视场内的面积占比。整卷宽 1.2 m、长数百米，
    单张图只覆盖其中极小一块，直接套用整卷阈值会让几乎每张图都越线判 C
    （实测 840 张测试图里 835 张判 C，等级完全失去区分度）。

    单张图在物理上给不出整卷占比——整卷占比需要全卷所有检测图累计。
    所以本项目把「等级」明确为**局部视场严重度初判**，其分档点由真实样本分布标定：
    取各缺陷类型面积占比的 P35 / P72 作为 A|B、B|C 分界，
    保证三档在产线数据上都落得到样本；同时保持类型间的严格度顺序
    （裂纹 < 夹杂 < 划痕 < 斑块）。

    这是「标定值」，不是「标准值」。拿到质检员的历史定级标注后，
    应改用监督校准（按标注拟合分界），脚本留了 --labels 扩展位。

用法：
    python -m app.vision.calibrate                    # 用测试集标定并写 calibration.json
    python -m app.vision.calibrate --split training   # 换数据划分
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import DATASET_DIR
from .detector import CLASS_NAMES, TYPE_MAPPING, get_detector

NEU_ROOT = DATASET_DIR
OUT_FILE = Path(__file__).resolve().parent / "calibration.json"

# 类型间的严格度顺序：数值越小越严（裂纹最严，斑块最宽）
STRICTNESS = ["裂纹", "夹杂", "划痕", "麻点", "氧化皮压入", "斑块"]
A_Q, B_Q = 0.35, 0.72


def collect(split: str, limit: int | None = None) -> dict[str, list[float]]:
    from PIL import Image

    if not (NEU_ROOT / "images" / split).exists():
        raise SystemExit(
            f"找不到数据集：{NEU_ROOT / 'images' / split}\n"
            f"请下载 NEU-Seg 后设置环境变量 NEU_SEG_DIR（示例见 .env.example），"
            f"或放到 {DATASET_DIR}"
        )
    files = sorted((NEU_ROOT / "images" / split).glob("*.jpg"))
    if limit:
        files = files[:limit]
    det = get_detector()
    buckets: dict[str, list[float]] = defaultdict(list)
    for f in files:
        d = det.detect(Image.open(f), save_id="calib")
        kb_type = TYPE_MAPPING.get(d.class_id)
        if kb_type:
            buckets[kb_type].append(d.area_ratio)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["training", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    buckets = collect(args.split, args.limit)
    if not buckets:
        raise SystemExit("未采集到样本，请检查数据集路径")

    thresholds: dict[str, list[float]] = {}
    stats: dict[str, dict] = {}
    for kb_type, vals in buckets.items():
        arr = np.array(vals, dtype=np.float64)
        a, b = float(np.quantile(arr, A_Q)), float(np.quantile(arr, B_Q))
        thresholds[kb_type] = [round(a, 4), round(b, 4)]
        stats[kb_type] = {
            "n": int(arr.size),
            "p35": round(a, 4),
            "p72": round(b, 4),
            "p50": round(float(np.median(arr)), 4),
            "max": round(float(arr.max()), 4),
        }

    # 保持严格度顺序：按 STRICTNESS 排序后，阈值应单调不降
    ordered = [t for t in STRICTNESS if t in thresholds]
    for prev, cur in zip(ordered, ordered[1:]):
        if thresholds[cur][0] < thresholds[prev][0]:
            thresholds[cur][0] = thresholds[prev][0]
        if thresholds[cur][1] < thresholds[prev][1]:
            thresholds[cur][1] = thresholds[prev][1]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "局部视场（单张图内缺陷面积占比），非整卷口径",
        "source": f"NEU-Seg {args.split} split",
        "quantiles": {"a_b": A_Q, "b_c": B_Q},
        "thresholds": thresholds,
        "stats": stats,
        "note": "标定值非标准值；整卷等级需按全卷累计面积判定",
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")
    for t in ordered:
        a, b = thresholds[t]
        print(f"  {t:6s} A≤{a:.2%}  B≤{b:.2%}  C>{b:.2%}   (n={stats[t]['n']}, 中位 {stats[t]['p50']:.2%})")


if __name__ == "__main__":
    main()
