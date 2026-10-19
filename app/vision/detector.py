"""缺陷检测器：UNet 权重优先，无权重时回退到轻量规则分割。

设计要点：
- 分割模型只负责「定位 + 面积」，等级判定和类型映射交给业务规则层，
  与《表面缺陷判定标准》的 A/B/C 阈值一一对应，可解释、可审计；
- 没有训练权重时也能完整跑通链路（规则分割兜底），这是演示可用性的关键。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import UPLOAD_DIR
from .unet import UNet

WEIGHTS = Path(__file__).resolve().parent / "weights" / "unet_steel.pt"
CALIB_FILE = Path(__file__).resolve().parent / "calibration.json"

CLASS_NAMES = {1: "条带状夹杂", 2: "斑块状缺陷", 3: "线状划痕"}
# 每类一个颜色。选色原则：在灰白钢板上都要够醒目，且三色之间在灰度上也有差异，
# 打印成黑白时不至于糊成一片。
CLASS_COLORS = {
    1: (230, 57, 53),    # 条带状夹杂 —— 红
    2: (245, 158, 11),   # 斑块状缺陷 —— 橙
    3: (56, 132, 255),   # 线状划痕 —— 蓝
}
RULE_COLOR = (230, 57, 53)
# 形态类别 -> 判定标准中的标准缺陷类型
TYPE_MAPPING = {1: "夹杂", 2: "斑块", 3: "划痕"}
# 等级口径说明：
#   标准 Q/HM-ZL-2025-01 的阈值（如划痕 1.5%）是「整卷板面占比」，
#   而模型算的是「200×200 局部视场占比」，两者量纲不同。
#   直接套用会让 99% 的图判 C（实测 835/840），等级失去区分度。
#   因此这里的分档点是**局部视场口径的标定值**，由 app/vision/calibrate.py
#   用真实样本分布生成（见 calibration.json）；下面的常量只在无标定文件时兜底。
LEVEL_RULES = {
    "划痕": (0.040, 0.120),
    "夹杂": (0.030, 0.100),
    "斑块": (0.060, 0.180),
    "麻点": (0.040, 0.120),
    "氧化皮压入": (0.030, 0.120),
    "裂纹": (0.010, 0.050),
}
_LEVEL_CACHE: tuple[dict[str, tuple[float, float]], str] | None = None


def level_thresholds() -> tuple[dict[str, tuple[float, float]], str]:
    """返回 (阈值表, 来源)。有标定文件时优先用标定值。"""
    global _LEVEL_CACHE
    if _LEVEL_CACHE is None:
        table = dict(LEVEL_RULES)
        source = "default"
        if CALIB_FILE.exists():
            try:
                data = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
                for k, v in (data.get("thresholds") or {}).items():
                    table[k] = (float(v[0]), float(v[1]))
                source = f"calibrated@{data.get('generated_at', '')}"
            except Exception:
                pass
        _LEVEL_CACHE = (table, source)
    return _LEVEL_CACHE


def judge_level(defect_type: str, area_ratio: float) -> str:
    table, _ = level_thresholds()
    a_max, b_max = table.get(defect_type, (0.040, 0.120))
    if area_ratio <= a_max:
        return "A"
    if area_ratio <= b_max:
        return "B"
    return "C"


def label_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """四连通域标记，返回 [x1,y1,x2,y2,area]。纯 numpy 实现，无 scipy 依赖。"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    regions = []
    ys, xs = np.where(mask > 0)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0]:
            continue
        current += 1
        stack = [(y0, x0)]
        labels[y0, x0] = current
        area = 0
        x1 = x2 = x0
        y1 = y2 = y0
        while stack:
            y, x = stack.pop()
            area += 1
            x1, x2 = min(x1, x), max(x2, x)
            y1, y2 = min(y1, y), max(y2, y)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        regions.append((int(x1), int(y1), int(x2), int(y2), int(area)))
    return regions


@dataclass
class Detection:
    model: str
    defect_type: str
    class_id: int
    area_ratio: float
    level: str
    confidence: float
    boxes: list[dict] = field(default_factory=list)
    annotated_url: str = ""
    max_region_ratio: float = 0.0
    level_scope: str = "局部视场"
    level_source: str = "default"
    a_max: float = 0.0
    b_max: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "defect_type": self.defect_type,
            "class_id": self.class_id,
            "area_ratio": round(self.area_ratio, 4),
            "max_region_ratio": round(self.max_region_ratio, 4),
            "level": self.level,
            "level_scope": self.level_scope,
            "level_source": self.level_source,
            "level_rule": {"a_max": round(self.a_max, 4), "b_max": round(self.b_max, 4)},
            "confidence": round(self.confidence, 4),
            "boxes": self.boxes,
            "annotated_url": self.annotated_url,
        }


def _edge(mask: np.ndarray) -> np.ndarray:
    """取二值掩码的内边界（像素本身在前景、且四邻域中存在背景）。

    没有 scipy 依赖，用填充 + 四向与运算实现形态学腐蚀，
    边界减腐蚀结果即可得到 1px 轮廓。
    """
    m = mask.astype(bool)
    pad = np.pad(m, 1, constant_values=False)
    nb = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    interior = pad[1:-1, 1:-1] & nb
    return m & ~interior


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    """两个矩形是否相交（留 gap 像素间隙，避免标签贴在一起显得糊）。"""
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _load_font(size: int):
    """按优先级加载中文字体；PIL 内置字体不含 CJK，画中文会变方块。"""
    from PIL import ImageFont

    for path in (
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except Exception:
        return None


def _annotate(
    img: Image.Image,
    boxes: list[dict],
    save_id: str,
    label_map: np.ndarray | None = None,
    names: dict[int, str] | None = None,
    colors: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.40,
) -> str:
    """按缺陷类别着色标注：填充 + 轮廓 + 同色外接框 + 底部图例。

    label_map 是网络输出分辨率（200×200）的类别图，值为 0=背景、1..3=缺陷类别。
    逐类单独放大再取阈值，避免「先放大类别图再取整」在边界处串色；
    颜色与标签同源（同一个 class_id），所以看图例就能对上图上每一块。
    """
    names = names or CLASS_NAMES
    colors = colors or CLASS_COLORS
    # 显示尺寸：长边压到 1000px。手机原图 4000×3000 若直接标注，
    # float32 画布就要 ~140 MB，且字号/图例按宽度算会放大到 100px 级别。
    w0, h0 = img.size
    scale = min(1.0, 1000.0 / max(w0, h0))
    if scale < 1.0:
        img = img.resize((round(w0 * scale), round(h0 * scale)), Image.BILINEAR)
    out = img.convert("RGB").copy()
    w, h = out.size
    canvas = np.asarray(out, dtype=np.float32)

    # label_map 和外接框都在 200×200 模型视场坐标系里；彩色蒙版经 Image.resize
    # 自动放大到画布尺寸，框坐标必须按同一比例映射，否则框和色块会错位
    # （实测 800×800 图：色块在右下角，框却画在左上角）。
    gh, gw = (label_map.shape if label_map is not None else (200, 200))[:2]
    sx, sy = w / gw, h / gh

    present: list[int] = []
    if label_map is not None and label_map.any():
        for cid in sorted(set(np.unique(label_map).tolist()) - {0}):
            binm = (label_map == cid).astype(np.uint8) * 255
            m = np.asarray(Image.fromarray(binm).resize((w, h), Image.BILINEAR)) > 100
            if not m.any():
                continue
            color = np.array(colors.get(cid, (200, 200, 200)), dtype=np.float32)
            fg = canvas[m]
            canvas[m] = fg * (1 - alpha) + color * alpha
            canvas[_edge(m)] = color * 0.75      # 轮廓压暗一档，边界更锐
            present.append(cid)

    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    stroke = max(2, round(w / 110))
    # 标签只放「序号 + 等级」，字号压到 9px 左右。
    # 之前写「条带状夹杂 A 级」11px，在 200px 视场里宽度接近半幅，几处框一挤就互相覆盖、
    # 分不清哪张标签属于哪个框——类别交给颜色和下方图例，等级保留在图上，信息一点没少。
    font = _load_font(max(9, round(w / 24)))
    placed: list[tuple[int, int, int, int]] = []
    legend: list[tuple[int, int, str, float]] = []

    # 按空间顺序编号（左上 → 右下），图例按同一顺序排，看图不会串号
    ordered = sorted(boxes, key=lambda b: (b["y1"], b["x1"]))
    for n, b in enumerate(ordered):
        cid = b.get("class_id", 0)
        color = colors.get(cid, (210, 210, 210))
        x1, y1 = round(b["x1"] * sx), round(b["y1"] * sy)
        x2, y2 = round(b["x2"] * sx), round(b["y2"] * sy)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

        ratio = b.get("ratio")
        if ratio is None:
            ratio = (b.get("area") or 0) / (200 * 200)
        legend.append((n + 1, cid, b.get("level", "?"), float(ratio)))

        text = f"{n + 1} {b.get('level', '?')}"
        if font is None:
            draw.text((x1 + 1, max(0, y1 - 10)), text, fill=(255, 255, 255),
                      stroke_width=1, stroke_fill=color)
            continue

        tb = draw.textbbox((0, 0), text, font=font)
        bw, bh = tb[2] + 5, tb[3] + 3
        # 候选位置依次：框上方、框下方、框内左上、框内右上、再往上叠一行
        spots = [
            (x1, y1 - bh - 1),
            (x1, y2 + 1),
            (x1 + 1, y1 + 1),
            (x2 - bw - 1, y1 + 1),
            (x1, y1 - 2 * bh - 2),
        ]
        chosen = None
        for cx, cy in spots:
            tx = max(0, min(cx, w - bw))
            ty = max(0, min(cy, h - bh))
            rect = (tx, ty, tx + bw, ty + bh)
            if not any(_overlap(rect, p) for p in placed):
                chosen = rect
                break
        if chosen is None:                     # 全被占，退回框上方并对齐钳制
            tx = max(0, min(x1, w - bw))
            ty = max(0, min(y1 - bh - 1, h - bh))
            chosen = (tx, ty, tx + bw, ty + bh)
        placed.append(chosen)
        tx, ty = chosen[0], chosen[1]
        draw.rectangle([chosen[0], chosen[1], chosen[2], chosen[3]], fill=color)
        draw.text((tx + 2, ty + 1), text, fill=(255, 255, 255), font=font)

    # ---- 图例：每处缺陷一行，序号与图上徽章一一对应 ----
    if legend:
        row_h = max(15, round(w * 0.062))
        pad = max(5, round(w * 0.022))
        strip = Image.new("RGB", (w, pad * 2 + row_h * len(legend)), (17, 24, 39))
        sd = ImageDraw.Draw(strip)
        lf = _load_font(max(9, round(w / 26)))
        sw = max(7, round(w * 0.024))
        for k, (num, cid, level, ratio) in enumerate(legend):
            y = pad + k * row_h + (row_h - sw) // 2
            sd.rectangle([pad, y, pad + sw, y + sw], fill=colors.get(cid, (200, 200, 200)))
            txt = f"{num}  {names.get(cid, f'类别{cid}')}  {level} 级  {ratio:.1%}"
            tx = pad + sw + max(4, round(w * 0.012))
            if lf is not None:
                sd.text((tx, y + sw / 2), txt, fill=(235, 238, 245), font=lf, anchor="lm")
            else:
                sd.text((tx, y), txt, fill=(235, 238, 245))
        merged = Image.new("RGB", (w, h + strip.height), (17, 24, 39))
        merged.paste(out, (0, 0))
        merged.paste(strip, (0, h))
        out = merged

    path = UPLOAD_DIR / f"annotated_{save_id}.png"
    out.save(path)
    return f"/files/{path.name}"


class UNetDetector:
    """加载训练好的 UNet 权重做多类分割。"""

    name = "UNet(NEU-Seg)"

    def __init__(self):
        import torch  # 延迟导入，无 GPU 环境也不影响纯文本问答

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(WEIGHTS, map_location=self.device, weights_only=False)
        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.model = UNet(in_ch=3, out_ch=ckpt.get("num_classes", 4))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.epoch = ckpt.get("epoch")

    @property
    def available(self) -> bool:
        return WEIGHTS.exists()

    def detect(self, img: Image.Image, save_id: str) -> Detection:
        torch = self.torch
        x = img.convert("RGB").resize((200, 200))
        arr = np.asarray(x, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()  # (C,H,W)
        pred = probs.argmax(0)

        best: Detection | None = None
        all_boxes: list[dict] = []
        for c in (1, 2, 3):
            mask = (pred == c).astype(np.uint8)
            if mask.sum() == 0:
                continue
            area_ratio = float(mask.mean())
            regions = [r for r in label_components(mask) if r[4] >= 12]
            regions.sort(key=lambda r: -r[4])
            conf = float(np.mean([probs[c][mask > 0]])) if mask.sum() else 0.0
            cname = self.class_names.get(c, f"class{c}")
            kb_type = TYPE_MAPPING.get(c, "划痕")
            for x1, y1, x2, y2, area in regions[:5]:
                all_boxes.append(
                    {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area": area,
                        "ratio": round(area / (200 * 200), 4),
                        "class_id": c,
                        "label": cname,
                        "level": judge_level(kb_type, area / (200 * 200)),
                    }
                )
            level = judge_level(kb_type, area_ratio)
            det = Detection(
                model=f"{self.name}(ep{self.epoch})",
                defect_type=cname,
                class_id=c,
                area_ratio=area_ratio,
                level=level,
                confidence=conf,
                annotated_url="",
            )
            if best is None or area_ratio > best.area_ratio:
                best = det
        if best is None:
            return Detection(
                model=f"{self.name}(ep{self.epoch})",
                defect_type="无缺陷",
                class_id=0,
                area_ratio=0.0,
                level="A",
                confidence=0.0,
                annotated_url="",
            )
        best.boxes = all_boxes[:8]
        best.max_region_ratio = max((b.get("ratio", 0) for b in best.boxes), default=0.0)
        table, best.level_source = level_thresholds()
        best.a_max, best.b_max = table.get(TYPE_MAPPING.get(best.class_id, "划痕"), (0.04, 0.12))
        # 标红用完整类别图（三类缺陷都算前景），保证回答里说的面积和图上红色区域一致
        best.annotated_url = _annotate(
            img, best.boxes, save_id, label_map=pred, names=self.class_names, colors=CLASS_COLORS
        )
        return best


class RuleDetector:
    """无权重时的兜底：灰度阈值 + 连通域。保证链路永远可跑。"""

    name = "规则分割(兜底)"

    def detect(self, img: Image.Image, save_id: str) -> Detection:
        gray = np.asarray(img.convert("L").resize((200, 200)), dtype=np.float32)
        blur = Image.fromarray(gray.astype(np.uint8)).filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(1))
        gray = np.asarray(blur, dtype=np.float32)
        thr = gray.mean() - 1.2 * gray.std()
        mask = (gray < thr).astype(np.uint8)
        regions = [r for r in label_components(mask) if r[4] >= 20]
        area_ratio = float(sum(r[4] for r in regions) / (200 * 200))
        boxes = [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": a, "class_id": 1, "label": "疑似缺陷", "level": "?"}
            for x1, y1, x2, y2, a in regions[:6]
        ]
        level = judge_level("划痕", area_ratio) if regions else "A"
        det = Detection(
            model=self.name,
            defect_type="疑似缺陷（未分类）",
            class_id=-1,
            area_ratio=area_ratio,
            level=level,
            confidence=0.5,
            boxes=boxes,
            max_region_ratio=max((b["area"] for b in boxes), default=0) / (200 * 200),
            level_source=level_thresholds()[1],
            a_max=level_thresholds()[0]["划痕"][0],
            b_max=level_thresholds()[0]["划痕"][1],
        )
        det.annotated_url = (
            _annotate(img, boxes, save_id, label_map=mask,
                      names={1: "疑似缺陷"}, colors={1: RULE_COLOR})
            if boxes else ""
        )
        return det


def get_detector() -> UNetDetector | RuleDetector:
    """单例：权重 30 MB，每次重建要 ~140 ms 且占显存，进程内只加载一次。"""
    global _DETECTOR
    if _DETECTOR is None:
        if WEIGHTS.exists():
            try:
                _DETECTOR = UNetDetector()
            except Exception:
                _DETECTOR = RuleDetector()
        else:
            _DETECTOR = RuleDetector()
    return _DETECTOR


_DETECTOR: UNetDetector | RuleDetector | None = None
