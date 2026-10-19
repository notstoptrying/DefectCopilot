"""Agent 工具集与 JSON Schema。

每个工具返回统一信封：{"tool","ok","data","summary"}，Agent 与前端都按
同一结构消费；schema 同时喂给真模型做 function calling。
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..config import CASE_CSV, TICKET_DB, TOP_K
from ..rag.index import SearchHit, get_index
from ..rag.rewrite import expand_query
from ..vision.detector import get_detector

# ---------- 工具 schema（OpenAI function calling 格式） ----------
TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在热轧钢板表面质量知识库中检索条款。知识库包含：六类表面缺陷判定分级标准、工艺参数控制规范、根因排查对照表、处置与返修流程、设备点检手册、工单填写规范、客户投诉与8D流程、历史事故复盘。任何关于判定标准、参数范围、流程步骤的问题都应先调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题，尽量保留用户原话中的关键术语"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 4"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_case_db",
            "description": "查询历史质量工单数据库（2026-05 至 2026-10，共 34 条）。可按缺陷类型、产线、等级过滤，返回统计与明细。当用户问到历史情况、发生频率、根因分布、损失金额时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_type": {"type": "string", "description": "缺陷类型：划痕/裂纹/夹杂/麻点/氧化皮压入/斑块"},
                    "line": {"type": "string", "description": "产线：1420 或 1780"},
                    "grade": {"type": "string", "description": "等级：A/B/C"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_defect_image",
            "description": "对上传的钢板表面图像做缺陷分割与分级。返回缺陷类别、面积占比、判定等级、检测框与标注图。用户上传图片后必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "上传接口返回的图片 ID"},
                },
                "required": ["image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "根据判定结论生成质量工单并落库。仅在用户明确要求生成工单、或图像判定完成且等级为 B/C 时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "coil_no": {"type": "string", "description": "卷号"},
                    "steel": {"type": "string", "description": "钢种，如 SPHC"},
                    "defect_type": {"type": "string"},
                    "grade": {"type": "string", "description": "A/B/C"},
                    "area_ratio": {"type": "number"},
                    "cause": {"type": "string", "description": "疑似根因"},
                    "disposal": {"type": "string", "description": "处置方式：修磨/切除/降级/判废"},
                    "line": {"type": "string", "description": "产线：1420 或 1780，默认 1420"},
                },
                "required": ["defect_type", "grade"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS_SCHEMA]


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict
    summary: str
    latency_ms: int

    def envelope(self) -> str:
        return json.dumps(
            {"tool": self.name, "ok": self.ok, "data": self.data, "summary": self.summary},
            ensure_ascii=False,
        )


# ---------- 各工具实现 ----------
def search_knowledge(query: str, top_k: int = TOP_K) -> ToolResult:
    t0 = time.time()
    index = get_index()
    expanded, rewritten = expand_query(query)
    hits: list[SearchHit] = index.search(expanded, top_k=max(1, min(top_k, 8)))
    data_hits = []
    for i, h in enumerate(hits, 1):
        data_hits.append(
            {
                "idx": i,
                "cid": h.chunk.cid,
                "doc": h.chunk.doc_id,
                "doc_title": h.chunk.doc_title,
                "heading": h.chunk.heading,
                "text": h.chunk.text[:420],
                "score": round(h.score, 4),
                "cosine": round(h.cosine, 4),
            }
        )
    data = {
        "query": query,
        "rewritten_terms": rewritten,
        "hits": data_hits,
        "max_cosine": round(max((h.cosine for h in hits), default=0.0), 4),
    }
    summary = (
        f"检索到 {len(data_hits)} 条相关条款"
        + (f"（最高相似度 {data['max_cosine']}）" if data_hits else "（无命中）")
    )
    return ToolResult("search_knowledge", True, data, summary, int((time.time() - t0) * 1000))


def query_case_db(defect_type: str = "", line: str = "", grade: str = "") -> ToolResult:
    t0 = time.time()
    rows: list[dict] = []
    with open(CASE_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if defect_type and r["缺陷类型"] != defect_type:
                continue
            if line and r["产线"] != line:
                continue
            if grade and r["等级"] != grade:
                continue
            rows.append(r)
    stats = {
        "count": len(rows),
        "avg_area": round(sum(float(r["面积占比"].rstrip("%")) for r in rows) / len(rows), 2) if rows else None,
        "total_loss": round(sum(float(r["损失万元"]) for r in rows), 1) if rows else 0,
        "top_causes": [(k, v) for k, v in Counter(r["疑似根因"] for r in rows).most_common(3)],
        "disposal": dict(Counter(r["处置方式"] for r in rows)),
        "rows": [
            {
                "date": r["日期"], "line": r["产线"], "steel": r["钢种"],
                "defect": r["缺陷类型"], "grade": r["等级"], "area": r["面积占比"],
                "cause": r["疑似根因"], "disposal": r["处置方式"], "loss": r["损失万元"],
            }
            for r in rows[:8]
        ],
    }
    summary = f"命中 {len(rows)} 条历史工单" + (f"，主要根因：{stats['top_causes'][0][0]}" if stats["top_causes"] else "")
    return ToolResult("query_case_db", True, stats, summary, int((time.time() - t0) * 1000))


_IMAGE_STORE: dict[str, Path] = {}


def register_image(path: Path) -> str:
    image_id = uuid.uuid4().hex[:10]
    _IMAGE_STORE[image_id] = path
    return image_id


def analyze_defect_image(image_id: str) -> ToolResult:
    t0 = time.time()
    path = _IMAGE_STORE.get(image_id)
    if path is None or not path.exists():
        return ToolResult("analyze_defect_image", False, {"image_id": image_id}, "图片不存在或已过期", 0)
    from PIL import Image

    det = get_detector().detect(Image.open(path), save_id=image_id)
    return ToolResult("analyze_defect_image", True, det.to_dict(), f"检出 {det.defect_type}，面积占比 {det.area_ratio:.2%}，建议 {det.level} 级", int((time.time() - t0) * 1000))


def create_ticket(coil_no: str = "", steel: str = "", defect_type: str = "未知", grade: str = "B", area_ratio: float = 0.0, cause: str = "待排查", disposal: str = "待定", line: str = "1420") -> ToolResult:
    t0 = time.time()
    TICKET_DB.parent.mkdir(parents=True, exist_ok=True)
    line = line if line in ("1420", "1780") else "1420"
    ticket_no = f"QT-L{line}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:3].upper()}"
    record = {
        "工单编号": ticket_no,
        "创建时间": datetime.now().isoformat(timespec="seconds"),
        "产线": f"L{line}",
        "卷号": coil_no or "未填写",
        "钢种": steel or "未填写",
        "缺陷类型": defect_type,
        "等级": grade,
        "面积占比": f"{area_ratio:.2%}",
        "疑似根因": cause,
        "处置方式": disposal,
        "判定人": "DefectCopilot(辅助判定)",
        "状态": "待质检员确认",
    }
    with open(TICKET_DB, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return ToolResult("create_ticket", True, record, f"已生成工单 {ticket_no}", int((time.time() - t0) * 1000))


class ToolBox:
    def __init__(self):
        self._impls: dict[str, Callable] = {
            "search_knowledge": search_knowledge,
            "query_case_db": query_case_db,
            "analyze_defect_image": analyze_defect_image,
            "create_ticket": create_ticket,
        }

    def execute(self, name: str, args: dict) -> ToolResult:
        fn = self._impls.get(name)
        if fn is None:
            return ToolResult(name, False, {}, f"未知工具 {name}", 0)
        try:
            return fn(**args)
        except TypeError:
            # 参数不匹配时降级为无参调用，避免一次失败中断整轮
            try:
                return fn()
            except Exception as exc:
                return ToolResult(name, False, {"error": str(exc)}, f"工具执行失败：{exc}", 0)
        except Exception as exc:
            return ToolResult(name, False, {"error": str(exc)}, f"工具执行失败：{exc}", 0)
