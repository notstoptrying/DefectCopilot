"""离线规则模型：无 API Key 时的演示通道。

它不是「假模型」——它和真模型走完全相同的 Agent 循环（同样的 tool_calls
协议、同样的 tool 消息回填），只是决策由确定性策略产生、回答由检索条款
抽取组织而成。好处是：零成本、可复现、结果稳定，且因为不做自由生成，
不会编造数字——这对质检场景恰好是刚需。

策略是一个三步状态机：
  1. 有图 → 先调图像分析（此时还不知道缺陷类型，不该盲目检索）
  2. 拿到类型 → 用「用户问题 + 类型」再检索规程（两段式，避免检索跑偏）
  3. 证据齐了 → 组织回答
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from ..agent.compose import compose
from ..config import MIN_EVIDENCE_SCORE
from ..rag.text import detect_defect_types, has_domain_terms
from .base import LLMClient, LLMEvent, ToolCall

_STAT_HINTS = ("历史", "统计", "多少", "几起", "占比", "损失", "频率", "工单", "发生过", "上次", "趋势")


def _parse(messages: list[dict]) -> tuple[str, str | None, list[dict]]:
    # 当前问题必须取「最后一条」user 消息：多轮对话时 messages 前面是上一轮的
    # user/assistant，取第一条会把旧问题当成新问题——实测带图追问时，
    # 检索与工单统计全部跑到上一轮的问题上（截图流程中暴露，单轮评测抓不到）。
    question = ""
    image_id = None
    tool_results: list[dict] = []
    for m in reversed(messages):
        if m.get("role") == "user" and not question:
            question = m.get("content") or ""
        if m.get("role") == "system" and "image_id=" in (m.get("content") or ""):
            found = re.search(r"image_id=([0-9a-zA-Z]+)", m["content"])
            if found:
                image_id = found.group(1)
        if m.get("role") == "tool":
            try:
                tool_results.append(json.loads(m.get("content") or "{}"))
            except json.JSONDecodeError:
                continue
    return question, image_id, tool_results


def _case_call(question: str) -> ToolCall | None:
    """「多少 / 历史 / 损失」这类词太泛，必须同时落在领域内才算查统计。"""
    if not any(h in question for h in _STAT_HINTS):
        return None
    if not (has_domain_terms(question) or detect_defect_types(question)):
        return None
    args: dict = {}
    types = detect_defect_types(question)
    if types:
        args["defect_type"] = types[0]
    if "1420" in question:
        args["line"] = "1420"
    elif "1780" in question:
        args["line"] = "1780"
    return ToolCall(id="c_case", name="query_case_db", arguments=args)


class MockClient(LLMClient):
    mode = "mock"
    name = "offline-rule-engine"

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[LLMEvent]:
        question, image_id, tool_results = _parse(messages)
        called = [r.get("tool") for r in tool_results if r.get("ok")]
        image = next((r["data"] for r in tool_results if r.get("tool") == "analyze_defect_image" and r.get("ok")), None)
        hits = next((r["data"].get("hits") or [] for r in tool_results if r.get("tool") == "search_knowledge" and r.get("ok")), [])
        cases = next((r["data"] for r in tool_results if r.get("tool") == "query_case_db" and r.get("ok")), None)

        # ---------- 第一步：有图先看图 ----------
        if image_id and "analyze_defect_image" not in called:
            yield LLMEvent(
                "tool_calls",
                tool_calls=[ToolCall(id="c_img", name="analyze_defect_image", arguments={"image_id": image_id})],
            )
            return

        # ---------- 第二步：按（问题 + 图像类型）检索规程 ----------
        if "search_knowledge" not in called:
            dtype = (image or {}).get("defect_type") or ""
            query = question
            if dtype and dtype != "无缺陷":
                # 图像给出的类型与等级是最强的检索信号，必须带进查询
                level = (image or {}).get("level") or ""
                query = f"{question} {dtype} {level}级 判定 处置"
            calls = [ToolCall(id="c_kb", name="search_knowledge", arguments={"query": query, "top_k": 4})]
            case = _case_call(question)
            if case:
                calls.append(case)
            yield LLMEvent("tool_calls", tool_calls=calls)
            return

        if cases is None:
            case = _case_call(question)
            if case:
                yield LLMEvent("tool_calls", tool_calls=[case])
                return

        # ---------- 第三步：组织回答 ----------
        max_cos = max((h.get("cosine", 0) for h in hits), default=0.0)
        weak_evidence = max_cos < MIN_EVIDENCE_SCORE
        off_domain = not has_domain_terms(question) and not image
        refused = not image and not (cases and cases.get("count")) and (weak_evidence or off_domain)
        text = compose(question, hits, image=image, cases=cases, refused=refused)

        step = 26
        for i in range(0, len(text), step):
            yield LLMEvent("delta", text=text[i : i + step])
