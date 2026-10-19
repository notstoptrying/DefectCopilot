"""Agent 运行时：工具调用循环 + 事件流。

循环本身与模型实现解耦：只要模型按 `LLMEvent` 协议产出 delta / tool_calls，
无论背后是 DeepSeek 还是本地规则引擎，都走同一条路径。
"""
from __future__ import annotations

import time
from typing import Iterator

from ..config import LLM_API_KEY, LLM_PROVIDER, MAX_STEPS
from ..llm.base import LLMClient, ToolCall
from ..llm.mock import MockClient
from ..llm.openai_compat import OpenAICompatClient
from ..rag.index import get_index
from .tools import TOOLS_SCHEMA, ToolBox

SYSTEM_PROMPT = """你是冷轧/热轧钢板表面质量领域的质检助手，服务于产线质检员与工艺工程师。

工作规则：
1. 涉及判定标准、参数范围、流程步骤的问题，必须先调用 search_knowledge 检索，不得凭记忆回答。
2. 回答结构：先给结论，再给判定依据（用 [1][2] 标注引用的条款序号），然后给可能根因与处置建议。
3. 用户上传图片时，先调用 analyze_defect_image，再结合检索到的判定标准给出等级建议。
4. 问到历史情况、发生频率、损失金额时，调用 query_case_db 取真实统计。
5. 知识库中没有依据时，直接说明「知识库未收录」，不要推测，不要编造数字。
6. 只回答与钢板表面质量、工艺、设备、质量流程相关的问题，超出范围时礼貌拒答。
7. 结论用简体中文，条理清晰，控制在 400 字以内，不要复述整段原文。"""


def build_client() -> LLMClient:
    if LLM_PROVIDER == "openai" and LLM_API_KEY:
        return OpenAICompatClient()
    return MockClient()


def _assistant_msg(calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": __import__("json").dumps(c.arguments, ensure_ascii=False)},
            }
            for c in calls
        ],
    }


class AgentRuntime:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or build_client()
        self.toolbox = ToolBox()

    def run(self, question: str, image_id: str | None = None, history: list[dict] | None = None) -> Iterator[dict]:
        t0 = time.time()
        system = SYSTEM_PROMPT
        if image_id:
            system += f"\n\n当前会话已上传一张缺陷图像，image_id={image_id}，请先分析图像。"
        messages: list[dict] = [{"role": "system", "content": system}]
        for h in (history or [])[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question or "请分析这张图片"})

        steps = 0
        citations: list[dict] = []
        tool_count = 0

        while steps < MAX_STEPS:
            steps += 1
            calls: list[ToolCall] = []
            for ev in self.client.stream_chat(messages, TOOLS_SCHEMA):
                if ev.kind == "delta":
                    yield {"type": "token", "text": ev.text}
                elif ev.kind == "tool_calls":
                    calls = ev.tool_calls
                elif ev.kind == "usage":
                    yield {"type": "usage", "usage": ev.usage}
                elif ev.kind == "error":
                    yield {"type": "error", "message": ev.message}
                    return

            if not calls:
                break

            messages.append(_assistant_msg(calls))
            for c in calls:
                tool_count += 1
                yield {"type": "tool_call", "id": c.id, "name": c.name, "args": c.arguments, "step": steps}
                result = self.toolbox.execute(c.name, c.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "name": c.name,
                        "content": result.envelope(),
                    }
                )
                brief = {k: v for k, v in result.data.items() if k in ("defect_type", "area_ratio", "level", "count", "max_cosine", "rewritten_terms")}
                yield {
                    "type": "tool_result",
                    "id": c.id,
                    "name": c.name,
                    "ok": result.ok,
                    "summary": result.summary,
                    "latency_ms": result.latency_ms,
                    "brief": brief,
                }
                if c.name == "search_knowledge" and result.ok:
                    citations = result.data.get("hits") or []
                if c.name == "analyze_defect_image" and result.ok:
                    yield {"type": "image", "data": result.data}

        if citations:
            yield {"type": "citations", "items": citations}

        index_stat = get_index().stats()
        yield {
            "type": "done",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "steps": steps,
            "tools": tool_count,
            "provider": self.client.mode,
            "index": index_stat,
        }
