"""OpenAI 兼容接口客户端（DeepSeek / 通义 / 智谱 / Moonshot / Ollama 等）。

只依赖 httpx，不装 openai SDK：少一层依赖，也避免版本兼容坑。
支持流式增量解析，工具调用参数按 index 累积分片拼接。
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
from .base import LLMClient, LLMEvent, ToolCall


class OpenAICompatClient(LLMClient):
    mode = "openai"

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or LLM_API_KEY
        self.name = model or LLM_MODEL

    def _payload(self, messages: list[dict], tools: list[dict] | None) -> dict:
        payload: dict = {
            "model": self.name,
            "messages": messages,
            "stream": True,
            "temperature": LLM_TEMPERATURE,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[LLMEvent]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if not self.api_key:
            yield LLMEvent("error", message="未配置 LLM_API_KEY，请改用离线模式或填入密钥")
            return

        acc: dict[int, dict] = {}
        usage: dict = {}
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=self._payload(messages, tools),
                timeout=LLM_TIMEOUT,
            ) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "ignore")[:400]
                    yield LLMEvent("error", message=f"HTTP {resp.status_code}: {body}")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        yield LLMEvent("delta", text=delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = acc.setdefault(
                            tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
        except httpx.HTTPError as exc:
            yield LLMEvent("error", message=f"请求失败：{exc}")
            return

        calls: list[ToolCall] = []
        for _, slot in sorted(acc.items()):
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=slot["id"] or f"call_{slot['name']}", name=slot["name"], arguments=args))

        if calls:
            yield LLMEvent("tool_calls", tool_calls=calls)
        if usage:
            yield LLMEvent("usage", usage=usage)

    def health(self) -> dict:
        return {
            "provider": self.mode,
            "model": self.name,
            "base_url": self.base_url,
            "key_configured": bool(self.api_key),
        }
