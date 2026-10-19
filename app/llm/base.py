"""模型客户端统一接口。

两种实现共用同一套事件协议，Agent 循环不需要知道自己接的是真模型还是
离线规则模型——这是整个项目能「无密钥演示、有密钥上线」的关键。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMEvent:
    """流式事件：delta(增量文本) / tool_calls(工具调用) / usage / error"""

    kind: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    message: str = ""


class LLMClient:
    name: str = "base"
    mode: str = ""

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None):
        raise NotImplementedError

    def health(self) -> dict:
        return {"provider": self.mode, "model": self.name}
