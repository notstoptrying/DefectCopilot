"""「生成工单」链路回归。

背景：README 承诺 4 个工具，但离线通道从未触发 create_ticket——
用户说「生成工单」时只会得到一段拒答式回答，第 4 个工具在演示里是死的。
这组测试锁住：显式要求才建单、不显式要求不建单、带图建单字段来自图像判定。
"""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample_defect.jpg"


def _run(question: str, **kw) -> list[dict]:
    from app.agent.runtime import AgentRuntime

    return list(AgentRuntime().run(question, **kw))


def _tool_names(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e["type"] == "tool_call"]


@pytest.fixture()
def ticket_db(tmp_path, monkeypatch):
    """把工单落库指到临时文件，避免测试污染 app/data/tickets.jsonl。"""
    import app.agent.tools as tools

    target = tmp_path / "tickets.jsonl"
    monkeypatch.setattr(tools, "TICKET_DB", target)
    return target


def test_explicit_request_creates_ticket(ticket_db):
    events = _run("给这卷划痕缺陷生成工单")
    assert "create_ticket" in _tool_names(events), "显式要求生成工单却没有触发建单"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "QT-L" in text, f"回答里没有工单号：{text[:120]}"
    assert "待质检员确认" in text
    assert ticket_db.exists(), "工单没有落库"


def test_no_ticket_without_explicit_request(ticket_db):
    events = _run("划痕怎么判定等级？")
    assert "create_ticket" not in _tool_names(events), "未明确要求却建了工单"
    assert not ticket_db.exists()


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少样例图")
def test_ticket_from_image_uses_judgement(ticket_db):
    from app.agent.tools import register_image

    iid = register_image(SAMPLE)
    events = _run("按这个判定结果生成工单", image_id=iid)
    assert "create_ticket" in _tool_names(events)
    calls = [e.get("args") or {} for e in events if e["type"] == "tool_call" and e["name"] == "create_ticket"]
    args = calls[0]
    assert args.get("defect_type") == "条带状夹杂", f"建单类型与图像判定不符：{args}"
    assert args.get("grade") in ("A", "B", "C")
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "QT-L" in text
