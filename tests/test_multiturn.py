"""多轮对话与带图追问的回归测试。

背景：mock 通道曾把「第一条 user 消息」当当前问题，单轮评测抓不到——
带图追问时检索与工单统计全部跑到上一轮的问题上。这组测试锁住多轮场景。
"""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample_defect.jpg"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="缺少样例图")


@pytest.fixture(scope="module")
def two_turn_events():
    from app.agent.runtime import AgentRuntime
    from app.agent.tools import register_image

    rt = AgentRuntime()
    first, second = [], []
    for ev in rt.run("除鳞水压低于多少会出氧化皮压入？"):
        first.append(ev)
    iid = register_image(SAMPLE)
    for ev in rt.run("这张图有什么缺陷，判几级？", image_id=iid, history=_compact(first)):
        second.append(ev)
    return first, second


def _compact(events: list[dict]) -> list[dict]:
    """把第一轮事件压成 history 结构，与前端 send() 的做法一致。"""
    text = "".join(e["text"] for e in events if e["type"] == "token")
    return [
        {"role": "user", "content": "除鳞水压低于多少会出氧化皮压入？"},
        {"role": "assistant", "content": text},
    ]


def _tools(events: list[dict]) -> list[tuple[str, dict]]:
    return [(e["name"], e.get("args") or {}) for e in events if e["type"] == "tool_call"]


def test_second_turn_searches_current_question(two_turn_events):
    """带图追问时，检索 query 必须围绕当前问题（含图像给出的类型），不能跑偏到上一轮。"""
    _, second = two_turn_events
    searches = [a for n, a in _tools(second) if n == "search_knowledge"]
    assert searches, "带图追问没有触发知识库检索"
    query = searches[0].get("query") or ""
    assert "这张图" in query or "条带状夹杂" in query, f"检索跑偏到旧问题：{query}"


def test_second_turn_case_db_uses_image_type(two_turn_events):
    """工单统计应按图像检出的缺陷类型查，而不是上一轮问题里的类型。"""
    _, second = two_turn_events
    cases = [a for n, a in _tools(second) if n == "query_case_db"]
    if cases:  # 该问题未触发工单统计时跳过（触发与否由 mock 决策决定）
        assert cases[0].get("defect_type") != "氧化皮压入", f"工单统计用了旧问题类型：{cases[0]}"


def test_second_turn_answer_matches_image(two_turn_events):
    """回答结论应与图像检出一致（条带状夹杂），而不是上一轮问题的氧化皮。"""
    _, second = two_turn_events
    text = "".join(e["text"] for e in second if e["type"] == "token")
    assert "条带状夹杂" in text, f"结论与图像不符：{text[:150]}"
    assert "氧化皮压入高风险" not in text.split("可能根因")[0], "结论段引用了上一轮问题的判定"
