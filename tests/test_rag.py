"""检索层与 Agent 行为的关键路径测试。

重点锁住三类之前真出过问题的地方：
1. 判定标准必须排第一（否则答案会引用错的条款）
2. 参数类问题必须答出具体数值（否则等于没答）
3. 域外问题必须拒答（否则会编）
"""
from __future__ import annotations

from app.agent.tools import search_knowledge
from app.rag.index import HybridIndex


def test_index_covers_knowledge_base():
    idx = HybridIndex.load()
    assert idx.n >= 50, f"索引只有 {idx.n} 个 chunk，知识库可能没加载全"
    docs = {c.doc_id for c in idx.chunks}
    assert len(docs) == 8, f"应覆盖 8 篇文档，实际 {len(docs)}"


def test_grade_question_ranks_standard_first():
    """问等级时，第一条必须来自 01 判定标准。"""
    hits = search_knowledge("划痕怎么判定等级", 4).data["hits"]
    assert hits, "检索无命中"
    assert hits[0]["doc"].startswith("01"), f"首条来自 {hits[0]['doc']} · {hits[0]['heading']}"


def test_cause_question_ranks_cause_table_first():
    hits = search_knowledge("夹杂的根因是什么", 4).data["hits"]
    assert hits[0]["doc"].startswith("03"), f"首条来自 {hits[0]['doc']}"


def test_param_question_returns_concrete_value():
    """参数类问题要答出具体数值，不能只给一句「见规范」。"""
    from app.agent.runtime import AgentRuntime

    text = "".join(
        ev["text"] for ev in AgentRuntime().run("除鳞水压低于多少会出氧化皮压入？") if ev["type"] == "token"
    )
    assert "MPa" in text or "兆帕" in text, f"未给出压力数值：{text[:120]}"


def test_out_of_domain_is_refused():
    from app.agent.runtime import AgentRuntime

    text = "".join(ev["text"] for ev in AgentRuntime().run("帮我写一首关于春天的诗") if ev["type"] == "token")
    assert ("未收录" in text) or ("未涵盖" in text), f"域外问题未拒答：{text[:120]}"


def test_colloquial_query_is_rewritten():
    """口语词要能映射到标准术语，否则用户问不出正确答案。"""
    r = search_knowledge("板面发黑是什么问题", 4)
    assert r.data["hits"], "口语查询无命中"
    joined = " ".join(h["heading"] + h["text"] for h in r.data["hits"])
    assert ("氧化皮" in joined) or ("斑块" in joined) or ("色差" in joined), "口语改写没落到标准术语上"
