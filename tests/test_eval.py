"""检索与回答质量的回归测试（指标阈值来自 eval/gold.jsonl）。

这些阈值不是「随便定个好听的数」，而是当前实测值的下沿：
低于它们说明某次改动把效果改坏了，需要先看是哪道题掉下来。
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def summary(eval_result):
    return eval_result[0]


def test_recall_at_3(summary):
    """检索前 3 条必须覆盖期望文档。低于 1.0 就说明排序退化了。"""
    assert summary["recall@3"] >= 0.95, f"Recall@3 掉到 {summary['recall@3']}"


def test_mrr(summary):
    """期望文档应尽量排在第 1 位。"""
    assert summary["mrr"] >= 0.85, f"MRR 掉到 {summary['mrr']}"


def test_citation_hit(summary):
    """回答引用的条款里要包含期望文档——防止「检索对了但答案没用上」。"""
    assert summary["cite_hit_rate"] >= 0.95, f"引用命中率 {summary['cite_hit_rate']}"


def test_keyword_coverage(summary):
    """关键事实覆盖率：答不出「15 MPa」这种硬指标即视为不合格。"""
    assert summary["kw_coverage"] >= 0.95, f"关键词覆盖 {summary['kw_coverage']}"


def test_keyword_full_match(summary):
    """全部关键事实都覆盖到的题目占比。"""
    assert summary["kw_full_match"] >= 0.9, f"全匹配率 {summary['kw_full_match']}"


def test_refuse_accuracy(summary):
    """域外问题必须拒答，不能硬编答案。"""
    assert summary["refuse_accuracy"] == 1.0, f"拒答准确率 {summary['refuse_accuracy']}"


def test_latency(summary):
    """离线通道的端到端延迟上限，防止某次改动引入逐条重建索引之类的退化。"""
    assert summary["p95_latency_ms"] < 500, f"P95 延迟 {summary['p95_latency_ms']} ms"
