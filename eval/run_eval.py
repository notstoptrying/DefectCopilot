"""评测脚本：在固定 gold 集上跑端到端 Agent，输出可写进简历的指标。

指标定义：
- Recall@3      : 期望文档是否出现在检索前 3 条（文档级召回）
- MRR           : 首个命中期望文档的倒数排名
- 引用命中率     : 回答实际引用的条款里是否包含期望文档
- 关键词覆盖率   : 回答是否覆盖了该题的关键事实（如「15 MPa」）
- 拒答准确率     : 域外问题是否明确说明「未收录」
- 平均延迟       : 端到端耗时（离线模型，不含网络与 LLM 生成）

用法：python -m eval.run_eval
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.runtime import AgentRuntime  # noqa: E402

GOLD = Path(__file__).resolve().parent / "gold.jsonl"
OUT = Path(__file__).resolve().parent / "report.json"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_one(rt: AgentRuntime, item: dict) -> dict:
    text = ""
    citations: list[dict] = []
    tools: list[str] = []
    elapsed = 0
    for ev in rt.run(item["question"]):
        if ev["type"] == "token":
            text += ev["text"]
        elif ev["type"] == "tool_call":
            tools.append(ev["name"])
        elif ev["type"] == "citations":
            citations = ev["items"]
        elif ev["type"] == "done":
            elapsed = ev["elapsed_ms"]

    cited_docs = [c["doc"][:2] for c in citations]
    expected = item["expected_docs"]
    top3 = cited_docs[:3]

    if expected:
        recall = 1.0 if any(d in top3 for d in expected) else 0.0
        rank = next((i + 1 for i, d in enumerate(cited_docs) if d in expected), None)
        mrr = 1.0 / rank if rank else 0.0
        cite_hit = 1.0 if any(d in cited_docs for d in expected) else 0.0
    else:
        recall = mrr = cite_hit = None

    kw_total = item["must_include"]
    kw_hit = sum(1 for k in kw_total if k in text)
    kw_cov = (kw_hit / len(kw_total)) if kw_total else None

    refused = ("未收录" in text) or ("不在" in text and "范围" in text)
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "answer_chars": len(text),
        "tools": tools,
        "recall@3": recall,
        "mrr": mrr,
        "cite_hit": cite_hit,
        "kw_coverage": kw_cov,
        "should_refuse": item["should_refuse"],
        "refused": refused,
        "refuse_ok": (refused == item["should_refuse"]),
        "elapsed_ms": elapsed,
        "answer": text,
    }


def evaluate(gold: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """跑完整评测集，返回 (汇总指标, 逐题明细)。供 CLI 与 pytest 共用。"""
    gold = gold if gold is not None else load_gold()
    rt = AgentRuntime()
    rows = [run_one(rt, it) for it in gold]

    ans = [r for r in rows if not r["should_refuse"]]
    out = [r for r in rows if r["should_refuse"]]

    def avg(key, data):
        vals = [r[key] for r in data if r[key] is not None]
        return round(statistics.mean(vals), 4) if vals else None

    summary = {
        "cases": len(rows),
        "answerable": len(ans),
        "out_of_domain": len(out),
        "recall@3": avg("recall@3", ans),
        "mrr": avg("mrr", ans),
        "cite_hit_rate": avg("cite_hit", ans),
        "kw_coverage": avg("kw_coverage", ans),
        "refuse_accuracy": avg("refuse_ok", out),
        "kw_full_match": round(sum(1 for r in ans if r["kw_coverage"] == 1.0) / max(1, len(ans)), 4),
        "avg_latency_ms": int(statistics.mean([r["elapsed_ms"] for r in rows])),
        "p95_latency_ms": int(sorted(r["elapsed_ms"] for r in rows)[int(len(rows) * 0.95) - 1]),
        "provider": rt.client.mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary, rows


def main() -> None:
    summary, rows = evaluate()

    OUT.write_text(
        json.dumps({"summary": summary, "details": [{k: v for k, v in r.items() if k != "answer"} for r in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*72}\n评测完成  provider={summary['provider']}  cases={summary['cases']}\n{'='*72}")
    for r in rows:
        flag = "✓" if (r["refuse_ok"] if r["should_refuse"] else (r["recall@3"] and (r["kw_coverage"] or 0) >= 0.99)) else "·"
        print(
            f"{flag} {r['id']} R@3={r['recall@3'] if r['recall@3'] is not None else '-'} "
            f"MRR={r['mrr'] if r['mrr'] is not None else '-'} "
            f"引用={r['cite_hit'] if r['cite_hit'] is not None else '-'} "
            f"关键词={r['kw_coverage'] if r['kw_coverage'] is not None else '-'} "
            f"拒答={'Y' if r['refused'] else 'N'} {r['elapsed_ms']}ms | {r['question']}"
        )
    print("-" * 72)
    for k, v in summary.items():
        print(f"{k:>16}: {v}")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
