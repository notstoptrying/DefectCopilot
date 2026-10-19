"""答案组织：把检索到的条款组织成「结论—依据—根因—处置」的结构化回答。

这段逻辑是离线模型（无 API Key 时的演示通道）的核心：它不做自由生成，
只做「抽取 + 组织 + 溯源」，所以不会编造数字——每条结论都能回指到原文条款。
接真模型时这层不参与生成，只作为兜底与对照基线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..rag.rewrite import route_intent
from ..rag.text import DEFECT_TYPES, overlap_score, split_sentences, tokenize

# 文档编号 -> 章节归类
DOC_BUCKET = {
    "01": "basis",
    "02": "param",
    "03": "cause",
    "04": "action",
    "05": "param",
    "06": "record",
    "07": "record",
    "08": "case",
}
BUCKET_TITLE = {
    "basis": "判定依据",
    "cause": "可能根因",
    "action": "建议处置",
    "param": "工艺与设备参数",
    "record": "记录与流程要求",
    "case": "相似案例",
}
BUCKET_LIMIT = {"basis": 3, "cause": 3, "action": 3, "param": 2, "record": 2, "case": 2}
BUCKET_ORDER = ["basis", "cause", "action", "param", "record", "case"]

_NUM_HINT = re.compile(r"\d+(\.\d+)?")


def _is_cause_row(text: str) -> bool:
    """根因表的一行形如「P1：结晶器保护渣卷入，…」，P 前缀是唯一可靠标记。"""
    return bool(re.match(r"^P[1-3]", text))


@dataclass
class Picked:
    text: str
    idx: int
    score: float


def _clean(line: str) -> str:
    line = line.strip()
    if line.startswith("|"):
        # 参数表是数据密度最高的一类内容（阈值、温度、压力都在这），
        # 直接过滤会让回答答不出「870 ℃」这种关键数字，所以转成「参数：值」的叙述句
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [c for c in cells if c and not re.fullmatch(r":?-{2,}:?", c)]
        header_words = ("控制范围", "目标值", "越限后果", "判别特征", "验证方法", "可能根因", "取值示例", "时限要求", "责任岗位")
        if sum(1 for w in header_words if w in line) >= 2:
            return ""  # 表头行，无信息量
        if len(cells) >= 2:
            head, rest = cells[0], "，".join(cells[1:])
            return f"{head}：{rest}"
        return "，".join(cells)
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+\.\s*", "", line)
    return line.strip()


def _score(sentence: str, q_set: set[str], defect_hint: str) -> float:
    s = overlap_score(q_set, sentence)
    if _NUM_HINT.search(sentence):
        s += 0.25          # 带数字的条款通常是硬指标，优先展示
    if "级" in sentence:
        s += 0.15
    if defect_hint and defect_hint in sentence:
        s += 0.35
    if len(sentence) < 8:
        s -= 0.3
    return s


def pick_sentences(
    hits: list[dict],
    q_set: set[str],
    defect_hint: str,
    limit: int,
    max_per_chunk: int = 2,
    accept: "callable | None" = None,
) -> list[Picked]:
    cands: list[Picked] = []
    for h in hits:
        body = h["text"].split("\n", 1)[1] if "\n" in h["text"] else h["text"]
        for raw in split_sentences(body):
            line = _clean(raw)
            if len(line) < 8 or line.startswith("文件编号"):
                continue
            if accept is not None and not accept(line):
                continue
            cands.append(Picked(text=line, idx=h["idx"], score=_score(line, q_set, defect_hint)))
    cands.sort(key=lambda x: -x.score)
    # 同一条款内不重复取太多，避免整段复制
    seen: set[str] = set()
    out: list[Picked] = []
    per_idx: dict[int, int] = {}
    for c in cands:
        key = c.text[:24]
        if key in seen:
            continue
        if per_idx.get(c.idx, 0) >= max_per_chunk:
            continue
        seen.add(key)
        per_idx[c.idx] = per_idx.get(c.idx, 0) + 1
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _where(box: dict, size: int = 200) -> str:
    """把外接框中心换算成「左/中/右 + 上/中/下」的方位词，方便质检员按图找人。"""
    try:
        cx = (float(box.get("x1", 0)) + float(box.get("x2", 0))) / 2
        cy = (float(box.get("y1", 0)) + float(box.get("y2", 0))) / 2
    except (TypeError, ValueError):
        return "画面内"
    col = "左侧" if cx < size / 3 else ("右侧" if cx > size * 2 / 3 else "中部")
    row = "偏上" if cy < size / 3 else ("偏下" if cy > size * 2 / 3 else "居中")
    return f"画面{col}{row}"


def compose(
    question: str,
    hits: list[dict],
    image: dict | None = None,
    cases: dict | None = None,
    refused: bool = False,
) -> str:
    q_set = set(tokenize(question))
    defect_hint = ""
    for t in DEFECT_TYPES:
        if t in question or (image and image.get("defect_type") == t):
            defect_hint = t
            break
    if image and image.get("defect_type"):
        defect_hint = image["defect_type"]

    if refused or (not hits and not image and not cases):
        return (
            "知识库中未收录与这个问题直接相关的条款。\n\n"
            "我可以回答的范围包括：六类表面缺陷的判定分级、根因排查顺序、处置与返修流程、"
            "工艺参数控制范围、设备点检要求、工单填写与 8D 流程。\n\n"
            "如果你手上有缺陷图片，可以直接上传，我按图像结果给出判定建议。"
        )

    blocks: list[str] = []
    used: set[int] = set()

    # 先分桶：结论要优先引用「判定标准」族，而不是随便挑一条高分句
    buckets: dict[str, list[dict]] = {}
    for h in hits:
        b = DOC_BUCKET.get(h["doc"][:2], "record")
        # 03 号文档末尾那节是「排查结论记录要求」，讲流程不讲根因，混进根因块会挤掉真原因
        if b == "cause" and "记录" in (h.get("heading") or ""):
            b = "record"
        buckets.setdefault(b, []).append(h)

    # ---------- 结论 ----------
    if image:
        n = len(image.get("boxes") or [])
        level = image.get("level", "B")
        dtype = image.get("defect_type", "缺陷")
        area = float(image.get("area_ratio") or 0)
        maxr = float(image.get("max_region_ratio") or 0)
        conf = image.get("confidence")
        conf_txt = f"，置信度 {conf:.0%}" if isinstance(conf, float) else ""
        model = image.get("model", "检测模型")
        rule = image.get("level_rule") or {}

        if not n or dtype == "无缺陷":
            blocks.append(
                f"**结论**\n未检出明显缺陷（{model}）。局部判定为 **A 级**，可直接放行；"
                f"若下游工序仍反馈异常，建议改为抽查复判。"
            )
        else:
            action = {
                "A": "属轻微，可修磨后交货，本卷无需降级",
                "B": "需切除缺陷段或降级销售",
                "C": "整卷隔离，不得直接进入下一工序",
            }.get(level, "需质检员复判")
            where = ""
            if image.get("boxes"):
                where = f"（{_where(image['boxes'][0])}）"
            parts = [
                f"**结论**\n局部视场内检出 **{n} 处 {dtype}**（{model}{conf_txt}），"
                f"合计占视场 **{area:.2%}**，最大单块 {maxr:.2%}{where}。"
                f"按该类缺陷的局部标定档位判为 **{level} 级** —— {action}。"
            ]
            if rule.get("b_max"):
                parts.append(
                    f"定档依据：A≤{rule['a_max']:.2%}、B≤{rule['b_max']:.2%}、超过即 C；"
                    f"本次 {area:.2%} 落在 **{level}** 区间。"
                )
            if level in ("B", "C"):
                parts.append("连带要求：同批次连续 3 卷出现同类 B 级及以上，按批量质量异常启动 8D。")
            if level == "A":
                parts.append("无需开质量工单；若缺陷位于板宽边部 50 mm 内，仍建议记录留档。")
            parts.append("注：等级为单图局部初判；标准中「1.5%」是整卷板面口径，整卷等级需按全卷累计面积与缺陷长度判定。")
            blocks.append("\n".join(parts))
    elif hits:
        intent = route_intent(question)
        # 问参数（水压/温度/速度/范围）时，结论应取参数族，而不是判定分级
        param_first = ("02" in intent or "05" in intent) and buckets.get("param")
        # 只问根因（没同时问判定）时，结论应落在根因表的最高优先级那条，而不是复述分级
        cause_first = (
            not param_first and "03" in intent and "01" not in intent and bool(buckets.get("cause"))
        )
        if param_first:
            basis_hits = buckets["param"]
        elif cause_first:
            basis_hits = buckets["cause"]
        else:
            basis_hits = buckets.get("basis") or hits
        top = pick_sentences(
            basis_hits, q_set, defect_hint, 3, max_per_chunk=3,
            accept=_is_cause_row if cause_first else None,
        )
        head = f"检索到 {len(hits)} 条直接相关条款"
        if cause_first and top:
            # 结论要引用 P1，而不是检索分数最高的那条（分数不含优先级语义）
            top.sort(key=lambda p: (p.text[:2], -p.score))
        levels = [] if (param_first or cause_first) else [p for p in top if re.match(r"^[ABC]\s*级", p.text)]
        if len(levels) >= 2:
            blocks.append(
                f"**结论**\n{head}。该缺陷按长度、宽度与面积占比分 A/B/C 三档，"
                f"三档阈值见下方「判定依据」，取其中满足的最高等级定级。"
            )
            used.update(p.idx for p in levels)
        elif cause_first and top:
            blocks.append(
                f"**结论**\n{head}。按根因对照表，最高优先级为「{top[0].text}」[{top[0].idx}]，"
                f"全部根因与判别特征见下方「可能根因」。"
            )
            used.add(top[0].idx)
        elif top:
            blocks.append(f"**结论**\n{head}，核心要点：{top[0].text} [{top[0].idx}]")
            used.add(top[0].idx)
        else:
            blocks.append(f"**结论**\n{head}，详见下方依据。")

    for b in BUCKET_ORDER:
        bh = buckets.get(b) or []
        if not bh:
            continue
        limit = BUCKET_LIMIT[b] + (1 if defect_hint and b in ("basis", "cause") else 0)
        accept = None
        per_chunk = 3 if b == "basis" else 2
        if b == "cause":
            # 根因表每行就是一条根因，形如「P1：结晶器保护渣卷入，…」；
            # 同节里的说明句、流程句不带 P 前缀，不应混进来
            accept = _is_cause_row
            per_chunk = 4
            limit = max(limit, 4)
        elif b == "record":
            # 记录/流程块只在问流程类问题时才有价值。问「划痕怎么产生」时把工单字段贴上来是噪音，
            # 所以按意图整块决定是否展示，而不是逐句过滤（逐句过滤会误删「编号规则」这类真答案）
            if not re.search(r"工单|编号|填写|记录|归档|台账|8D|8d|流程|投诉|时限|签字|审批", question):
                continue
        picked = pick_sentences(bh, q_set, defect_hint, limit, max_per_chunk=per_chunk, accept=accept)
        if not picked:
            continue
        if b == "cause":
            # 按 P1 → P3 排，检索分数高但优先级低的根因不该排在前面
            picked.sort(key=lambda p: (p.text[:2], -p.score))
        if b == "basis":
            # 等级条款按 A→B→C 展示，避免出现 C、A、B 的错乱顺序
            rank = {"A": 0, "B": 1, "C": 2}
            picked.sort(key=lambda p: rank.get(p.text[0], 3) if re.match(r"^[ABC]\s*级", p.text) else 3)
        lines = []
        for p in picked:
            txt = p.text
            if len(txt) > 180:
                txt = txt[:178] + "…"
            lines.append(f"- {txt} [{p.idx}]")
            used.add(p.idx)
        blocks.append(f"**{BUCKET_TITLE[b]}**\n" + "\n".join(lines))

    # ---------- 历史工单 ----------
    if cases and cases.get("count"):
        rows = cases.get("rows") or []
        head = f"**相似历史工单**（命中 {cases['count']} 条"
        if cases.get("avg_area") is not None:
            head += f"，平均面积占比 {cases['avg_area']}%"
        if cases.get("total_loss") is not None:
            head += f"，累计损失 {cases['total_loss']} 万元"
        head += "）"
        lines = [
            f"- {r['date']} {r['line']} 线 {r['steel']} · {r['defect']} {r['grade']} 级 · "
            f"面积 {r['area']} · 根因：{r['cause']} · 处置：{r['disposal']}"
            for r in rows[:5]
        ]
        if cases.get("top_causes"):
            lines.append("- 根因分布：" + "、".join(f"{k}（{v} 次）" for k, v in cases["top_causes"][:3]))
        blocks.append(head + "\n" + "\n".join(lines))

    # ---------- 参考条款 ----------
    if hits:
        refs = []
        for h in hits:
            mark = "" if h["idx"] in used else "（未直接引用）"
            refs.append(f"[{h['idx']}] {h['doc_title']} · {h['heading']}{mark}")
        blocks.append("**参考条款**\n" + "\n".join(refs))

    if image and image.get("annotated_url"):
        blocks.append(f"标注结果：{image['annotated_url']}")

    return "\n\n".join(blocks)
