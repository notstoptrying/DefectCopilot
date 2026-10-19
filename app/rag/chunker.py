"""文档切分：按 Markdown 标题结构切成语义块。

质检类文档是强结构化的（章 → 节 → 条目），所以按标题切比固定长度窗口
更能保证「一个 chunk 讲一件事」。chunk 头部拼上「文档名 · 小节名」，
既保留上下文，也让 BM25 能命中标题里的关键词。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, KB_DIR


@dataclass
class Chunk:
    cid: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    order: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def body(self) -> str:
        """去掉标题前缀后的正文。"""
        if "\n" in self.text:
            return self.text.split("\n", 1)[1]
        return self.text

    def to_dict(self) -> dict:
        return {
            "cid": self.cid,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "heading": self.heading,
            "text": self.text,
            "order": self.order,
        }


def _split_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    """返回 (文档标题, [(小节标题, 正文)])"""
    lines = md.split("\n")
    doc_title = ""
    sections: list[tuple[str, str]] = []
    cur_heading = "概述"
    buf: list[str] = []
    preface: list[str] = []

    for line in lines:
        if line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
            continue
        if re.match(r"^#{2,3}\s+", line):
            body = "\n".join(buf).strip()
            if body:
                sections.append((cur_heading, body))
            elif not sections and buf:
                preface.append("")
            cur_heading = re.sub(r"^#{2,3}\s+", "", line).strip()
            buf = []
            continue
        buf.append(line)

    body = "\n".join(buf).strip()
    if body:
        sections.append((cur_heading, body))
    return doc_title or "未命名文档", sections


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    """按段落聚合成不超过 max_chars 的块，块间保留尾部重叠。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur}\n{p}"
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
            # 从句子边界开始重叠，避免截断半句话
            if tail:
                cut = re.search(r"[。；！？\n]", tail)
                if cut and cut.start() < len(tail) - 8:
                    tail = tail[cut.start() + 1 :]
            cur = f"{tail}\n{p}".strip()
    if cur:
        chunks.append(cur)

    # 单块仍超长时按句子硬切
    result: list[str] = []
    for c in chunks:
        if len(c) <= max_chars * 1.6:
            result.append(c)
            continue
        sents = re.split(r"(?<=[。；！？])", c)
        buf = ""
        for s in sents:
            if buf and len(buf) + len(s) > max_chars:
                result.append(buf)
                buf = s
            else:
                buf += s
        if buf:
            result.append(buf)
    return result


def build_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    kb_dir = kb_dir or KB_DIR
    chunks: list[Chunk] = []
    order = 0
    for path in sorted(kb_dir.glob("*.md")):
        doc_id = path.stem
        doc_title, sections = _split_sections(path.read_text(encoding="utf-8"))
        for heading, body in sections:
            for piece in _pack(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP):
                order += 1
                chunks.append(
                    Chunk(
                        cid=f"{doc_id}#{order:03d}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        heading=heading,
                        text=f"【{doc_title} · {heading}】\n{piece}",
                        order=order,
                    )
                )
    return chunks


if __name__ == "__main__":
    cs = build_chunks()
    print(f"chunks={len(cs)}")
    for c in cs[:3]:
        print("-" * 40)
        print(c.cid, c.heading, len(c.text))
        print(c.text[:200])
