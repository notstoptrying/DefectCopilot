"""混合检索索引：BM25 稀疏召回 + TF-IDF 向量召回 + RRF 融合。

为什么不是单一检索：
- BM25 对「除鳞水压」「Ra 1.2 μm」这类精确术语命中好，但不懂同义改写；
- 向量召回对「板面发黑是什么问题」这类口语化提问更稳，但会把数值约束稀释；
- 两路召回用 RRF 融合，不需要调分数量纲，工程上最稳。

向量部分默认使用字符 n-gram TF-IDF（本地、零依赖、可复现）；接入外部
embedding 服务时只需替换 `_build_vectors`，检索与融合逻辑不变。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import INDEX_DIR, KB_DIR, RRF_K, TOP_K
from .chunker import Chunk, build_chunks
from .rewrite import expand_query, route_intent
from .text import DEFECT_TYPES, overlap_score, tokenize

BM25_K1 = 1.5
BM25_B = 0.75
# 融合权重：BM25 主导术语精确命中，向量补语义泛化，覆盖率与标题命中做轻量重排
W_BM25 = 0.45
W_VEC = 0.25
W_COV = 0.20
W_HEAD = 0.10
W_INTENT = 0.12


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float
    bm25_rank: int | None
    vec_rank: int | None

    def brief(self) -> dict:
        return {
            "cid": self.chunk.cid,
            "doc": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "heading": self.chunk.heading,
            "snippet": self.chunk.body[:200].replace("\n", " "),
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
        }


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.n = len(chunks)
        self.doc_tokens: list[list[str]] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.vocab: dict[str, int] = {}
        self.idf_arr: np.ndarray = np.zeros(0, dtype=np.float32)
        self.matrix: np.ndarray = np.zeros((self.n, 0), dtype=np.float32)
        self._prepare_tokens()
        self._build_vectors()

    # ---------- 构建 ----------
    def _prepare_tokens(self) -> None:
        self.doc_tokens = []
        for c in self.chunks:
            toks = tokenize(c.text)
            # 小节标题是这一块的「主题标签」，重复两次以提升权重：
            # 没有这层加权，「划痕」会被正文里反复出现该词的根因对照表压过判定标准正文。
            toks.extend(tokenize(c.heading) * 2)
            self.doc_tokens.append(toks)
        self.tf = [Counter(t) for t in self.doc_tokens]
        for tf in self.tf:
            for t in tf:
                self.df[t] += 1
        total = sum(len(t) for t in self.doc_tokens) or 1
        self.avgdl = total / max(1, self.n)
        self.idf = {
            t: math.log(1 + (self.n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()
        }

    def _build_vectors(self) -> None:
        tokens_sorted = sorted(self.df)
        self.vocab = {t: i for i, t in enumerate(tokens_sorted)}
        self.idf_arr = np.array(
            [math.log((self.n + 1) / (self.df[t] + 1)) + 1.0 for t in tokens_sorted],
            dtype=np.float32,
        )
        mat = np.zeros((self.n, len(self.vocab)), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            for t, cnt in tf.items():
                j = self.vocab.get(t)
                if j is not None:
                    mat[i, j] = (1.0 + math.log(cnt)) * self.idf_arr[j]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = mat / norms

    # ---------- 检索 ----------
    def _bm25_scores(self, q_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        q_set = set(q_tokens)
        for i in range(self.n):
            dl = len(self.doc_tokens[i]) or 1
            s = 0.0
            tf = self.tf[i]
            for t in q_set:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (BM25_K1 + 1)) / (
                    f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
                )
            scores[i] = s
        return scores

    def _vector_scores(self, q_tokens: list[str]) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for t, cnt in Counter(q_tokens).items():
            j = self.vocab.get(t)
            if j is not None:
                q[j] = (1.0 + math.log(cnt)) * self.idf_arr[j]
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return np.zeros(self.n, dtype=np.float32)
        return self.matrix @ (q / norm)

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchHit]:
        if not self.chunks or not query.strip():
            return []
        expanded, rewritten = expand_query(query)
        q_tokens = tokenize(expanded)
        if not q_tokens:
            return []
        q_set = set(tokenize(query))
        hint = [t for t in DEFECT_TYPES if t in expanded]

        bm25 = self._bm25_scores(q_tokens)
        cos = self._vector_scores(q_tokens)
        nb = bm25 / (float(bm25.max()) or 1.0)

        pool = min(self.n, max(top_k * 5, 24))
        bm_rank = {int(i): r for r, i in enumerate(np.argsort(-bm25)[:pool])}
        vec_rank = {int(i): r for r, i in enumerate(np.argsort(-cos)[:pool])}
        # RRF 只用于合并候选集：两路各自的前 pool 名取并集，避免单路漏召
        candidates = set(bm_rank) | set(vec_rank)

        scored: list[tuple[float, int]] = []
        intent_docs = route_intent(query)
        for i in candidates:
            cov = overlap_score(q_set, self.chunks[i].text)
            head = 1.0 if any(h in self.chunks[i].heading for h in hint) else 0.0
            final = (
                W_BM25 * float(nb[i])
                + W_VEC * float(cos[i])
                + W_COV * cov
                + W_HEAD * head
            )
            # 意图路由：命中对应文档族再加一档先验
            if intent_docs and self.chunks[i].doc_id[:2] in intent_docs:
                final += W_INTENT
            if final <= 0:
                continue
            scored.append((final, i))

        scored.sort(key=lambda x: -x[0])
        return [
            SearchHit(
                chunk=self.chunks[i],
                score=float(s),
                bm25=float(bm25[i]),
                cosine=float(cos[i]),
                bm25_rank=bm_rank.get(i),
                vec_rank=vec_rank.get(i),
            )
            for s, i in scored[:top_k]
        ]

    # ---------- 持久化 ----------
    def save(self, out_dir: Path | None = None) -> None:
        out_dir = out_dir or INDEX_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            out_dir / "vectors.npz",
            matrix=self.matrix,
            idf=self.idf_arr,
            vocab=np.array(json.dumps(self.vocab, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, src_dir: Path | None = None, rebuild: bool = False) -> "HybridIndex":
        src_dir = src_dir or INDEX_DIR
        chunk_file = src_dir / "chunks.json"
        vec_file = src_dir / "vectors.npz"
        fresh = (
            not rebuild
            and chunk_file.exists()
            and vec_file.exists()
            and chunk_file.stat().st_mtime >= max((p.stat().st_mtime for p in KB_DIR.glob("*.md")), default=0)
        )
        if not fresh:
            idx = cls(build_chunks())
            idx.save(src_dir)
            return idx

        chunks = [Chunk(**r) for r in json.loads(chunk_file.read_text(encoding="utf-8"))]
        idx = cls.__new__(cls)
        idx.chunks = chunks
        idx.n = len(chunks)
        idx.doc_tokens, idx.tf, idx.df, idx.idf, idx.avgdl = [], [], Counter(), {}, 0.0
        idx.vocab, idx.idf_arr = {}, np.zeros(0, dtype=np.float32)
        blob = np.load(vec_file, allow_pickle=False)
        idx.vocab = json.loads(blob["vocab"].tolist())
        idx.idf_arr = blob["idf"]
        idx.matrix = blob["matrix"]
        idx._prepare_tokens()
        return idx

    def stats(self) -> dict:
        return {
            "docs": len({c.doc_id for c in self.chunks}),
            "chunks": len(self.chunks),
            "vocab": len(self.vocab),
            "avg_chunk_chars": int(np.mean([len(c.text) for c in self.chunks])) if self.chunks else 0,
        }


_CACHE: HybridIndex | None = None


def get_index(rebuild: bool = False) -> HybridIndex:
    global _CACHE
    if _CACHE is None or rebuild:
        _CACHE = HybridIndex.load(rebuild=rebuild)
    return _CACHE
