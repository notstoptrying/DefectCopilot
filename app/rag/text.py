"""中文文本处理：分词、分句、缺陷类型识别。

不引入 jieba 等外部词典依赖，采用「汉字单字 + 二元组 + 英文数字词」的
混合切分策略：单字保证召回，二元组保证短语精度，对质检这类专有名词密集
的短文本效果足够，且零依赖、可复现。
"""
from __future__ import annotations

import re

_HAN = r"\u4e00-\u9fff"
_RE_EN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_RE_HAN_RUN = re.compile(f"[{_HAN}]+")
_RE_TOKEN = re.compile(f"[{_HAN}]+|[a-z0-9]+(?:[.\\-][a-z0-9]+)*")

# 六类标准缺陷（与《表面缺陷判定标准》一致）
DEFECT_TYPES = ["氧化皮压入", "划痕", "裂纹", "夹杂", "麻点", "斑块"]

# 停用词：对检索无区分度的高频词
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "对", "为", "被", "把",
    "吗", "呢", "啊", "吧", "么", "怎么", "如何", "什么", "哪些", "这个",
    "那个", "一个", "可以", "需要", "应该", "请问", "我们", "你们", "他们",
    "以及", "如果", "因为", "所以", "但是", "而且", "就是", "要", "会", "能",
}


def normalize(text: str) -> str:
    """统一全角数字/字母/百分号，压缩空白。

    注意：只转换全角字母数字与少数符号，保留中文标点（，。：；），
    否则展示给用户的回答里会出现英文逗号冒号，读起来很割裂。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            code -= 0xFEE0  # 全角数字/字母 -> 半角
        elif ch in "％＃＆．＋－／":
            code = ord({"％": "%", "＃": "#", "＆": "&", "．": ".", "＋": "+", "－": "-", "／": "/"}[ch])
        out.append(chr(code))
    s = "".join(out)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """切分为检索单元。"""
    s = normalize(text).lower()
    tokens: list[str] = []
    for m in _RE_TOKEN.finditer(s):
        piece = m.group(0)
        if re.fullmatch(f"[{_HAN}]+", piece):
            chars = list(piece)
            if not keep_stopwords:
                chars = [c for c in chars if c not in STOPWORDS]
            tokens.extend(chars)
            for i in range(len(piece) - 1):
                bg = piece[i : i + 2]
                if keep_stopwords or not (bg[0] in STOPWORDS or bg[1] in STOPWORDS):
                    tokens.append(bg)
        else:
            if len(piece) > 1 or piece.isdigit():
                tokens.append(piece)
    return tokens


def split_sentences(text: str) -> list[str]:
    """按句号/分号/换行切句，保留列表项作为一个语义单元。不做字符转换，保证原文可展示。"""
    raw: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "|")):
            raw.append(line)
            continue
        parts = re.split(r"(?<=[。！？；!?;])", line)
        raw.extend(p.strip() for p in parts if p.strip())
    return [r for r in raw if len(r) >= 4]


def detect_defect_types(text: str) -> list[str]:
    """从文本中识别提到的缺陷类型，长词优先，避免「氧化皮压入」被切成「斑块」。"""
    found: list[str] = []
    for t in DEFECT_TYPES:
        if t in text:
            found.append(t)
    # 「氧化皮压入」命中时不再单独计「氧化皮」相关词
    return found


# 领域词表：用于判断问题是否落在本知识库范围内。
# 域外问题即便余弦相似度不算低（中文短句普遍 0.1 左右），也应明确拒答。
DOMAIN_TERMS = (
    "钢", "轧", "辊", "除鳞", "缺陷", "划痕", "裂纹", "夹杂", "麻点", "氧化皮", "斑块", "色差",
    "判定", "等级", "判废", "修磨", "切除", "降级", "返修", "参数", "温度", "水压", "水量",
    "工单", "质检", "质量", "产线", "点检", "投诉", "8D", "8d", "板面", "表面", "卷取",
    "精轧", "粗轧", "层流", "酸洗", "喷嘴", "矫直", "连铸", "还炉", "回炉",
)


def has_domain_terms(text: str) -> bool:
    return any(t in text for t in DOMAIN_TERMS)


def overlap_score(query_tokens: set[str], text: str) -> float:
    """查询词覆盖率，用于轻量重排。"""
    if not query_tokens:
        return 0.0
    toks = set(tokenize(text))
    hit = len(query_tokens & toks)
    return hit / max(1, len(query_tokens))
