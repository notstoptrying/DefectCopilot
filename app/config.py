"""全局配置：模型通道、路径、检索参数。

所有密钥只从环境变量读取，不落盘、不写进代码。
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "kb"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CASE_CSV = DATA_DIR / "cases.csv"
TICKET_DB = DATA_DIR / "tickets.jsonl"
WEB_DIR = BASE_DIR.parent / "web"

# ---------- 模型通道 ----------
# mock  : 离线规则模型，无需任何密钥，可完整跑通 Agent 流程
# openai: 任意 OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / Moonshot / Ollama ...）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------- 数据集路径（仅训练与标定脚本使用）----------
# 训练/标定要读 NEU-Seg 数据集，但它不进仓库（原文几百 MB）。
# 默认找仓库内 data/NEU_Seg，也可用环境变量指向本机任意位置，避免把个人绝对路径写死进代码。
DATASET_DIR = Path(os.getenv("NEU_SEG_DIR", DATA_DIR / "NEU_Seg"))

# ---------- 检索参数 ----------
CHUNK_MAX_CHARS = 460
CHUNK_OVERLAP = 80
TOP_K = 4
RRF_K = 60
# 低于该相似度视为「知识库未收录」，触发拒答而不是硬编答案
MIN_EVIDENCE_SCORE = float(os.getenv("MIN_EVIDENCE_SCORE", "0.09"))

# ---------- Agent 参数 ----------
MAX_STEPS = 5


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
