"""FastAPI 服务：SSE 流式问答 + 图片上传 + 健康检查 + 评测结果读取。

启动：
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent.runtime import AgentRuntime, build_client
from .agent.tools import register_image
from .config import UPLOAD_DIR, WEB_DIR, ensure_dirs
from .rag.index import get_index

ensure_dirs()

app = FastAPI(title="DefectCopilot", description="多模态 RAG Agent 应用框架 · 工业质检场景", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
_runtime = AgentRuntime()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/files/{name}")
def uploaded(name: str) -> FileResponse:
    path = UPLOAD_DIR / Path(name).name
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict:
    client = build_client()
    return {
        "status": "ok",
        "llm": client.health(),
        "index": get_index().stats(),
        "mode_note": "离线规则模型，无需 API Key" if client.mode == "mock" else "接入真实大模型",
    }


@app.get("/api/samples")
def samples() -> dict:
    return {
        "items": [
            "划痕怎么判定等级？",
            "除鳞水压低于多少会出氧化皮压入？",
            "板面发黑是什么问题？",
            "连续几卷出问题要启动 8D？",
            "麻点的密度标准是多少？",
            "最近氧化皮压入发生过几次，损失多少？",
            "裂纹能不能修磨后交货？",
            "给这卷划痕缺陷生成工单",
        ]
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "x.jpg").suffix or ".jpg"
    name = f"raw_{uuid.uuid4().hex[:10]}{suffix}"
    target = UPLOAD_DIR / name
    target.write_bytes(await file.read())
    image_id = register_image(target)
    return {"image_id": image_id, "url": f"/files/{name}", "filename": file.filename}


@app.post("/api/chat")
async def chat(payload: dict) -> StreamingResponse:
    question = (payload.get("question") or "").strip()
    image_id = payload.get("image_id")
    history = payload.get("history") or []
    if not question and not image_id:
        return JSONResponse(
            {"error": "question 与 image_id 至少提供一项"},
            status_code=400,
        )

    def event_stream():
        try:
            for ev in _runtime.run(question, image_id=image_id, history=history):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 任何异常都回传给前端，避免连接悬挂
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/eval")
def eval_report() -> dict:
    path = Path(__file__).resolve().parent.parent / "eval" / "report.json"
    if not path.exists():
        return {"available": False, "message": "尚未运行评测，执行 python -m eval.run_eval"}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}


@app.post("/api/index/rebuild")
def rebuild() -> dict:
    idx = get_index(rebuild=True)
    return {"ok": True, **idx.stats()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
