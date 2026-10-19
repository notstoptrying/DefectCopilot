const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("btnSend");
const fileEl = $("file");

let attached = null;   // {image_id, url, name}
let answerBuf = "";   // 当前回答的累积文本
let busy = false;
const history = [];    // 仅保留文本轮次，供多轮上下文

/* ---------------- 初始化 ---------------- */
async function init() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    $("statusDot").className = "dot on";
    const mode = d.llm.provider === "mock" ? "离线规则模型（无需 Key）" : `${d.llm.model}`;
    $("llmMode").textContent = mode;
    $("indexStat").textContent = `${d.index.docs} 篇文档 · ${d.index.chunks} 个知识块 · 词表 ${d.index.vocab}`;
  } catch (e) {
    $("statusDot").className = "dot err";
    $("llmMode").textContent = "服务未就绪";
  }

  try {
    const r = await fetch("/api/samples");
    const d = await r.json();
    const box = $("chips");
    box.innerHTML = "";
    d.items.forEach((q) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = q;
      b.onclick = () => { inputEl.value = q; send(); };
      box.appendChild(b);
    });
  } catch (e) { /* 忽略示例加载失败 */ }
}

/* ---------------- 渲染工具 ---------------- */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inlineMd(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+)\]/g, '<span class="cite">$1</span>');
}

function md(text) {
  const lines = esc(text).split("\n");
  let html = "";
  let inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }
    const h = line.match(/^\*\*(.+)\*\*$/);
    if (h) { closeList(); html += `<h4>${inlineMd(h[1])}</h4>`; continue; }
    if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inlineMd(line.replace(/^[-*]\s+/, ""))}</li>`;
      continue;
    }
    closeList();
    html += `<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return html;
}

function addUser(text, imageUrl) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text) || "（图片）"}${
    imageUrl ? `<div class="msg-image"><img src="${imageUrl}" alt="uploaded" /></div>` : ""
  }</div>`;
  messagesEl.appendChild(el);
  scrollBottom();
}

function addBot() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="avatar">AI</div><div class="bubble"><div class="body">
      <div class="msg-images"></div>
      <div class="msg-text"><span class="typing"><i></i><i></i><i></i></span></div>
    </div></div>`;
  messagesEl.appendChild(el);
  scrollBottom();
  // 图片与文本分开两个容器：流式 token 只重写 msg-text，
  // 否则每来一个 token 都会把刚插入的标注图冲掉（这正是「只返结论、不返标红图」的原因）
  return { text: el.querySelector(".msg-text"), images: el.querySelector(".msg-images") };
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------------- 工具轨迹 ---------------- */
const toolNames = {
  search_knowledge: "知识库检索",
  query_case_db: "历史工单统计",
  analyze_defect_image: "缺陷图像分析",
  create_ticket: "生成质量工单",
};
let traceCount = 0;

function resetTrace() {
  $("trace").innerHTML = "";
  $("citations").innerHTML = "";
  $("citePill").textContent = "0 条";
  $("stepPill").textContent = "0 步";
  traceCount = 0;
}

function pushTraceCall(ev) {
  traceCount += 1;
  $("stepPill").textContent = `${traceCount} 步`;
  const item = document.createElement("div");
  item.className = "trace-item";
  item.id = `tr-${ev.id}`;
  const argTxt = Object.entries(ev.args || {}).map(([k, v]) => `${k}=${v}`).join(" · ") || "—";
  item.innerHTML = `<div class="idx">${traceCount}</div>
    <div class="trace-body">
      <div class="trace-name">${toolNames[ev.name] || ev.name}<span class="trace-time">调用中…</span></div>
      <div class="trace-args">${esc(argTxt)}</div>
    </div>`;
  $("trace").appendChild(item);
  $("trace").scrollTop = $("trace").scrollHeight;
}

function completeTrace(ev) {
  const item = $(`tr-${ev.id}`);
  if (!item) return;
  const time = item.querySelector(".trace-time");
  time.textContent = `${ev.latency_ms}ms`;
  const body = item.querySelector(".trace-body");
  const res = document.createElement("div");
  res.className = "trace-res";
  res.innerHTML = `<span class="${ev.ok ? "badge-ok" : "badge-err"}">${ev.ok ? "✓" : "✕"}</span> ${esc(ev.summary)}`;
  body.appendChild(res);
}

/* ---------------- 引用 ---------------- */
function renderCitations(items) {
  const box = $("citations");
  if (!items || !items.length) {
    box.innerHTML = `<div class="empty">本轮无检索结果</div>`;
    return;
  }
  $("citePill").textContent = `${items.length} 条`;
  box.innerHTML = items
    .map(
      (c) => `<div class="cite-card">
        <div class="top"><span class="cite-num">${c.idx}</span>
          <span class="cite-doc">${esc(c.doc_title)} · ${esc(c.heading)}</span></div>
        <div class="cite-snip">${esc((c.text || "").replace(/^【[^】]*】\n?/, ""))}</div>
      </div>`
    )
    .join("");
}

/* ---------------- 发送与流式接收 ---------------- */
async function send() {
  const text = inputEl.value.trim();
  if ((!text && !attached) || busy) return;

  busy = true;
  sendBtn.disabled = true;
  inputEl.value = "";
  autoGrow();

  addUser(text, attached ? attached.url : null);
  resetTrace();
  $("metrics").innerHTML = `<span class="muted small">计算中…</span>`;

  const ctx = addBot();
  answerBuf = "";

  const payload = {
    question: text,
    image_id: attached ? attached.image_id : null,
    history: history.slice(-6),
  };
  clearAttach();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let raw = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      raw += decoder.decode(value, { stream: true });
      const parts = raw.split("\n\n");
      raw = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(data); } catch (e) { continue; }
        handleEvent(ev, ctx);
      }
    }
    history.push({ role: "user", content: text });
    if (answerBuf.trim()) history.push({ role: "assistant", content: answerBuf });
  } catch (e) {
    ctx.text.innerHTML = `<p style="color:#dc2626">请求失败：${esc(String(e))}</p>`;
  } finally {
    busy = false;
    sendBtn.disabled = false;
    scrollBottom();
  }
}

function handleEvent(ev, ctx) {
  const body = ctx.text;
  switch (ev.type) {
    case "token": {
      answerBuf += ev.text;
      body.innerHTML = md(answerBuf);
      scrollBottom();
      break;
    }
    case "tool_call":
      pushTraceCall(ev);
      break;
    case "tool_result":
      completeTrace(ev);
      break;
    case "citations":
      renderCitations(ev.items);
      break;
    case "image": {
      const d = ev.data || {};
      if (!d.annotated_url) break;
      const wrap = document.createElement("div");
      wrap.className = "msg-image";
      wrap.innerHTML = `<a href="${d.annotated_url}" target="_blank" rel="noopener" title="点击查看原图">
          <img src="${d.annotated_url}" alt="缺陷标红标注图" /></a>
        <div class="cap">彩色区域按缺陷类别着色（图例在图片下方 · ${(d.boxes || []).length} 处外接框） · 局部视场占比 ${(
        (d.area_ratio || 0) * 100
      ).toFixed(2)}% · ${esc(d.level || "")} 级 · ${esc(d.model || "")}</div>`;
      ctx.images.appendChild(wrap);
      ctx.images.scrollIntoView({ block: "nearest", behavior: "smooth" });
      break;
    }
    case "error":
      body.innerHTML = `<p style="color:#dc2626">${esc(ev.message)}</p>`;
      break;
    case "done": {
      const m = $("metrics");
      m.innerHTML = `
        <div class="metric"><div class="v">${ev.elapsed_ms}ms</div><div class="k">端到端耗时</div></div>
        <div class="metric"><div class="v">${ev.steps}</div><div class="k">推理轮次</div></div>
        <div class="metric"><div class="v">${ev.tools}</div><div class="k">工具调用</div></div>`;
      break;
    }
  }
}

/* ---------------- 图片上传 ---------------- */
$("btnUpload").onclick = () => fileEl.click();
fileEl.onchange = async () => {
  const f = fileEl.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const d = await r.json();
    attached = { image_id: d.image_id, url: d.url, name: f.name };
    $("attachThumb").src = d.url;
    $("attachName").textContent = f.name;
    $("attach").hidden = false;
  } catch (e) {
    alert("上传失败：" + e);
  } finally {
    fileEl.value = "";
  }
};
$("attachClear").onclick = clearAttach;

function clearAttach() {
  attached = null;
  $("attach").hidden = true;
}

/* ---------------- 输入框 ---------------- */
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 132) + "px";
}
inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
sendBtn.onclick = send;

init();
