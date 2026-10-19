/**
 * 前端流式渲染回归测试（jsdom，无需浏览器）
 *
 * 背景：早期实现把「标注图」和「回答正文」放在同一个容器里，
 * 每个 token 都会 body.innerHTML = md(answerBuf)，把刚插入的
 * 标注图整块冲掉——表现就是「只返结论，不返标红图」。
 *
 * 本测试用假的 SSE 流按真实事件顺序（tool_call → image → token…）
 * 驱动 web/app.js，断言流式结束后标注图仍在 DOM 中。
 *
 * 运行：node tests/ui_stream_test.js
 */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "web/index.html"), "utf8");
// 可通过 APP_JS 指向旧版脚本，用于验证本测试能复现「图片被流式覆盖」的问题
const appJs = fs.readFileSync(process.env.APP_JS || path.join(root, "web/app.js"), "utf8");

const dom = new JSDOM(html, { url: "http://localhost:8000/", runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;
const doc = window.document;

window.Element.prototype.scrollIntoView = function () {};
window.TextDecoder = TextDecoder;
window.ReadableStream = ReadableStream;

const IMAGE_DATA = {
  model: "UNet(NEU-Seg)(ep49)",
  defect_type: "条带状夹杂",
  class_id: 1,
  area_ratio: 0.0721,
  level: "C",
  confidence: 0.9238,
  boxes: [
    { x1: 28, y1: 123, x2: 52, y2: 199, area: 1462, label: "条带状夹杂", level: "C" },
    { x1: 173, y1: 41, x2: 183, y2: 100, area: 433, label: "条带状夹杂", level: "C" },
  ],
  annotated_url: "/files/annotated_demo.png",
};

const ANSWER = "**结论**\n图像检出 2 处条带状夹杂，面积占板面 7.21%，建议定为 C 级。";

function sse(events) {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const e of events) {
        controller.enqueue(enc.encode(`data: ${JSON.stringify(e)}\n\n`));
      }
      controller.enqueue(enc.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
}

// 按真实顺序切分 token（模拟逐字流式），用于复现「图片被覆盖」
const tokens = [];
for (let i = 0; i < ANSWER.length; i += 4) {
  tokens.push({ type: "token", text: ANSWER.slice(i, i + 4) });
}

const events = [
  { type: "tool_call", id: "c1", name: "analyze_defect_image", args: { image_id: "abc123" } },
  { type: "tool_result", id: "c1", ok: true, summary: "检出条带状夹杂，面积占比 7.21%，建议 C 级", latency_ms: 30 },
  { type: "image", data: IMAGE_DATA },
  ...tokens,
  { type: "done", elapsed_ms: 16, steps: 1, tools: 1 },
];

window.fetch = async (url, opts) => {
  if (String(url).includes("/api/chat")) {
    return { ok: true, body: sse(events) };
  }
  if (String(url).includes("/api/samples")) {
    return { ok: true, json: async () => ({ items: [] }) };
  }
  return {
    ok: true,
    json: async () => ({ status: "ok", llm: { provider: "mock", model: "offline-rule-engine" }, index: { docs: 8, chunks: 59, vocab: 2893 } }),
  };
};

window.eval(appJs);

(async () => {
  await new Promise((r) => setTimeout(r, 60));
  doc.getElementById("input").value = "这张图有什么缺陷，判几级？";
  await window.send();
  await new Promise((r) => setTimeout(r, 120));

  const imgs = doc.querySelectorAll(".msg-images img");
  const text = doc.querySelector(".msg-text");
  const textContent = (text ? text.textContent : "").replace(/\s+/g, "");

  const checks = [
    ["标注图仍在 DOM 中", imgs.length === 1],
    ["标注图 src 正确", imgs.length === 1 && imgs[0].getAttribute("src") === IMAGE_DATA.annotated_url],
    ["图片说明含面积与等级", doc.querySelector(".msg-image .cap")?.textContent.includes("7.21%")],
    ["正文结论完整保留", textContent.includes("C级") && textContent.includes("条带状夹杂")],
    ["正文未被图片槽污染", !textContent.includes("红色区域为模型判定的缺陷")],
    ["工具轨迹已渲染", doc.querySelectorAll(".trace-item").length === 1],
  ];

  let failed = 0;
  for (const [name, ok] of checks) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
    if (!ok) failed += 1;
  }
  console.log(failed === 0 ? "\n全部通过" : `\n${failed} 项失败`);
  process.exit(failed === 0 ? 0 : 1);
})();
