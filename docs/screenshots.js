/**
 * 生成 README 用的界面截图（开发辅助脚本，不参与服务运行）。
 *
 * 用本机 Edge/Chrome 跑真实交互：提问 → 等流式输出结束 → 截图；
 * 再上传一张缺陷图 → 等标注图出现 → 截图。截的是真实结果，不是画出来的示意。
 *
 * 用法：
 *   npm install                    # 需要 puppeteer-core
 *   python -m uvicorn app.main:app --port 8000
 *   node docs/screenshots.js
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const OUT = path.join(__dirname, "images");
const SAMPLE = path.join(__dirname, "..", "tests", "fixtures", "sample_defect.jpg");
const BASE = process.env.BASE_URL || "http://127.0.0.1:8000";

const CANDIDATES = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];

function findBrowser() {
  for (const p of CANDIDATES) if (fs.existsSync(p)) return p;
  throw new Error("找不到浏览器，请设置 CHROME_PATH 指向 chrome/msedge 可执行文件");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitIdle(page) {
  // 前端 send() 会置 busy=true，SSE 读完后复位；直接轮询最可靠
  for (let i = 0; i < 60; i++) {
    if (!(await page.evaluate(() => window.busy || document.querySelector("#btnSend")?.disabled))) return;
    await sleep(300);
  }
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || findBrowser(),
    headless: "new",
    args: ["--hide-scrollbars", "--disable-gpu", "--force-device-scale-factor=2"],
    defaultViewport: { width: 1360, height: 900, deviceScaleFactor: 2 },
  });
  const page = await browser.newPage();
  await page.goto(BASE, { waitUntil: "networkidle2" });
  await sleep(800);

  // 1) 首屏
  await page.screenshot({ path: path.join(OUT, "ui-home.png") });

  // 2) 文本问答：问一个有明确数值的问题，能看到工具轨迹与引用
  await page.click("#input");
  await page.keyboard.type("除鳞水压低于多少会出氧化皮压入？", { delay: 10 });
  await page.click("#btnSend");
  await waitIdle(page);
  await sleep(300);
  await page.screenshot({ path: path.join(OUT, "ui-chat.png") });

  // 3) 图像问答：上传样例图，等标注图渲染出来
  const input = await page.$("#file");
  await input.uploadFile(SAMPLE);
  await sleep(1200);
  await page.click("#input");
  await page.keyboard.type("这张图有什么缺陷，判几级？", { delay: 10 });
  await page.click("#btnSend");
  await waitIdle(page);
  await sleep(800);
  await page.screenshot({ path: path.join(OUT, "ui-vision.png") });

  await browser.close();
  console.log("已生成：", fs.readdirSync(OUT).join(", "));
})().catch((e) => {
  console.error("截图失败：", e.message);
  process.exit(1);
});
