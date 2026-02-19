/* ═══════════════════════════════════════════════════════════════════════════
   Pattern Delineation — Neural Analysis Lab
   Frontend application logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Configuration ───────────────────────────────────────────────────────────
const API_URL =
  window.API_URL ||
  "https://pattern-delineation-production.up.railway.app";

const FETCH_TIMEOUT = 120_000; // 120 s

// ── DOM refs ────────────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);

const statusDot      = $("#status-dot");
const statusLabel    = $("#status-label");
const footerDevice   = $("#footer-device");

const processingBox  = $("#processing-box");

const thresholdSlider = $("#threshold");
const threshVal      = $("#thresh-val");

const dotsSlider     = $("#dots");
const dotsVal        = $("#dots-val");
const jitterSlider   = $("#jitter");
const jitterVal      = $("#jitter-val");
const demoBtn        = $("#demo-btn");

const heatmapToggle  = $("#heatmap-toggle");
const downloadBtn    = $("#download-btn");

const viewerResults     = $("#viewer-results");
const canvasInput       = $("#canvas-input");
const canvasMask        = $("#canvas-mask");
const canvasHeatmap     = $("#canvas-heatmap");
const maskLabel         = $("#mask-label");

const paneEmpties    = document.querySelectorAll(".pane-empty");

const gtRow          = $("#gt-row");
const canvasGt       = $("#canvas-gt");
const diceBadge      = $("#dice-badge");
const diceValue      = $("#dice-value");

const historyList    = $("#history-list");
const historyEmpty   = $("#history-empty");

// ── State ───────────────────────────────────────────────────────────────────
let heatmapVisible   = false;
let probImageData    = null;   // raw probability PNG data for heatmap
let runHistory       = [];     // { dice, dots, jitter, timestamp, thumbB64 }
let hasResults       = false;  // tracks whether canvases have content

// ── Helpers ─────────────────────────────────────────────────────────────────
function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = `toast ${type ? "toast--" + type : ""}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

async function fetchWithTimeout(url, opts = {}, timeout = FETCH_TIMEOUT) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    clearTimeout(id);
    return res;
  } catch (e) {
    clearTimeout(id);
    if (e.name === "AbortError") throw new Error("Request timed out");
    throw e;
  }
}

function showProcessing(show) {
  processingBox.hidden = !show;
  demoBtn.disabled     = show;
}

// ── Draw base64 PNG onto a canvas ───────────────────────────────────────────
function drawB64(canvas, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      canvas.width  = img.width;
      canvas.height = img.height;
      canvas.getContext("2d").drawImage(img, 0, 0);
      resolve(img);
    };
    img.src = "data:image/png;base64," + b64;
  });
}

// ── Heatmap colorization ────────────────────────────────────────────────────
// Takes a grayscale probability map and draws a cyan→magenta heatmap
function drawHeatmap(canvas, b64Prob) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      canvas.width  = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = imageData.data;

      for (let i = 0; i < d.length; i += 4) {
        const v = d[i] / 255;           // prob 0→1

        // Gradient: black → cyan → white
        let r, g, b;
        if (v < 0.5) {
          const t = v * 2;
          r = 0;
          g = Math.round(t * 224);
          b = Math.round(t * 255);
        } else {
          const t = (v - 0.5) * 2;
          r = Math.round(t * 255);
          g = Math.round(224 + t * 31);
          b = 255;
        }

        d[i]     = r;
        d[i + 1] = g;
        d[i + 2] = b;
        d[i + 3] = Math.round(v * 220); // alpha based on confidence
      }
      ctx.putImageData(imageData, 0, 0);
      resolve();
    };
    img.src = "data:image/png;base64," + b64Prob;
  });
}

// ── Animate metric rings on load ────────────────────────────────────────────
function animateRings() {
  const circ = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--ring-circumference")
  );
  document.querySelectorAll(".metric__ring-fill").forEach((el) => {
    const pct = parseFloat(el.dataset.pct) / 100;
    el.style.strokeDashoffset = circ * (1 - pct);
  });
}

// ── Health check ────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetchWithTimeout(`${API_URL}/health`, {}, 10_000);
    const j = await res.json();
    statusDot.className   = "status-dot online";
    statusLabel.textContent = "Model Online";
    footerDevice.textContent = j.device || "cpu";
  } catch {
    statusDot.className   = "status-dot error";
    statusLabel.textContent = "Offline";
  }
}

// ── Show results ────────────────────────────────────────────────────────────
function showResults() {
  paneEmpties.forEach((el) => (el.hidden = true));
  hasResults = true;
  downloadBtn.disabled = false;
}

// ── Demo ────────────────────────────────────────────────────────────────────
async function demo() {
  showProcessing(true);
  try {
    const dots   = dotsSlider.value;
    const jitter = jitterSlider.value;

    const res = await fetchWithTimeout(
      `${API_URL}/demo?num_dots=${dots}&jitter=${jitter}`
    );
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const j = await res.json();

    showResults();
    gtRow.hidden = false;

    await Promise.all([
      drawB64(canvasInput, j.input),
      drawB64(canvasMask,  j.mask),
      drawB64(canvasGt,    j.ground_truth),
      drawHeatmap(canvasHeatmap, j.probability),
    ]);

    canvasHeatmap.hidden = !heatmapVisible;
    probImageData = j.probability;

    // Dice score
    diceValue.textContent = j.dice.toFixed(4);

    // Color code dice
    if (j.dice >= 0.85) {
      diceValue.style.color = "var(--success)";
      diceValue.style.textShadow = "0 0 16px rgba(34,197,94,0.4)";
    } else if (j.dice >= 0.6) {
      diceValue.style.color = "var(--warning)";
      diceValue.style.textShadow = "0 0 16px rgba(245,158,11,0.4)";
    } else {
      diceValue.style.color = "var(--danger)";
      diceValue.style.textShadow = "0 0 16px rgba(239,68,68,0.4)";
    }

    toast(`Demo complete — Dice: ${j.dice.toFixed(4)}`, "success");

    // ── Add to run history ──
    addHistoryItem(j.dice, dots, jitter, j.input);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    showProcessing(false);
  }
}

// ── Run History ─────────────────────────────────────────────────────────────
function addHistoryItem(dice, dots, jitter, thumbB64) {
  const entry = {
    dice,
    dots,
    jitter,
    timestamp: new Date(),
    thumbB64,
  };
  runHistory.unshift(entry);
  renderHistory();
}

function renderHistory() {
  if (runHistory.length === 0) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;
  historyList.innerHTML = "";

  runHistory.forEach((entry, i) => {
    const el = document.createElement("div");
    el.className = "history-item";

    const num = document.createElement("span");
    num.className = "history-item__num";
    num.textContent = `#${runHistory.length - i}`;

    const thumb = document.createElement("span");
    thumb.className = "history-item__thumb";
    const tc = document.createElement("canvas");
    thumb.appendChild(tc);
    // Draw thumbnail
    const img = new Image();
    img.onload = () => {
      tc.width = img.width;
      tc.height = img.height;
      tc.getContext("2d").drawImage(img, 0, 0);
    };
    img.src = "data:image/png;base64," + entry.thumbB64;

    const diceSpan = document.createElement("span");
    diceSpan.className = "history-item__dice";
    diceSpan.textContent = entry.dice.toFixed(4);
    if (entry.dice >= 0.85) diceSpan.style.color = "var(--success)";
    else if (entry.dice >= 0.6) diceSpan.style.color = "var(--warning)";
    else diceSpan.style.color = "var(--danger)";

    const params = document.createElement("span");
    params.className = "history-item__params";
    params.textContent = `${entry.dots} dots · jitter ${parseFloat(entry.jitter).toFixed(3)}`;

    const time = document.createElement("span");
    time.className = "history-item__time";
    time.textContent = entry.timestamp.toLocaleTimeString();

    el.append(num, thumb, diceSpan, params, time);
    historyList.appendChild(el);
  });
}

// ── Download composite ──────────────────────────────────────────────────────
function downloadResults() {
  if (!hasResults) return;

  const w = canvasInput.width;
  const h = canvasInput.height;
  const gap = 8;
  const cols = 3;
  const totalW = w * cols + gap * (cols - 1);
  const totalH = h + 28;

  const offscreen = document.createElement("canvas");
  offscreen.width = totalW;
  offscreen.height = totalH;
  const ctx = offscreen.getContext("2d");

  ctx.fillStyle = "#06080d";
  ctx.fillRect(0, 0, totalW, totalH);

  // Labels
  ctx.fillStyle = "#7a8ba5";
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText("Input", 0, 12);
  ctx.fillText("Prediction", w + gap, 12);
  ctx.fillText("Ground Truth", (w + gap) * 2, 12);

  const yOff = 20;
  ctx.drawImage(canvasInput, 0, yOff, w, h);
  ctx.drawImage(canvasMask,  w + gap, yOff, w, h);
  ctx.drawImage(canvasGt,    (w + gap) * 2, yOff, w, h);

  const link = document.createElement("a");
  link.download = `pattern-delineation-${Date.now()}.png`;
  link.href = offscreen.toDataURL("image/png");
  link.click();
}

// ── Event wiring ────────────────────────────────────────────────────────────

// Slider live labels
thresholdSlider.addEventListener("input", () => {
  threshVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
});
dotsSlider.addEventListener("input", () => {
  dotsVal.textContent = dotsSlider.value;
});
jitterSlider.addEventListener("input", () => {
  jitterVal.textContent = parseFloat(jitterSlider.value).toFixed(3);
});

// Demo button
demoBtn.addEventListener("click", demo);

// Heatmap toggle
heatmapToggle.addEventListener("click", () => {
  heatmapVisible = !heatmapVisible;
  heatmapToggle.classList.toggle("active", heatmapVisible);
  canvasHeatmap.hidden = !heatmapVisible;
  maskLabel.textContent = heatmapVisible
    ? "Confidence Heatmap"
    : "AI Segmentation Mask";
});

// Download button
downloadBtn.addEventListener("click", downloadResults);

// ── Init ────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  animateRings();
  checkHealth();
});
