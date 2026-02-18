// ─── Configuration ──────────────────────────────────────────────────────────
// Replace with your Render API URL after deploying
const API_URL = window.API_URL || "https://pattern-delineation-api.onrender.com";

// ─── DOM refs ───────────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const spinner = $("#spinner");
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const fileName = $("#file-name");
const predictBtn = $("#predict-btn");
const thresholdSlider = $("#threshold");
const threshVal = $("#thresh-val");
const dotsSlider = $("#dots");
const dotsVal = $("#dots-val");
const jitterSlider = $("#jitter");
const jitterVal = $("#jitter-val");
const demoBtn = $("#demo-btn");

// ─── Helpers ────────────────────────────────────────────────────────────────
function showSpinner() { spinner.hidden = false; }
function hideSpinner() { spinner.hidden = true; }

function b64ToSrc(b64) {
  return `data:image/png;base64,${b64}`;
}

// ─── Slider labels ──────────────────────────────────────────────────────────
thresholdSlider.addEventListener("input", () => {
  threshVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
});
dotsSlider.addEventListener("input", () => {
  dotsVal.textContent = dotsSlider.value;
});
jitterSlider.addEventListener("input", () => {
  jitterVal.textContent = parseFloat(jitterSlider.value).toFixed(3);
});

// ─── File upload (click + drag-and-drop) ────────────────────────────────────
let selectedFile = null;

fileInput.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    fileName.textContent = selectedFile.name;
    predictBtn.disabled = false;
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  selectedFile = e.dataTransfer.files[0];
  if (selectedFile) {
    fileName.textContent = selectedFile.name;
    predictBtn.disabled = false;
  }
});

// ─── Predict (upload) ───────────────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  showSpinner();
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    const thresh = parseFloat(thresholdSlider.value);
    const res = await fetch(`${API_URL}/predict/json?threshold=${thresh}`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    // Show input preview
    const reader = new FileReader();
    reader.onload = () => { $("#res-input").src = reader.result; };
    reader.readAsDataURL(selectedFile);

    $("#res-mask").src = b64ToSrc(data.mask);
    $("#res-prob").src = b64ToSrc(data.probability);
    $("#upload-results").hidden = false;
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    hideSpinner();
  }
});

// ─── Demo (generate random pattern) ─────────────────────────────────────────
demoBtn.addEventListener("click", async () => {
  showSpinner();
  try {
    const dots = parseInt(dotsSlider.value);
    const jitter = parseFloat(jitterSlider.value);
    const res = await fetch(`${API_URL}/demo?num_dots=${dots}&jitter=${jitter}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    $("#demo-input").src = b64ToSrc(data.input);
    $("#demo-gt").src = b64ToSrc(data.ground_truth);
    $("#demo-mask").src = b64ToSrc(data.mask);
    $("#demo-prob").src = b64ToSrc(data.probability);
    $("#dice-score").textContent = `Dice Score: ${data.dice}`;
    $("#demo-results").hidden = false;
  } catch (err) {
    alert("Demo failed: " + err.message);
  } finally {
    hideSpinner();
  }
});
