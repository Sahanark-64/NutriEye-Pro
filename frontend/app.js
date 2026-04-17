// ── Config ──────────────────────────────────────────────────────────────────
const API = "http://127.0.0.1:5000";

// Food emoji map for visual flair
const FOOD_EMOJI = {
  apple:"🍎", banana:"🍌", pizza:"🍕", burger:"🍔", orange:"🍊",
  rice:"🍚", egg:"🥚", bread:"🍞", chicken:"🍗", salad:"🥗",
  pasta:"🍝", sandwich:"🥪", sushi:"🍣", steak:"🥩", soup:"🍲",
  hotdog:"🌭", donut:"🍩", ice_cream:"🍦", french_fries:"🍟",
  chocolate_cake:"🎂", grapes:"🍇"
};

// ── State ────────────────────────────────────────────────────────────────────
let webcamStream = null;
let lastResult   = null;   // last nutrition result, used for saving to log

// ── Tab Switcher ─────────────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  if (btn) btn.classList.add("active");
  if (name !== "webcam") stopWebcam();
}

// ── File Upload ──────────────────────────────────────────────────────────────
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById("previewImg");
    img.src = ev.target.result;
    img.style.display = "block";
    document.getElementById("dropText").style.display = "none";
  };
  reader.readAsDataURL(file);
}

// Make drop zone clickable
document.addEventListener("DOMContentLoaded", () => {
  const dz = document.getElementById("dropZone");
  if (!dz) return;

  dz.addEventListener("click", () => document.getElementById("fileInput").click());

  dz.addEventListener("dragover", e => { e.preventDefault(); dz.style.borderColor = "#56ccf2"; });
  dz.addEventListener("dragleave", () => { dz.style.borderColor = ""; });
  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      document.getElementById("fileInput").files = e.dataTransfer.files;
      handleFileSelect({ target: { files: [file] } });
    }
  });

  // Dashboard: load log on page load
  if (document.getElementById("logBody")) renderDashboard();
});

async function analyzeUpload() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) { showToast("Please select an image first."); return; }

  const weightInput = document.getElementById("weightGrams");
  const weight = parseFloat(weightInput.value) || 100;

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  showLoading(true);
  try {
    const res  = await fetch(`${API}/predict`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { showToast("Error: " + data.error); return; }
    displayResults(scaleByWeight(data, weight), weight);
  } catch (err) {
    showToast("Cannot reach server. Is Flask running?");
    console.error(err);
  } finally {
    showLoading(false);
  }
}

// ── Webcam ───────────────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById("webcamVideo").srcObject = webcamStream;
    showToast("Camera started.");
  } catch (err) {
    showToast("Camera access denied or not available.");
    console.error(err);
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    const v = document.getElementById("webcamVideo");
    if (v) v.srcObject = null;
  }
}

async function captureAndAnalyze() {
  const video  = document.getElementById("webcamVideo");
  const canvas = document.getElementById("webcamCanvas");
  if (!webcamStream) { showToast("Start the camera first."); return; }

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const base64 = canvas.toDataURL("image/jpeg", 0.85);

  showLoading(true);
  try {
    const res  = await fetch(`${API}/webcam`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 })
    });
    const data = await res.json();
    if (data.error) { showToast("Error: " + data.error); return; }
    displayResults(data);
  } catch (err) {
    showToast("Cannot reach server. Is Flask running?");
    console.error(err);
  } finally {
    showLoading(false);
  }
}

// ── Charts ───────────────────────────────────────────────────────────────────
let donutChart = null;
let barChart   = null;

function renderCharts(protein, carbs, fat, calories) {
  // Destroy old charts if they exist
  if (donutChart) donutChart.destroy();
  if (barChart)   barChart.destroy();

  const donutCtx = document.getElementById("donutChart").getContext("2d");
  donutChart = new Chart(donutCtx, {
    type: "doughnut",
    data: {
      labels: ["Protein", "Carbs", "Fat"],
      datasets: [{
        data: [protein, carbs, fat],
        backgroundColor: ["#3498db", "#f39c12", "#2ecc71"],
        borderColor: "transparent",
        hoverOffset: 6
      }]
    },
    options: {
      plugins: {
        legend: { labels: { color: "#fff", font: { size: 11 } } }
      },
      cutout: "65%"
    }
  });

  const barCtx = document.getElementById("barChart").getContext("2d");
  barChart = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: ["Calories", "Protein", "Carbs", "Fat"],
      datasets: [
        {
          label: "This Food",
          data: [calories, protein, carbs, fat],
          backgroundColor: ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"],
          borderRadius: 6
        },
        {
          label: "Daily Goal",
          data: [2000, 50, 300, 65],
          backgroundColor: "rgba(255,255,255,0.1)",
          borderRadius: 6
        }
      ]
    },
    options: {
      plugins: {
        legend: { labels: { color: "#fff", font: { size: 10 } } }
      },
      scales: {
        x: { ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

// ── Scale nutrition by weight ─────────────────────────────────────────────
function scaleByWeight(data, weight) {
  const factor = weight / 100;  // CSV values are per 100g
  const scaled = { ...data };
  ["calories", "protein", "carbs", "fat"].forEach(key => {
    const val = parseFloat(data[key]);
    if (!isNaN(val)) scaled[key] = Math.round(val * factor * 10) / 10;
  });
  return scaled;
}

// ── Display Results ───────────────────────────────────────────────────────────
function displayResults(data, weight) {
  lastResult = data;

  const panel = document.getElementById("resultsPanel");
  panel.style.display = "block";

  const food = (data.food || "unknown").replace(/_/g, " ");
  document.getElementById("foodName").textContent = food;
  document.getElementById("foodEmoji").textContent = FOOD_EMOJI[data.food] || "🍽️";
  document.getElementById("confidenceBadge").textContent =
    data.confidence ? `${data.confidence}% confidence` : "Demo mode";
  // Show weight used
  const w = weight || 100;
  document.getElementById("confidenceBadge").textContent =
    (data.confidence ? `${data.confidence}% confidence` : "Demo mode") + ` · ${w}g`;

  document.getElementById("calories").textContent = data.calories ?? "N/A";
  document.getElementById("protein").textContent  = data.protein  ?? "N/A";
  document.getElementById("carbs").textContent    = data.carbs    ?? "N/A";
  document.getElementById("fat").textContent      = data.fat      ?? "N/A";
  document.getElementById("vitamins").textContent = data.vitamins ?? "N/A";

  // Calorie progress bar
  const cal = parseFloat(data.calories) || 0;
  const pct = Math.min((cal / 2000) * 100, 100).toFixed(1);
  document.getElementById("calorieBarFill").style.width = pct + "%";
  document.getElementById("caloriePercent").textContent = pct + "%";

  // Render charts
  renderCharts(
    parseFloat(data.protein)  || 0,
    parseFloat(data.carbs)    || 0,
    parseFloat(data.fat)      || 0,
    parseFloat(data.calories) || 0
  );

  // Smooth scroll to results
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── BMI Calculator ────────────────────────────────────────────────────────────
async function calcBMI() {
  const weight = document.getElementById("weight").value;
  const height = document.getElementById("height").value;
  if (!weight || !height) { showToast("Enter both weight and height."); return; }

  try {
    const res  = await fetch(`${API}/bmi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weight, height })
    });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }

    const resultDiv = document.getElementById("bmiResult");
    resultDiv.style.display = "block";
    document.getElementById("bmiValue").textContent    = data.bmi;
    const catEl = document.getElementById("bmiCategory");
    catEl.textContent  = data.category;
    catEl.style.color  = data.color;

    resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showToast("Cannot reach server. Is Flask running?");
  }
}

// ── Daily Log (localStorage) ──────────────────────────────────────────────────
function getLog() {
  return JSON.parse(localStorage.getItem("nutrieye_log") || "[]");
}

function saveLog(log) {
  localStorage.setItem("nutrieye_log", JSON.stringify(log));
}

function saveToLog() {
  if (!lastResult) { showToast("No result to save."); return; }
  const log = getLog();
  log.push({
    time:     new Date().toLocaleTimeString(),
    food:     lastResult.food,
    calories: lastResult.calories,
    protein:  lastResult.protein,
    carbs:    lastResult.carbs,
    fat:      lastResult.fat
  });
  saveLog(log);
  showToast("Saved to daily log!");
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
function renderDashboard() {
  const log = getLog();
  const tbody = document.getElementById("logBody");
  const table = document.getElementById("logTable");
  const empty = document.getElementById("emptyLog");

  if (!log.length) {
    table.style.display = "none";
    empty.style.display = "block";
    updateSummary(0, 0, 0, 0);
    return;
  }

  table.style.display = "table";
  empty.style.display = "none";
  tbody.innerHTML = "";

  let totCal = 0, totPro = 0, totCarb = 0, totFat = 0;

  log.forEach((entry, idx) => {
    totCal  += parseFloat(entry.calories) || 0;
    totPro  += parseFloat(entry.protein)  || 0;
    totCarb += parseFloat(entry.carbs)    || 0;
    totFat  += parseFloat(entry.fat)      || 0;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${entry.time}</td>
      <td>${(entry.food || "").replace(/_/g, " ")}</td>
      <td>${entry.calories}</td>
      <td>${entry.protein}g</td>
      <td>${entry.carbs}g</td>
      <td>${entry.fat}g</td>
      <td><button class="btn-danger" onclick="deleteEntry(${idx})">✕</button></td>
    `;
    tbody.appendChild(tr);
  });

  updateSummary(totCal, totPro, totCarb, totFat);
}

function updateSummary(cal, pro, carb, fat) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  const bar = (id, val, max) => {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min((val / max) * 100, 100) + "%";
  };

  set("totalCalories", Math.round(cal));
  set("totalProtein",  Math.round(pro)  + "g");
  set("totalCarbs",    Math.round(carb) + "g");
  set("totalFat",      Math.round(fat)  + "g");

  set("calProgress",  `${Math.round(cal)} / 2000 kcal`);
  set("proProgress",  `${Math.round(pro)} / 50g`);
  set("carbProgress", `${Math.round(carb)} / 300g`);
  set("fatProgress",  `${Math.round(fat)} / 65g`);

  bar("calBar",  cal,  2000);
  bar("proBar",  pro,  50);
  bar("carbBar", carb, 300);
  bar("fatBar",  fat,  65);
}

function deleteEntry(idx) {
  const log = getLog();
  log.splice(idx, 1);
  saveLog(log);
  renderDashboard();
  showToast("Entry removed.");
}

function clearLog() {
  if (!confirm("Clear all meal entries?")) return;
  saveLog([]);
  renderDashboard();
  showToast("Log cleared.");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showLoading(show) {
  const s = document.getElementById("loadingSpinner");
  if (s) s.style.display = show ? "block" : "none";
}

function showToast(msg) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}
