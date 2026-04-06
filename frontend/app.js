// ═══════════════════════════════════════════
//  SENTINEL_AQI — Frontend Logic
//  Connects to FastAPI at http://127.0.0.1:8000
// ═══════════════════════════════════════════

const API_BASE = "http://127.0.0.1:8000";
let currentCity = "Delhi";
let autoRefreshTimer = null;

// ─── PAGE NAVIGATION ───
function switchPage(pageId) {
    document.querySelectorAll(".page-section").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".nav-link").forEach(n => {
        n.classList.remove("active");
        n.classList.add("text-on-surface/50");
    });

    const page = document.getElementById("page-" + pageId);
    const link = document.querySelector(`.nav-link[data-page="${pageId}"]`);
    if (page) page.classList.add("active");
    if (link) {
        link.classList.add("active");
        link.classList.remove("text-on-surface/50");
    }

    // Fetch data relevant to the page
    if (pageId === "dashboard" || pageId === "live" || pageId === "health") fetchCurrentStatus();
    if (pageId === "metrics") fetchModelMetrics();
    if (pageId === "history") fetchHistorical();
    if (pageId === "compare") fetchCityComparison();
    if (pageId === "alerts") loadAlertSettings();
}

// ─── CITY SELECTOR ───
function onCityChange() {
    currentCity = document.getElementById("citySelector").value;
    fetchCurrentStatus();
}

// ─── INITIALIZE ───
async function init() {
    try {
        const res = await fetch(`${API_BASE}/api/cities`);
        const data = await res.json();
        const sel = document.getElementById("citySelector");
        sel.innerHTML = data.cities.map(c =>
            `<option value="${c}" ${c === "Delhi" ? "selected" : ""}>${c}</option>`
        ).join("");
        currentCity = sel.value;

        setConnectionStatus(true);
        fetchCurrentStatus();

        // Auto-refresh every 60s
        autoRefreshTimer = setInterval(() => fetchCurrentStatus(), 60000);
    } catch (e) {
        setConnectionStatus(false);
        console.error("Init failed:", e);
    }
}

function setConnectionStatus(online) {
    const dot = document.getElementById("connDot");
    const label = document.getElementById("connLabel");
    if (online) {
        dot.className = "w-2 h-2 rounded-full bg-primary animate-pulse-glow";
        label.textContent = "API CONNECTED";
        label.className = "text-[10px] font-bold text-primary/80 uppercase tracking-widest";
    } else {
        dot.className = "w-2 h-2 rounded-full bg-secondary";
        label.textContent = "API OFFLINE";
        label.className = "text-[10px] font-bold text-secondary/80 uppercase tracking-widest";
    }
}

// ─── HELPER: Classify AQI for colors/labels ───
function classifyAqi(aqi) {
    if (aqi <= 50) return { level: "Good", color: "#00e400", badge: "Safe Range" };
    if (aqi <= 100) return { level: "Moderate", color: "#ffff00", badge: "Acceptable" };
    if (aqi <= 150) return { level: "Unhealthy (S)", color: "#ff7e00", badge: "Caution" };
    if (aqi <= 200) return { level: "Unhealthy", color: "#ff0000", badge: "Warning" };
    if (aqi <= 300) return { level: "Very Unhealthy", color: "#8f3f97", badge: "Danger" };
    return { level: "Hazardous", color: "#7e0023", badge: "Emergency" };
}

function pollutantStatus(val, low, high) {
    if (val <= low) return "OPTIMAL";
    if (val <= high) return "NORMAL";
    return "ELEVATED";
}

function addLogEntry(type, message) {
    const log = document.getElementById("processLog");
    const time = new Date().toLocaleTimeString("en-US", { hour12: false });
    const colorClass = type === "OK" ? "text-primary" : type === "WARN" ? "text-secondary" : "text-on-surface-variant";
    const bgClass = type === "WARN" ? "bg-secondary/5 border-l-2 border-secondary/40" : type === "OK" ? "bg-surface-container-highest/30 border-l-2 border-primary/40" : "";

    const entry = document.createElement("div");
    entry.className = `flex gap-4 p-2 ${bgClass} rounded`;
    entry.innerHTML = `<span class="text-primary/60">${time}</span><span class="${colorClass}">${message}</span>`;
    log.prepend(entry);

    // Keep only 15 entries
    while (log.children.length > 15) log.removeChild(log.lastChild);
}

// ═══════════════════════════════════════════
//  FETCH: Current Status (Dashboard + Live + Health)
// ═══════════════════════════════════════════
async function fetchCurrentStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/current-status?city=${currentCity}`);
        const data = await res.json();

        if (data.error) {
            addLogEntry("WARN", `ERROR: ${data.error}`);
            return;
        }

        const aqi = data.aqi || 0;
        const pm25 = data.pm25 || 0;
        const pm10 = data.pm10 || 0;
        const no2 = data.no2 || 0;
        const so2 = data.so2 || 0;
        const health = data.health_risk || {};
        const cls = classifyAqi(aqi);

        // ── Update Top Bar ──
        document.getElementById("sourceTag").textContent = `${data.source || "UNKNOWN"} — ${currentCity}`;
        document.getElementById("lastUpdated").textContent = `Updated: ${new Date().toLocaleTimeString()}`;

        // ── Dashboard Page ──
        document.getElementById("dashAqiValue").textContent = Math.round(aqi);
        document.getElementById("dashAqiLabel").textContent = cls.level;
        document.getElementById("dashAqiValue").style.color = cls.color;

        // AQI Arc (maps 0-500 to stroke-dashoffset 552-0)
        const offset = Math.max(0, 552.92 - (aqi / 500) * 552.92);
        document.getElementById("aqiArc").setAttribute("stroke-dashoffset", offset);
        document.getElementById("aqiArc").style.stroke = cls.color;

        document.getElementById("dashBadge").textContent = cls.badge;
        document.getElementById("dashBadge").style.color = cls.color;
        document.getElementById("dashUpdated").textContent = `Last updated: just now`;
        document.getElementById("dashAdvice").textContent = health.advice || cls.level;
        document.getElementById("dashHealthImpact").textContent = cls.level;
        document.getElementById("dashHealthImpact").style.color = cls.color;
        document.getElementById("dashSource").textContent = (data.source || "").split(" ")[0];
        document.getElementById("dashPm25").textContent = pm25.toFixed(1);
        document.getElementById("dashPm10").textContent = pm10.toFixed(1);
        document.getElementById("dashSo2").textContent = so2.toFixed(1);
        document.getElementById("dashNo2").textContent = no2.toFixed(1);
        document.getElementById("dashCity").textContent = currentCity;
        document.getElementById("dashRiskLevel").textContent = health.level || cls.level;
        document.getElementById("dashRiskLevel").style.color = health.color || cls.color;
        document.getElementById("riskTag").textContent = cls.badge;
        document.getElementById("riskTag").style.color = cls.color;
        document.getElementById("no2Status").textContent = pollutantStatus(no2, 40, 80);

        // ── Alert Banner (uses user threshold) ──
        const threshold = parseInt(localStorage.getItem('aqiThreshold') || '150');
        if (aqi > threshold) {
            document.getElementById("alertBanner").classList.remove("hidden");
            document.getElementById("alertText").textContent =
                `CRITICAL AIR QUALITY ALERT: ${currentCity.toUpperCase()} AQI ${Math.round(aqi)} — ${health.level || cls.level}`;
            triggerAlert(currentCity, Math.round(aqi), cls.level);
        } else {
            document.getElementById("alertBanner").classList.add("hidden");
        }

        // ── Live Monitor Page ──
        document.getElementById("livePm25").textContent = pm25.toFixed(1);
        document.getElementById("livePm10").textContent = pm10.toFixed(1);
        document.getElementById("liveNo2").textContent = no2.toFixed(1);
        document.getElementById("liveSo2").textContent = so2.toFixed(1);

        document.getElementById("livePm25Tag").textContent = pollutantStatus(pm25, 25, 60);
        document.getElementById("livePm10Tag").textContent = pollutantStatus(pm10, 50, 100);
        document.getElementById("liveNo2Tag").textContent = pollutantStatus(no2, 40, 80);
        document.getElementById("liveSo2Tag").textContent = pollutantStatus(so2, 20, 80);

        // Progress bars (scale to reasonable max)
        document.getElementById("livePm25Bar").style.width = Math.min(pm25 / 250 * 100, 100) + "%";
        document.getElementById("livePm10Bar").style.width = Math.min(pm10 / 500 * 100, 100) + "%";
        document.getElementById("liveNo2Bar").style.width = Math.min(no2 / 200 * 100, 100) + "%";
        document.getElementById("liveSo2Bar").style.width = Math.min(so2 / 200 * 100, 100) + "%";

        // Color bars based on level
        const barColor = (val, low, high) => val > high ? "bg-secondary" : val > low ? "bg-[#ffff00]" : "bg-primary";
        document.getElementById("livePm25Bar").className = `${barColor(pm25, 25, 60)} h-full transition-all duration-500`;
        document.getElementById("livePm10Bar").className = `${barColor(pm10, 50, 100)} h-full transition-all duration-500`;
        document.getElementById("liveNo2Bar").className = `${barColor(no2, 40, 80)} h-full transition-all duration-500`;
        document.getElementById("liveSo2Bar").className = `${barColor(so2, 20, 80)} h-full transition-all duration-500`;

        // ── Health Risk Page ──
        document.getElementById("healthLevel").textContent = health.level || cls.level;
        document.getElementById("healthLevel").style.color = health.color || cls.color;
        document.getElementById("healthAdvice").textContent = health.advice || "Air quality data retrieved successfully.";
        document.getElementById("healthMainCard").style.borderColor = health.color || cls.color;
        populatePrecautions(aqi);

        // ── Process Log ──
        addLogEntry("OK", `GET /api/current-status → AQI:${Math.round(aqi)} PM2.5:${pm25.toFixed(1)} SRC:${data.source}`);

        setConnectionStatus(true);
    } catch (e) {
        setConnectionStatus(false);
        addLogEntry("WARN", `FETCH FAILED: ${e.message}`);
    }
}

// ─── HEALTH PRECAUTIONS ───
function populatePrecautions(aqi) {
    const grid = document.getElementById("precautionGrid");
    let items = [];

    if (aqi <= 50) {
        items = [
            { icon: "park", title: "Outdoor Safe", desc: "All outdoor activities are safe. Enjoy the fresh air." },
            { icon: "self_improvement", title: "No Restrictions", desc: "No health precautions necessary." },
        ];
    } else if (aqi <= 100) {
        items = [
            { icon: "visibility", title: "Monitor Conditions", desc: "Unusually sensitive people should consider limiting prolonged outdoor exertion." },
            { icon: "air", title: "Ventilate Indoor", desc: "Open windows to maintain indoor air circulation." },
        ];
    } else if (aqi <= 200) {
        items = [
            { icon: "masks", title: "Wear N95 Mask", desc: "Use certified masks when outdoor exposure exceeds 30 min." },
            { icon: "child_care", title: "Protect Children", desc: "Keep children and elderly indoors during peak hours." },
            { icon: "local_hospital", title: "Monitor Symptoms", desc: "Watch for coughing, difficulty breathing, or eye irritation." },
            { icon: "home", title: "Air Purifier", desc: "Run indoor air purifiers and keep windows closed." },
        ];
    } else {
        items = [
            { icon: "emergency", title: "Stay Indoors", desc: "Avoid ALL outdoor activity. Seal windows and doors." },
            { icon: "masks", title: "N95 Mandatory", desc: "If going outside is unavoidable, use N95/P100 respirators." },
            { icon: "local_hospital", title: "Seek Medical Help", desc: "People with respiratory conditions should contact their doctor." },
            { icon: "water_drop", title: "Stay Hydrated", desc: "Drink plenty of water to help your body flush toxins." },
        ];
    }

    grid.innerHTML = items.map(i => `
        <div class="bg-surface-container-high p-5 rounded-xl border border-outline-variant/10 flex gap-4 items-start">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center flex-shrink-0">
                <span class="material-symbols-outlined text-secondary" style="font-variation-settings: 'FILL' 1;">${i.icon}</span>
            </div>
            <div>
                <h4 class="text-sm font-bold text-on-surface">${i.title}</h4>
                <p class="text-xs text-on-surface-variant mt-1">${i.desc}</p>
            </div>
        </div>
    `).join("");
}

// ═══════════════════════════════════════════
//  FETCH: AI Prediction (Forecast Page)
// ═══════════════════════════════════════════
async function runPrediction() {
    const btn = document.getElementById("runPredictBtn");
    btn.textContent = "⏳ Running LSTM Inference...";
    btn.disabled = true;

    document.getElementById("pred24").textContent = "⏳";
    document.getElementById("pred48").textContent = "⏳";
    document.getElementById("pred72").textContent = "⏳";
    document.getElementById("predStatus").textContent = "Querying model... This can take 30-60 seconds.";

    try {
        const res = await fetch(`${API_BASE}/api/predict?city=${currentCity}`, { method: "POST" });
        const data = await res.json();

        if (data.error) {
            document.getElementById("predStatus").textContent = `Error: ${data.error}`;
            addLogEntry("WARN", `PREDICT ERROR: ${data.error}`);
            return;
        }

        if (data.status === "MODEL_UNAVAILABLE") {
            document.getElementById("predStatus").textContent = `Model unavailable: ${data.error}`;
            return;
        }

        const p = data.predictions || {};
        const p24 = p["24h"]?.aqi || data.predicted_aqi_tomorrow || 0;
        const p48 = p["48h"]?.aqi || p24;
        const p72 = p["72h"]?.aqi || p48;

        const c24 = classifyAqi(p24);
        const c48 = classifyAqi(p48);
        const c72 = classifyAqi(p72);

        document.getElementById("pred24").textContent = p24.toFixed(1);
        document.getElementById("pred24").style.color = c24.color;
        document.getElementById("pred24Risk").textContent = c24.level;
        document.getElementById("pred24Risk").style.color = c24.color;

        document.getElementById("pred48").textContent = p48.toFixed(1);
        document.getElementById("pred48").style.color = c48.color;
        document.getElementById("pred48Risk").textContent = c48.level;
        document.getElementById("pred48Risk").style.color = c48.color;

        document.getElementById("pred72").textContent = p72.toFixed(1);
        document.getElementById("pred72").style.color = c72.color;
        document.getElementById("pred72Risk").textContent = c72.level;
        document.getElementById("pred72Risk").style.color = c72.color;

        document.getElementById("predSource").textContent = data.data_source || "HYBRID";
        document.getElementById("predStatus").textContent = `Prediction complete at ${new Date().toLocaleTimeString()}`;

        addLogEntry("OK", `POST /api/predict → 24h:${p24.toFixed(1)} 48h:${p48.toFixed(1)} 72h:${p72.toFixed(1)} [${data.data_source}]`);
    } catch (e) {
        document.getElementById("predStatus").textContent = `Connection failed: ${e.message}`;
        addLogEntry("WARN", `PREDICT FETCH FAILED: ${e.message}`);
    } finally {
        btn.textContent = "⚡ Run AI Prediction";
        btn.disabled = false;
    }
}

// ═══════════════════════════════════════════
//  FETCH: Model Metrics
// ═══════════════════════════════════════════
async function fetchModelMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/model-metrics`);
        const data = await res.json();

        // LSTM
        const lstm = data.lstm || {};
        const lm = lstm.metrics || {};
        document.getElementById("metricLstmMae").textContent = lm.mae ? parseFloat(lm.mae).toFixed(2) : "N/A";
        document.getElementById("metricLstmRmse").textContent = lm.rmse ? parseFloat(lm.rmse).toFixed(2) : "N/A";
        document.getElementById("metricLstmEpochs").textContent = lm.epochs_ran || "N/A";
        document.getElementById("metricLstmStatus").textContent = lstm.artifact_loaded ? "✅ Loaded" : "❌ " + (lstm.load_error || "Not loaded");

        // Classifier
        const cls = data.classifier || {};
        const cm = cls.metrics || {};
        document.getElementById("metricClsModel").textContent = cm.selected_model || cls.selected_model || "N/A";
        document.getElementById("metricClsF1").textContent = cm.selected_model_f1_weighted
            ? parseFloat(cm.selected_model_f1_weighted).toFixed(4)
            : "N/A";
        document.getElementById("metricClsStatus").textContent = cls.artifact_loaded
            ? "✅ Loaded"
            : cls.artifact_exists ? "⚠️ Not active" : "❌ Not found";

        // ARIMA
        const arima = data.arima_baseline || {};
        const am = arima.metrics || {};
        const agg = am.aggregate || {};
        document.getElementById("metricArimaMae").textContent = agg.mae_mean ? parseFloat(agg.mae_mean).toFixed(2) : "N/A";
        document.getElementById("metricArimaRmse").textContent = agg.rmse_mean ? parseFloat(agg.rmse_mean).toFixed(2) : "N/A";
        document.getElementById("metricArimaCities").textContent = am.cities_trained || "N/A";

        addLogEntry("OK", "GET /api/model-metrics → All metrics loaded");
    } catch (e) {
        addLogEntry("WARN", `METRICS FETCH FAILED: ${e.message}`);
    }
}

// ═══════════════════════════════════════════
//  FETCH: Export Report
// ═══════════════════════════════════════════
async function exportReport() {
    const result = document.getElementById("reportResult");
    result.textContent = "⏳ Generating report...";

    try {
        const res = await fetch(`${API_BASE}/api/report-summary/export?city=${currentCity}`, { method: "POST" });
        const data = await res.json();

        if (data.status === "saved") {
            result.innerHTML = `✅ Report saved: <span class="text-primary font-mono">${data.file}</span>`;
            addLogEntry("OK", `REPORT EXPORTED → ${data.file}`);
        } else {
            result.textContent = "❌ Export failed: " + JSON.stringify(data);
        }
    } catch (e) {
        result.textContent = "❌ Connection failed: " + e.message;
    }
}


// ═══════════════════════════════════════════
//  FEATURE: Historical Trends (Chart.js)
// ═══════════════════════════════════════════
let historyChartInstance = null;
let pollutantChartInstance = null;

async function fetchHistorical() {
    const days = document.getElementById('historyDays')?.value || 90;
    try {
        const res = await fetch(`${API_BASE}/api/historical?city=${currentCity}&days=${days}`);
        const data = await res.json();

        if (data.error || !data.data) {
            addLogEntry("WARN", `HISTORICAL ERROR: ${data.error}`);
            return;
        }

        const records = data.data;
        const labels = records.map(r => r.date);
        const aqiData = records.map(r => r.aqi);
        const pm25Data = records.map(r => r.pm25);
        const pm10Data = records.map(r => r.pm10);
        const no2Data = records.map(r => r.no2);

        // Destroy previous instances
        if (historyChartInstance) historyChartInstance.destroy();
        if (pollutantChartInstance) pollutantChartInstance.destroy();

        const chartOpts = {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#aaaab3', font: { family: 'Manrope', size: 11 } } },
            },
            scales: {
                x: { ticks: { color: '#74757d', maxTicksLimit: 12, font: { size: 10 } }, grid: { color: '#23262e' } },
                y: { ticks: { color: '#74757d', font: { size: 10 } }, grid: { color: '#23262e' } },
            },
        };

        // AQI Trend Line
        historyChartInstance = new Chart(document.getElementById('historyChart'), {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'AQI',
                    data: aqiData,
                    borderColor: '#ff716a',
                    backgroundColor: 'rgba(255,113,106,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1,
                    borderWidth: 2,
                }]
            },
            options: { ...chartOpts, plugins: { ...chartOpts.plugins, title: { display: true, text: `AQI Trend — ${currentCity} (${records.length} records)`, color: '#e5e4ed', font: { size: 14, family: 'Manrope', weight: 'bold' } } } },
        });

        // Pollutant Breakdown
        pollutantChartInstance = new Chart(document.getElementById('pollutantChart'), {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'PM2.5', data: pm25Data, borderColor: '#57fe81', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
                    { label: 'PM10', data: pm10Data, borderColor: '#52b9ff', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
                    { label: 'NO2', data: no2Data, borderColor: '#ffff00', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
                ]
            },
            options: chartOpts,
        });

        addLogEntry("OK", `GET /api/historical → ${records.length} records for ${currentCity}`);
    } catch (e) {
        addLogEntry("WARN", `HISTORICAL FETCH FAILED: ${e.message}`);
    }
}


// ═══════════════════════════════════════════
//  FEATURE: City Comparison
// ═══════════════════════════════════════════
let compareChartInstance = null;

async function fetchCityComparison() {
    const cities = "Delhi,Mumbai,Bangalore,Chennai,Kolkata";
    try {
        const res = await fetch(`${API_BASE}/api/compare?cities=${cities}`);
        const data = await res.json();

        if (!data.cities || data.cities.length === 0) {
            addLogEntry("WARN", "COMPARE: No city data returned");
            return;
        }

        const grid = document.getElementById('compareGrid');
        grid.innerHTML = data.cities.map(c => {
            if (c.error) return `<div class="bg-surface-container-low rounded-xl p-6 border border-outline-variant/10"><h3 class="text-lg font-bold text-on-surface">${c.city}</h3><p class="text-secondary">Data unavailable</p></div>`;
            const cls = classifyAqi(c.aqi);
            return `
                <div class="bg-surface-container-low rounded-xl p-6 border border-outline-variant/10 hover:border-[${cls.color}]/30 transition-all">
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <h3 class="text-lg font-bold text-on-surface">${c.city}</h3>
                            <span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">${c.source}</span>
                        </div>
                        <span class="text-3xl font-black" style="color:${cls.color}">${Math.round(c.aqi)}</span>
                    </div>
                    <div class="w-full h-2 bg-surface-container-highest rounded-full overflow-hidden mb-3">
                        <div class="h-full rounded-full transition-all" style="width:${Math.min(c.aqi/500*100,100)}%; background:${cls.color}"></div>
                    </div>
                    <p class="text-xs font-bold uppercase mb-3" style="color:${cls.color}">${cls.level}</p>
                    <div class="grid grid-cols-2 gap-2 text-[10px]">
                        <div class="bg-surface-container-highest/40 p-2 rounded"><span class="text-on-surface-variant">PM2.5</span><br><span class="font-bold text-on-surface">${c.pm25?.toFixed(1) || '--'}</span></div>
                        <div class="bg-surface-container-highest/40 p-2 rounded"><span class="text-on-surface-variant">PM10</span><br><span class="font-bold text-on-surface">${c.pm10?.toFixed(1) || '--'}</span></div>
                        <div class="bg-surface-container-highest/40 p-2 rounded"><span class="text-on-surface-variant">NO2</span><br><span class="font-bold text-on-surface">${c.no2?.toFixed(1) || '--'}</span></div>
                        <div class="bg-surface-container-highest/40 p-2 rounded"><span class="text-on-surface-variant">SO2</span><br><span class="font-bold text-on-surface">${c.so2?.toFixed(1) || '--'}</span></div>
                    </div>
                </div>
            `;
        }).join('');

        // Bar chart
        if (compareChartInstance) compareChartInstance.destroy();
        const validCities = data.cities.filter(c => !c.error);
        compareChartInstance = new Chart(document.getElementById('compareChart'), {
            type: 'bar',
            data: {
                labels: validCities.map(c => c.city),
                datasets: [{
                    label: 'AQI',
                    data: validCities.map(c => c.aqi),
                    backgroundColor: validCities.map(c => classifyAqi(c.aqi).color + '80'),
                    borderColor: validCities.map(c => classifyAqi(c.aqi).color),
                    borderWidth: 2,
                    borderRadius: 8,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#aaaab3', font: { family: 'Manrope', weight: 'bold' } }, grid: { display: false } },
                    y: { ticks: { color: '#74757d' }, grid: { color: '#23262e' }, beginAtZero: true },
                },
            },
        });

        addLogEntry("OK", `GET /api/compare → ${validCities.length} cities loaded`);
    } catch (e) {
        addLogEntry("WARN", `COMPARE FETCH FAILED: ${e.message}`);
    }
}


// ═══════════════════════════════════════════
//  FEATURE: User Alerts System
// ═══════════════════════════════════════════
function loadAlertSettings() {
    const threshold = localStorage.getItem('aqiThreshold') || '150';
    document.getElementById('alertThreshold').value = threshold;
    document.getElementById('currentThreshold').textContent = threshold;
}

function saveAlertThreshold() {
    const val = document.getElementById('alertThreshold').value;
    localStorage.setItem('aqiThreshold', val);
    document.getElementById('currentThreshold').textContent = val;
    document.getElementById('alertSaveMsg').innerHTML = `<span class="text-primary">✅ Threshold saved: AQI ${val}</span>`;
    setTimeout(() => document.getElementById('alertSaveMsg').textContent = '', 3000);
    addLogEntry("OK", `ALERT THRESHOLD SET → ${val}`);
}

function requestNotificationPermission() {
    if ('Notification' in window) {
        Notification.requestPermission().then(p => {
            document.getElementById('alertSaveMsg').innerHTML =
                p === 'granted'
                    ? '<span class="text-primary">✅ Browser notifications enabled!</span>'
                    : '<span class="text-secondary">⚠️ Notifications denied by browser.</span>';
        });
    } else {
        document.getElementById('alertSaveMsg').innerHTML = '<span class="text-secondary">Browser does not support notifications.</span>';
    }
}

function triggerAlert(city, aqi, level) {
    // Add to alert history
    const history = document.getElementById('alertHistory');
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = document.createElement('div');
    entry.className = 'flex gap-4 p-3 bg-secondary/5 border-l-2 border-secondary/40 rounded text-sm';
    entry.innerHTML = `<span class="text-secondary/60 font-mono">${time}</span><span class="text-secondary font-bold">${city} AQI ${aqi}</span><span class="text-on-surface-variant">— ${level}</span>`;
    if (history.querySelector('p')) history.innerHTML = '';
    history.prepend(entry);
    while (history.children.length > 20) history.removeChild(history.lastChild);

    // Browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('⚠️ SENTINEL AQI Alert', {
            body: `${city} AQI has reached ${aqi} (${level}). Take precautions.`,
            icon: '🔴',
        });
    }
}


// ═══════════════════════════════════════════
//  FEATURE: CSV Download
// ═══════════════════════════════════════════
function downloadCsv() {
    const days = document.getElementById('csvDays')?.value || 90;
    window.open(`${API_BASE}/api/export-csv?city=${currentCity}&days=${days}`, '_blank');
    addLogEntry("OK", `CSV DOWNLOAD → ${currentCity} ${days} days`);
}


// ─── BOOT ───
document.addEventListener("DOMContentLoaded", init);
