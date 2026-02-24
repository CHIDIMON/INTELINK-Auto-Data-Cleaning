// ==========================================
// 1. CONFIGURATION & STATE
// ==========================================
const NGROK_URL = "";

function getApiUrl() {
    let url = NGROK_URL.trim();
    if (url === "" || url.includes("your-ngrok-url")) {
        return `http://127.0.0.1:8000`; // บังคับใช้ 127.0.0.1 เสมอเพื่อเลี่ยงปัญหา CORS กับ localhost
    }
    return url.endsWith('/') ? url.slice(0, -1) : url;
}

// ==========================================
// ⏳ ระบบเช็ค SESSION TIMEOUT (3 ชั่วโมง)
// ==========================================
const SESSION_LIMIT_MS = 3 * 60 * 60 * 1000; // 3 ชั่วโมง (หน่วยเป็นมิลลิวินาที)
let loginTimestamp = localStorage.getItem('login_timestamp');

if (loginTimestamp) {
    let timeElapsed = Date.now() - parseInt(loginTimestamp);
    if (timeElapsed > SESSION_LIMIT_MS) {
        localStorage.removeItem('username');
        localStorage.removeItem('plan');
        localStorage.removeItem('login_timestamp');

        if (!window.location.pathname.endsWith('login.html')) {
            alert("เซสชันของคุณหมดอายุแล้ว (เกิน 3 ชั่วโมง) กรุณาล็อกอินใหม่เพื่อความปลอดภัย");
            window.location.href = "login.html";
        }
    }
}

// เช็คสถานะการล็อกอินจาก LocalStorage ปัจจุบัน
let currentUser = localStorage.getItem('username') || '';
let userPlan = localStorage.getItem('plan') || 'free';
let isLoggedIn = (currentUser !== '' && currentUser !== 'Guest');

// ตัวแปรสำหรับ Index (Clean & Train)
let currentRawFile = "";
let currentCleanFile = "";
let datasetColumns = [];

// ตัวแปรสำหรับ Predict (Manual Form)
let globalHeaders = [];

// ==========================================
// 2. PAGE ROUTER (รันโค้ดตามหน้าเว็บที่เปิด)
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    const currentPage = document.body.dataset.page;
    updateUIBasedOnAuth(); // อัปเดตเมนูซ้ายมือทุกหน้า

    // รันโค้ดเฉพาะหน้า
    if (currentPage === "index") {
        initIndexPage();
    } else if (currentPage === "predict") {
        // initPredictPage(); (ถ้ามีฟังก์ชันเริ่มต้นตอนเปิดหน้า Predict)
    }
});

// ==========================================
// 3. GLOBAL FUNCTIONS (ใช้ได้ทุกหน้า)
// ==========================================
function updateUIBasedOnAuth() {
    const guestView = document.getElementById('authGuest');
    const memberView = document.getElementById('authMember');
    const historySection = document.getElementById('historySection');
    const guestMessage = document.getElementById('guestMessage');
    const planBadge = document.getElementById('planBadge');
    const userNameDisplay = document.getElementById('userNameDisplay');

    if (isLoggedIn) {
        if (guestView) guestView.classList.add('d-none');
        if (memberView) memberView.classList.remove('d-none');
        if (historySection) historySection.classList.remove('d-none');
        if (guestMessage) guestMessage.classList.add('d-none');
        if (userNameDisplay) userNameDisplay.innerText = currentUser;

        if (planBadge) {
            if (userPlan === 'pro') {
                planBadge.className = "badge pro-badge rounded-pill";
                planBadge.innerHTML = "<i class='bi bi-star-fill me-1'></i>PRO Plan";
            } else {
                planBadge.className = "badge bg-info text-dark rounded-pill";
                planBadge.innerText = "Free Plan";
            }
        }
    } else {
        if (guestView) guestView.classList.remove('d-none');
        if (memberView) memberView.classList.add('d-none');
        if (historySection) historySection.classList.add('d-none');
        if (guestMessage) guestMessage.classList.remove('d-none');

        if (planBadge) {
            planBadge.className = "badge bg-secondary rounded-pill";
            planBadge.innerText = "Guest Mode";
        }
    }
}

function logout() {
    localStorage.removeItem('username');
    localStorage.removeItem('plan');
    localStorage.removeItem('login_timestamp');
    window.location.href = "index.html";
}

async function downloadFile(url, filename) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("Download failed");
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(a);
    } catch (err) {
        alert("Download Error: " + err.message);
    }
}

function copyPythonCode() {
    const codeText = document.getElementById('pythonCodeSnippet');
    if (codeText) {
        navigator.clipboard.writeText(codeText.innerText).then(() => {
            alert("✅ Copied to clipboard!");
        });
    }
}

// ==========================================
// 4. AUTHENTICATION (Login / Register)
// ==========================================
async function handleLogin(e) {
    e.preventDefault();
    const btn = document.querySelector('.btn-login');
    const originalText = btn.innerText;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Checking...';
    btn.disabled = true;

    const formData = {
        email: document.getElementById('emailInput').value,
        password: document.getElementById('passwordInput').value
    };

    try {
        const res = await fetch(`${getApiUrl()}/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });
        const data = await res.json();

        if (res.ok) {
            localStorage.setItem('username', data.username);
            localStorage.setItem('plan', data.plan);
            localStorage.setItem('login_timestamp', Date.now());

            btn.innerHTML = '<div class="d-flex align-items-center justify-content-center w-100"><i class="bi bi-check-lg me-2"></i>Success!</div>';
            btn.classList.replace('btn-login', 'btn-success');
            setTimeout(() => { window.location.href = "index.html"; }, 800);
        } else {
            alert("Login Failed: " + data.message);
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    } catch (err) {
        alert("Connection Error: " + err.message + "\n(เช็คว่ารัน Python API หรือยัง?)");
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const btn = document.querySelector('button[type="submit"]');
    const originalText = btn.innerText;
    btn.innerText = "Creating...";
    btn.disabled = true;

    const adminKeyInput = document.getElementById('adminKey');
    const adminKeyValue = adminKeyInput ? adminKeyInput.value : "";

    const formData = {
        username: document.getElementById('usernameInput').value,
        email: document.getElementById('emailInput').value,
        password: document.getElementById('passwordInput').value,
        admin_key: adminKeyValue
    };

    try {
        const res = await fetch(`${getApiUrl()}/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });
        const data = await res.json();

        if (res.ok) {
            alert("✅ Account created successfully!");
            window.location.href = "login.html";
        } else {
            alert("❌ Error: " + data.message);
        }
    } catch (err) {
        alert("Connection Error: " + err.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// ==========================================
// 5. INDEX PAGE LOGIC (Clean & AutoML)
// ==========================================
function initIndexPage() {
    if (isLoggedIn) loadHistory();

    const cleanSelect = document.getElementById('cleanModeSelect');
    if (cleanSelect) {
        cleanSelect.addEventListener('change', (e) => {
            const manualOpts = document.getElementById('manualOptions');
            if (manualOpts) {
                if (e.target.value === 'manual') manualOpts.classList.remove('d-none');
                else manualOpts.classList.add('d-none');
            }
        });
    }

    const mlSelect = document.getElementById('mlMode');
    if (mlSelect) {
        mlSelect.addEventListener('change', (e) => {
            const manualOpts = document.getElementById('mlManualOptions');
            if (manualOpts) {
                if (e.target.value === 'manual') manualOpts.classList.remove('d-none');
                else manualOpts.classList.add('d-none');
            }
        });
    }
}

async function loadHistory() {
    const list = document.getElementById('historyList');
    if (!list) return;

    list.innerHTML = '<div class="text-center text-muted small py-2"><span class="spinner-border spinner-border-sm"></span> Loading...</div>';
    try {
        const res = await fetch(`${getApiUrl()}/history?username=${currentUser}`);
        const data = await res.json();

        if (data.status === "success" && data.history.length > 0) {
            list.innerHTML = "";
            data.history.forEach(item => {
                const el = document.createElement('div');
                el.className = 'history-item';
                el.onclick = () => {
                    if (confirm(`Do you want to download "${item.cleaned_filename}"?`)) {
                        downloadFile(`${getApiUrl()}/download/${item.cleaned_filename}`, item.cleaned_filename);
                    }
                };
                el.innerHTML = `
                    <div class="d-flex align-items-center">
                        <i class="bi bi-file-earmark-spreadsheet text-success me-2"></i>
                        <div class="text-truncate" style="font-size:0.85rem; font-weight:600;" title="${item.filename}">
                            ${item.filename}
                        </div>
                    </div>
                    <div class="text-muted ms-4" style="font-size:0.7rem;">${item.date}</div>
                `;
                list.appendChild(el);
            });
        } else {
            list.innerHTML = '<div class="text-muted small text-center py-2">No history found</div>';
        }
    } catch (err) {
        list.innerHTML = '<div class="text-danger small text-center py-2">Failed to load history</div>';
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput || fileInput.files.length === 0) return;

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const dashboardContent = document.getElementById('dashboardContent');
    const controlPanel = document.getElementById('controlPanel');
    const mlSection = document.getElementById('mlStudioSection');

    if (emptyState) emptyState.classList.add('d-none');
    if (loadingState) loadingState.classList.remove('d-none');
    if (dashboardContent) dashboardContent.classList.add('d-none');
    if (mlSection) mlSection.classList.add('d-none');

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const res = await fetch(`${getApiUrl()}/upload`, {
            method: "POST", body: formData
        });
        const data = await res.json();

        if (data.status === "success") {
            currentRawFile = data.filename;
            datasetColumns = data.all_columns || Object.keys(data.missing_values);

            const statRows = document.getElementById('statRows');
            const statCols = document.getElementById('statCols');
            if (statRows) statRows.innerText = data.rows.toLocaleString();
            if (statCols) statCols.innerText = data.columns;

            const totalMissing = Object.values(data.missing_values).reduce((a, b) => a + b, 0);
            const healthBadge = document.getElementById('healthBadge');
            const healthText = document.getElementById('healthText');
            if (healthBadge && healthText) {
                if (totalMissing === 0) {
                    healthBadge.className = "badge bg-success rounded-pill px-3 py-2";
                    healthBadge.innerHTML = '<i class="bi bi-check-circle me-1"></i>Perfect';
                    healthText.innerText = "Clean Data";
                    healthText.className = "fw-bold text-success mb-0 mt-1";
                } else {
                    healthBadge.className = "badge bg-warning text-dark rounded-pill px-3 py-2";
                    healthBadge.innerHTML = `<i class="bi bi-exclamation-triangle me-1"></i>${totalMissing} Issues`;
                    healthText.innerText = "Needs Cleaning";
                    healthText.className = "fw-bold text-warning mb-0 mt-1";
                }
            }

            if (loadingState) loadingState.classList.add('d-none');
            if (dashboardContent) dashboardContent.classList.remove('d-none');
            if (controlPanel) controlPanel.classList.remove('d-none');

            renderCharts(data.missing_values, data.correlation_matrix);
            if (data.dtype_counts) renderDtypesChart(data.dtype_counts);
            if (data.preview_data) renderPreviewTable(data.preview_data);

        } else { alert("Upload Failed: " + data.message); resetUI(); }
    } catch (err) { alert("Error: ไม่สามารถติดต่อ Backend ได้ กรุณาเช็คว่ารัน Python (main.py) หรือยัง"); resetUI(); }
}

async function processData() {
    const modeSelect = document.getElementById('cleanModeSelect');
    if (!modeSelect) return;

    const mode = modeSelect.value;
    let action = mode;
    if (mode === 'manual') {
        const manualAct = document.getElementById('manualAction');
        if (manualAct) action = manualAct.value;
    }

    const resultSection = document.getElementById('resultSection');
    const mlStudioSection = document.getElementById('mlStudioSection');
    const logArea = document.getElementById('logArea');

    if (resultSection) resultSection.classList.remove('d-none');
    if (mlStudioSection) mlStudioSection.classList.add('d-none');
    if (logArea) logArea.innerHTML = "<span class='text-muted'>Initializing AI Engine...</span><br>";

    window.scrollTo(0, document.body.scrollHeight);

    const formData = new FormData();
    formData.append("filename", currentRawFile);
    formData.append("action", action);
    if (isLoggedIn) formData.append("username", currentUser);

    try {
        const res = await fetch(`${getApiUrl()}/clean`, {
            method: "POST", body: formData
        });
        const data = await res.json();

        if (data.status === "success") {
            currentCleanFile = data.clean_filename || `clean_${currentRawFile}`;

            const downloadBtn = document.getElementById('downloadBtn');
            if (downloadBtn) {
                downloadBtn.onclick = () => {
                    downloadFile(`${getApiUrl()}${data.download_url}`, currentCleanFile);
                };
            }

            if (isLoggedIn) loadHistory();

            if (data.logs && logArea) {
                data.logs.forEach((log, i) => {
                    setTimeout(() => {
                        let color = "text-white";
                        if (log.includes("AI")) color = "text-warning fw-bold";
                        if (log.includes("Drop") || log.includes("Removed")) color = "text-danger";
                        if (log.includes("Fill")) color = "text-info";

                        logArea.innerHTML += `<div class="${color} mb-1">> ${log}</div>`;
                        logArea.scrollTop = logArea.scrollHeight;
                    }, i * 150);
                });

                setTimeout(() => {
                    logArea.innerHTML += `<div class="text-success fw-bold mt-2">> DATA CLEANED. READY FOR ML.</div>`;
                    logArea.scrollTop = logArea.scrollHeight;
                    setupMLStudio();
                }, data.logs.length * 150 + 500);
            }
        }
    } catch (err) {
        if (logArea) logArea.innerHTML += `<div class="text-danger">> ERROR: ${err.message}</div>`;
    }
}

function setupMLStudio() {
    const mlStudioSection = document.getElementById('mlStudioSection');
    if (mlStudioSection) mlStudioSection.classList.remove('d-none');

    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

    const targetSelect = document.getElementById('targetColumn');
    if (targetSelect) {
        targetSelect.innerHTML = '<option value="" disabled selected>-- Select Column --</option>';
        targetSelect.innerHTML += '<option value="NONE" class="text-primary fw-bold">🔍 No Target (Unsupervised / Clustering)</option>';

        datasetColumns.forEach(col => {
            targetSelect.innerHTML += `<option value="${col}">${col}</option>`;
        });
    }
}

async function trainModels() {
    const targetSelect = document.getElementById('targetColumn');
    if (!targetSelect) return;

    const target = targetSelect.value;
    if (!target) return alert("Please select a Target Variable first!");

    const btn = document.getElementById('trainBtn');
    if (btn) {
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
        btn.disabled = true;
    }

    const leaderboardWait = document.getElementById('leaderboardWait');
    const leaderboardResults = document.getElementById('leaderboardResults');

    if (leaderboardWait) leaderboardWait.classList.add('d-none');
    if (leaderboardResults) leaderboardResults.classList.add('d-none');

    const mode = document.getElementById('mlMode') ? document.getElementById('mlMode').value : 'auto';
    const config = {
        models: [],
        scaling: document.getElementById('mlScaling') ? document.getElementById('mlScaling').value : 'standard',
        encoding: document.getElementById('mlEncoding') ? document.getElementById('mlEncoding').value : 'onehot'
    };

    if (mode === 'manual') {
        document.querySelectorAll('.ml-model-cb:checked').forEach(cb => config.models.push(cb.value));
        if (config.models.includes("KNN Regressor")) config.models.push("KNN Classifier");
        if (config.models.includes("Linear Regression")) config.models.push("Logistic Regression");
        if (config.models.includes("SVM")) { config.models.push("SVM (SVR)"); config.models.push("SVM (SVC)"); }
        if (config.models.includes("Neural Network (MLP)")) config.models.push("Neural Network (MLP)");
    }

    const formData = new FormData();
    formData.append("filename", currentCleanFile);
    formData.append("target_column", target);
    formData.append("mode", mode);
    formData.append("manual_config", JSON.stringify(config));

    try {
        const res = await fetch(`${getApiUrl()}/train_model`, { method: "POST", body: formData });
        const data = await res.json();

        if (data.status === "success") {
            const taskTypeBadge = document.getElementById('taskTypeBadge');
            if (taskTypeBadge) taskTypeBadge.innerText = data.task_type + " Task";

            const tbody = document.getElementById('leaderboardBody');
            if (tbody) {
                tbody.innerHTML = "";

                data.leaderboard.forEach((item, index) => {
                    let medal = index === 0 ? "🏆" : (index === 1 ? "🥈" : "🥉");
                    let rankClass = index === 0 ? "fw-bold text-success bg-success-subtle" : "";
                    let precision = item.metrics.Precision ? item.metrics.Precision + "%" : "-";
                    let recall = item.metrics.Recall ? item.metrics.Recall + "%" : "-";

                    tbody.innerHTML += `
                        <tr class="leaderboard-row ${rankClass}">
                            <td class="fw-bold">${medal} #${index + 1}</td>
                            <td class="text-start">${item.model}</td>
                            <td class="fw-bold fs-5">${item.score}%</td>
                            <td class="text-muted small">${precision}</td>
                            <td class="text-muted small">${recall}</td>
                            <td>
                                <button onclick="downloadFile('${getApiUrl()}${item.download_url}', 'model_${item.model}.pkl')" class="btn btn-sm btn-outline-primary rounded-pill">
                                    <i class="bi bi-download"></i> .pkl
                                </button>
                            </td>
                        </tr>
                    `;
                });
            }
            if (leaderboardResults) leaderboardResults.classList.remove('d-none');
        } else {
            alert("Training Error: " + data.message);
            if (leaderboardWait) leaderboardWait.classList.remove('d-none');
        }
    } catch (err) {
        alert("Connection Error: " + err.message);
        if (leaderboardWait) leaderboardWait.classList.remove('d-none');
    } finally {
        if (btn) {
            btn.innerHTML = '<i class="bi bi-play-circle-fill me-2"></i>Train Models';
            btn.disabled = false;
        }
    }
}

function renderCharts(missing, corr) {
    const missingChart = document.getElementById('missingChart');
    if (missingChart) {
        const mX = Object.keys(missing); const mY = Object.values(missing);
        if (mY.some(v => v > 0)) {
            Plotly.newPlot('missingChart', [{ x: mX, y: mY, type: 'bar', marker: { color: '#ef4444' } }], { margin: { t: 30, l: 50, r: 30, b: 100 }, xaxis: { tickangle: -45, automargin: true }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, { displayModeBar: false, responsive: true });
        } else {
            missingChart.innerHTML = "<div class='text-center text-muted py-5'>✅ No missing values</div>";
        }
    }

    const heatmapChart = document.getElementById('heatmapChart');
    if (heatmapChart) {
        if (corr) {
            Plotly.newPlot('heatmapChart', [{ z: corr.z, x: corr.x, y: corr.y, type: 'heatmap', colorscale: 'Viridis' }], { margin: { t: 10, l: 50, r: 10, b: 50 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, { displayModeBar: false, responsive: true });
        } else {
            heatmapChart.innerHTML = "<div class='text-center text-muted py-5'>Not enough numeric data</div>";
        }
    }
}

function renderDtypesChart(dtypes) {
    const dtypesChart = document.getElementById('dtypesChart');
    if (!dtypes || !dtypesChart) return;
    const data = [{ values: Object.values(dtypes), labels: Object.keys(dtypes), type: 'pie', hole: .4, textinfo: "label+percent", marker: { colors: ['#6366f1', '#10b981', '#f59e0b', '#ef4444'] } }];
    Plotly.newPlot('dtypesChart', data, { margin: { t: 0, l: 0, r: 0, b: 0 }, showlegend: true, height: 250, paper_bgcolor: 'rgba(0,0,0,0)' }, { displayModeBar: false });
}

function renderPreviewTable(preview) {
    if (!preview) return;
    const thead = document.getElementById('tableHead');
    const tbody = document.getElementById('tableBody');
    if (!thead || !tbody) return;

    let headHtml = ""; preview.columns.forEach(col => { headHtml += `<th scope="col" class="fw-bold text-secondary text-nowrap">${col}</th>`; }); thead.innerHTML = headHtml;
    let bodyHtml = "";
    preview.data.forEach(row => {
        bodyHtml += "<tr>";
        row.forEach(cell => {
            let cellData = cell === null ? '<span class="text-danger fst-italic">null</span>' : cell;
            bodyHtml += `<td class="text-nowrap">${cellData}</td>`;
        });
        bodyHtml += "</tr>";
    });
    tbody.innerHTML = bodyHtml;
}

function resetUI() {
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');
    if (loadingState) loadingState.classList.add('d-none');
    if (emptyState) emptyState.classList.remove('d-none');
}

// ==========================================
// 6. PREDICT PAGE LOGIC (อัปโหลดโมเดล & พยากรณ์)
// ==========================================

// ✅ รองรับการแสดงชื่อไฟล์
function handlePredictFileSelect(inputId, labelId) {
    const fileInput = document.getElementById(inputId);
    const file = fileInput ? fileInput.files[0] : null;

    if (file) {
        const labelEl = document.getElementById(labelId);
        if (labelEl) labelEl.innerHTML = `<span class="text-dark">${file.name}</span>`;

        if (fileInput && fileInput.parentElement) {
            fileInput.parentElement.style.borderColor = "#10b981";
            fileInput.parentElement.style.backgroundColor = "#f0fdf4";
        }

        if (inputId === 'dataFile') {
            prepareTargetSelection(file);
        }
    }
}

// ✅ เผื่อไว้กรณี HTML เรียกใช้ showName
function showName(inputId, labelId) {
    handlePredictFileSelect(inputId, labelId);
}

function prepareTargetSelection(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result;
        globalHeaders = text.split('\n')[0].split(',').map(h => h.trim()).filter(h => h !== '');

        const targetSelect = document.getElementById('manualTargetColumn');
        if (!targetSelect) return;

        targetSelect.innerHTML = '<option value="" disabled selected>-- เลือกคอลัมน์ที่ต้องการ Predict (Y) --</option>';
        globalHeaders.forEach(header => {
            targetSelect.innerHTML += `<option value="${header}">${header}</option>`;
        });

        const manualTestSection = document.getElementById('manualTestSection');
        const targetSelectionWrapper = document.getElementById('targetSelectionWrapper');
        const dynamicFormWrapper = document.getElementById('dynamicFormWrapper');

        if (manualTestSection) manualTestSection.classList.remove('d-none');
        if (targetSelectionWrapper) targetSelectionWrapper.classList.remove('d-none');
        if (dynamicFormWrapper) dynamicFormWrapper.classList.add('d-none');
    };
    reader.readAsText(file.slice(0, 1000));
}

function generateManualForm() {
    const targetSelect = document.getElementById('manualTargetColumn');
    const targetCol = targetSelect ? targetSelect.value : null;
    if (!targetCol) return;

    const form = document.getElementById('dynamicForm');
    if (!form) return;

    form.innerHTML = '';

    globalHeaders.forEach(header => {
        if (header === targetCol) return;

        form.innerHTML += `
            <div class="col-md-3 col-sm-6">
                <label class="form-label small fw-bold text-muted mb-1">${header}</label>
                <input type="text" class="form-control manual-input-box p-2" data-col="${header}" placeholder="Enter ${header}">
            </div>
        `;
    });

    const dynamicFormWrapper = document.getElementById('dynamicFormWrapper');
    if (dynamicFormWrapper) dynamicFormWrapper.classList.remove('d-none');
}

function renderPredictionChart(summary) {
    const chartDiv = document.getElementById('predictionChart');
    if (!chartDiv) return;

    const badge = document.getElementById('chartTypeBadge');
    if (badge) badge.innerText = summary.type.toUpperCase();

    if (summary.type === 'classification') {
        const labels = Object.keys(summary.distribution);
        const values = Object.values(summary.distribution);
        const data = [{
            values: values, labels: labels, type: 'pie', hole: .4,
            marker: { colors: ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'] }
        }];
        Plotly.newPlot(chartDiv, data, { margin: { t: 10, b: 10, l: 10, r: 10 }, paper_bgcolor: 'rgba(0,0,0,0)' }, { displayModeBar: false });
    } else {
        const labels = Object.keys(summary.distribution);
        const values = Object.values(summary.distribution);
        const data = [{
            x: labels, y: values, type: 'bar',
            marker: { color: ['#10b981', '#3b82f6', '#ef4444'] }
        }];
        Plotly.newPlot(chartDiv, data, { margin: { t: 20, b: 30, l: 40, r: 10 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, { displayModeBar: false });
    }
}

// ✅ ระบบหลักทนทานต่อการลบ HTML ทิ้ง
async function executePrediction(modelFile, dataFile, btnId, isManual = false) {
    const btn = document.getElementById(btnId);
    let originalText = "Predicting...";

    if (btn) {
        originalText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';
        btn.disabled = true;
    }

    const resultSection = document.getElementById('resultSection');
    if (resultSection) resultSection.classList.add('d-none');

    const formData = new FormData();
    formData.append("model_file", modelFile);
    formData.append("data_file", dataFile);

    try {
        const res = await fetch(`${getApiUrl()}/predict`, {
            method: "POST", body: formData
        });

        if (!res.ok) throw new Error(`Server Error Code ${res.status}`);

        const result = await res.json();

        if (result.status === "success") {
            if (resultSection) resultSection.classList.remove('d-none');

            // อัปเดตข้อมูลแบบตรวจสอบก่อนว่ามีช่องให้แสดงหรือไม่
            const statTotalRows = document.getElementById('statTotalRows');
            if (statTotalRows) statTotalRows.innerText = result.total_rows.toLocaleString();

            if (result.summary) {
                const statMatch = document.getElementById('statMatch');
                if (statMatch) statMatch.innerText = result.summary.data_quality + "%";

                const statInsightLabel = document.getElementById('statInsightLabel');
                if (statInsightLabel) statInsightLabel.innerText = result.summary.insight_label;

                const statInsightValue = document.getElementById('statInsightValue');
                if (statInsightValue) statInsightValue.innerText = result.summary.insight_value;
            }

            const chartRow = document.getElementById('chartActionRow');
            if (isManual) {
                if (chartRow) chartRow.classList.add('d-none');
            } else {
                if (chartRow) chartRow.classList.remove('d-none');
                if (result.summary) renderPredictionChart(result.summary);
            }

            const table = document.getElementById('resultTable');
            if (table) {
                let html = '<thead class="table-light"><tr>';
                result.preview.columns.forEach(col => {
                    let bgClass = col === 'Predicted_Result' ? 'bg-warning text-dark fs-6' : '';
                    html += `<th class="${bgClass} fw-bold text-nowrap">${col}</th>`;
                });
                html += '</tr></thead><tbody>';
                result.preview.data.forEach(row => {
                    html += '<tr>';
                    row.forEach((cell, index) => {
                        let isPred = result.preview.columns[index] === 'Predicted_Result';
                        let cellStyle = isPred ? 'fw-bold text-dark bg-warning-subtle fs-5' : '';
                        let cellData = cell === null ? '<span class="text-danger small">NaN</span>' : cell;
                        html += `<td class="${cellStyle} align-middle">${cellData}</td>`;
                    });
                    html += '</tr>';
                });
                table.innerHTML = html + '</tbody>';
            }

            if (!isManual) {
                const dlBtn = document.getElementById('dlResultBtn');
                if (dlBtn) {
                    dlBtn.onclick = () => {
                        downloadFile(`${getApiUrl()}${result.download_url}`, `predicted_data.csv`);
                    }
                }
            }

            if (resultSection) resultSection.scrollIntoView({ behavior: 'smooth' });

        } else {
            alert("Error: " + result.message);
        }
    } catch (err) {
        alert("Connection Error: " + err.message + "\n(เช็คว่าเปิด Terminal Python อยู่หรือไม่?)");
    } finally {
        if (btn) {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }
}

function predictBatch() {
    // ลองหา ID สองแบบ เผื่อหน้า HTML คุณตั้ง ID ปุ่ม/ไฟล์ไม่เหมือนเดิม
    const modelInput = document.getElementById('modelFile');
    const dataInput = document.getElementById('dataFile');

    if (!modelInput || !dataInput || !modelInput.files[0] || !dataInput.files[0]) {
        return alert("Please upload both Model (.pkl) and Reference Data (.csv)");
    }

    // สังเกตว่าในรูปของคุณ ปุ่มมีสีส้มๆ อาจจะตั้ง id="predictBtn" หรือ id อื่น
    // ถ้าคุณไม่ได้ใส่ id ให้มัน ให้กำหนด ID "predictBtn" ที่ปุ่ม Run Prediction ในหน้า HTML ของคุณด้วยนะครับ
    executePrediction(modelInput.files[0], dataInput.files[0], 'predictBtn', false);
}

function predictManual() {
    const modelInput = document.getElementById('modelFile');
    if (!modelInput || !modelInput.files[0]) return alert("Please upload Model (.pkl) first");

    const inputs = document.querySelectorAll('.manual-input-box');
    let headers = [];
    let values = [];

    inputs.forEach(input => {
        headers.push(input.dataset.col);
        values.push(input.value.trim());
    });

    const csvContent = headers.join(',') + '\n' + values.join(',');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const manualDataFile = new File([blob], "manual_input.csv", { type: 'text/csv' });

    executePrediction(modelInput.files[0], manualDataFile, 'manualPredictBtn', true);
}