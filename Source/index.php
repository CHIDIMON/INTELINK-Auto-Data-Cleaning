// index.php
<?php
// 1. เริ่ม Session
require_once 'session.php'; 

// 2. ตรวจสอบสถานะ (ไม่ต้อง Redirect ถ้าเป็น Guest)
if (isset($_SESSION['user_logged_in']) && $_SESSION['user_logged_in'] === true) {
    check_session_timeout(); 
    $isLoggedIn = true;
    $username = $_SESSION['username'] ?? 'Member';
    $plan = $_SESSION['plan'] ?? 'free';
} else {
    // โหมด Guest
    $isLoggedIn = false;
    $username = 'Guest';
    $plan = 'free';
}
?>
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart AI Data Cleaner & AutoML</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --primary-color: #6366f1;
            --ml-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --glass-bg: rgba(255, 255, 255, 0.95);
            --sidebar-width: 320px;
            --bg-color: #f8fafc;
            --text-main: #334155;
        }

        body {
            font-family: 'Sarabun', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            overflow-x: hidden;
        }

        h1, h2, h3, h4, h5, .brand-font { font-family: 'Outfit', sans-serif; }

        .dashboard-container { display: flex; min-height: 100vh; }
        
        /* Sidebar Styling */
        .sidebar {
            width: var(--sidebar-width);
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
            padding: 2rem;
            position: fixed; 
            height: 100vh; 
            overflow-y: auto;
            z-index: 100;
            box-shadow: 4px 0 24px rgba(0,0,0,0.02);
            display: flex;
            flex-direction: column;
        }

        /* Main Content Styling */
        .main-content {
            flex: 1;
            margin-left: var(--sidebar-width);
            padding: 2rem 3rem;
            max-width: 1600px;
        }

        .card-modern {
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            overflow: hidden;
        }
        .card-modern:hover { 
            transform: translateY(-4px); 
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05);
            border-color: #cbd5e1;
        }

        .btn-ai {
            background: var(--primary-gradient);
            border: none; color: white; padding: 14px 24px;
            border-radius: 12px; font-weight: 600; width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .btn-ai:hover { opacity: 0.95; transform: scale(1.02); color: white; box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3); }

        .btn-ml {
            background: var(--ml-gradient); border: none; color: white; padding: 12px 20px;
            border-radius: 12px; font-weight: 600; width: 100%; transition: 0.3s;
        }
        .btn-ml:hover { opacity: 0.95; transform: scale(1.02); color: white; box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3); }

        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 16px; padding: 40px 20px;
            text-align: center; cursor: pointer; transition: 0.3s; background: #f1f5f9;
        }
        .upload-area:hover { border-color: var(--primary-color); background: #e0e7ff; color: var(--primary-color); }

        .console-log {
            background: #0f172a; color: #38bdf8; border-radius: 12px; padding: 20px;
            font-family: 'Courier New', monospace; font-size: 0.85rem; height: 150px; overflow-y: auto;
        }

        .history-item {
            padding: 10px; border-radius: 8px; cursor: pointer; transition: 0.2s;
            border: 1px solid transparent; font-size: 0.9rem; position: relative;
        }
        .history-item:hover { 
            background: #f1f5f9; border-color: #e2e8f0; transform: translateX(5px);
        }
        .history-item:hover::after {
            content: '\F30A'; font-family: "bootstrap-icons"; position: absolute; right: 15px; top: 50%; transform: translateY(-50%); color: var(--primary-color); font-size: 1.2rem;
        }
        
        .leaderboard-row { transition: 0.2s; border-radius: 10px; }
        .leaderboard-row:hover { background-color: #f1f5f9; transform: scale(1.01); }

        .pro-badge {
            background: linear-gradient(45deg, #FFD700, #FFA500);
            color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }

        @media (max-width: 992px) {
            .sidebar { position: relative; width: 100%; height: auto; border-right: none; }
            .main-content { margin-left: 0; padding: 1.5rem; }
            .dashboard-container { flex-direction: column; }
        }
    </style>
</head>
<body>

<div class="dashboard-container">
    
    <!-- ================= SIDEBAR ================= -->
    <aside class="sidebar">
        <!-- Brand -->
        <div class="mb-4 d-flex align-items-center gap-2">
            <div class="bg-primary text-white rounded-3 p-2 d-flex align-items-center justify-content-center" style="width:40px; height:40px;">
                <i class="bi bi-robot fs-5"></i>
            </div>
            <div>
                <h4 class="fw-bold brand-font text-dark mb-0">SmartClean</h4>
                <!-- แสดง Badge ตาม Plan -->
                <?php if ($plan === 'pro'): ?>
                    <div class="badge pro-badge rounded-pill"><i class='bi bi-star-fill me-1'></i>PRO Plan</div>
                <?php elseif ($isLoggedIn): ?>
                    <div class="badge bg-info text-dark rounded-pill">Free Plan</div>
                <?php else: ?>
                    <div class="badge bg-secondary rounded-pill">Guest Mode</div>
                <?php endif; ?>
            </div>
        </div>

        <!-- 1. Upload Section -->
        <div class="mb-4">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h6 class="fw-bold text-uppercase text-muted small">Step 1: Input Data</h6>
            </div>
            <div class="upload-area" id="dropZone" onclick="document.getElementById('csvFile').click()">
                <i class="bi bi-cloud-arrow-up-fill fs-1 mb-2 text-secondary"></i>
                <h6 class="fw-bold text-dark mb-1">Click to Upload</h6>
                <p class="small text-muted mb-0">CSV Only</p>
                <input type="file" id="csvFile" accept=".csv" hidden onchange="uploadFile()">
            </div>
        </div>

        <!-- 2. Cleaning Options -->
        <div id="controlPanel" class="d-none mb-4">
            <h6 class="fw-bold text-uppercase text-muted small mb-3">Step 2: Clean Strategy</h6>
            <select id="cleanModeSelect" class="form-select mb-3">
                <!-- ✅ เพิ่มตัวเลือก AI Agent -->
                <option value="ai_agent">🤖 Gemini AI Agent (Smartest)</option>
                <option value="auto_smart">✨ Classic Auto-Clean</option>
                <option value="manual">🛠️ Manual Configuration</option>
            </select>
            
            <div id="manualOptions" class="d-none mb-3 ps-2 border-start border-3 border-primary">
                <select id="manualAction" class="form-select form-select-sm bg-light">
                    <option value="drop_missing">Drop Missing Rows</option>
                    <option value="fill_mean">Fill Numeric with Mean</option>
                    <option value="remove_outliers">Remove Outliers (IQR)</option>
                </select>
            </div>

            <button onclick="processData()" class="btn-ai shadow-sm" id="processBtn">
                <i class="bi bi-magic me-2"></i>Clean Data
            </button>
        </div>

        <!-- 3. History Section -->
        <?php if ($isLoggedIn): ?>
        <div id="historySection" class="mt-auto pt-3 border-top">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h6 class="fw-bold text-uppercase text-muted small mb-0"><i class="bi bi-clock-history me-1"></i> Recent Files</h6>
                <button class="btn btn-sm text-primary p-0" onclick="loadHistory()"><i class="bi bi-arrow-clockwise"></i></button>
            </div>
            <div id="historyList" class="d-flex flex-column gap-1" style="max-height: 150px; overflow-y: auto;">
                <div class="text-center small text-muted py-2">Loading...</div>
            </div>
        </div>
        <?php else: ?>
        <div id="guestMessage" class="mt-auto pt-3 border-top text-center">
            <p class="small text-muted mb-2">Want to save your work?</p>
            <a href="login.php" class="btn btn-outline-primary btn-sm w-100 rounded-pill">Login to Save History</a>
        </div>
        <?php endif; ?>
    </aside>

    <!-- ================= MAIN CONTENT ================= -->
    <main class="main-content">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-5">
            <div>
                <h2 class="fw-bold text-dark mb-1">Data Insights & AutoML</h2>
                <p class="text-muted mb-0">Visualize, Clean, and Build ML Models automatically.</p>
            </div>
            
            <!-- Auth Buttons -->
            <div id="authContainer">
                <?php if ($isLoggedIn): ?>
                    <div class="dropdown">
                        <button class="btn btn-white border text-dark fw-bold px-3 shadow-sm rounded-pill dropdown-toggle" type="button" data-bs-toggle="dropdown">
                            <i class="bi bi-person-circle me-2 text-primary"></i><span><?php echo htmlspecialchars($username); ?></span>
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end shadow-lg border-0 rounded-4 mt-2">
                            <li><h6 class="dropdown-header">Account</h6></li>
                            <li><a class="dropdown-item" href="#"><i class="bi bi-person me-2"></i>Profile</a></li>
                            <?php if ($plan === 'free'): ?>
                                <li><a class="dropdown-item text-warning" href="#"><i class="bi bi-star-fill me-2"></i>Upgrade to Pro</a></li>
                            <?php endif; ?>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item text-danger" href="logout.php"><i class="bi bi-box-arrow-right me-2"></i>Logout</a></li>
                        </ul>
                    </div>
                <?php else: ?>
                    <!-- Guest View -->
                    <a href="login.php" class="btn btn-white border text-primary fw-bold px-4 shadow-sm rounded-pill">
                        <i class="bi bi-box-arrow-in-right me-2"></i>Login
                    </a>
                <?php endif; ?>
            </div>
        </div>

        <!-- Empty State -->
        <div id="emptyState" class="text-center py-5 my-5">
            <div class="mb-4 opacity-50"><i class="bi bi-cloud-upload display-1 text-secondary"></i></div>
            <h4 class="fw-bold text-secondary">Waiting for Data</h4>
            <p class="text-muted">Please upload a CSV file from the left panel to begin.</p>
        </div>

        <!-- Loading -->
        <div id="loadingState" class="d-none text-center py-5 my-5">
            <div class="spinner-border text-primary mb-3" role="status"></div>
            <h5 class="fw-bold text-primary animate-pulse">Analyzing Dataset...</h5>
        </div>

        <!-- Dashboard Content -->
        <div id="dashboardContent" class="d-none">
            <!-- Stats Row -->
            <div class="row g-4 mb-4">
                <div class="col-md-3">
                    <div class="card-modern p-4">
                        <small class="text-muted text-uppercase fw-bold">Rows</small>
                        <h2 class="fw-bold text-dark mb-0" id="statRows">-</h2>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card-modern p-4">
                        <small class="text-muted text-uppercase fw-bold">Columns</small>
                        <h2 class="fw-bold text-dark mb-0" id="statCols">-</h2>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card-modern p-4 d-flex align-items-center justify-content-between h-100 bg-white">
                        <div>
                            <small class="text-muted text-uppercase fw-bold">Data Health</small>
                            <h4 class="fw-bold text-dark mb-0 mt-1" id="healthText">Analyzing...</h4>
                        </div>
                        <div id="healthBadge" class="badge bg-secondary rounded-pill px-3 py-2 fs-6">Unknown</div>
                    </div>
                </div>
            </div>

            <!-- Row 1: Missing Values (FULL WIDTH) -->
            <div class="row g-4 mb-4">
                <div class="col-12">
                    <div class="card-modern p-4 h-100">
                        <h6 class="fw-bold mb-3"><i class="bi bi-bar-chart-fill text-primary me-2"></i>Missing Values Distribution</h6>
                        <div id="missingChart" style="height: 350px;"></div>
                    </div>
                </div>
            </div>

            <!-- Row 2: Correlation & Data Types (HALF WIDTH) -->
            <div class="row g-4 mb-4">
                <!-- Correlation Matrix -->
                <div class="col-lg-6">
                    <div class="card-modern p-4 h-100">
                        <h6 class="fw-bold mb-3"><i class="bi bi-grid-3x3-gap-fill text-danger me-2"></i>Correlation Matrix</h6>
                        <div id="heatmapChart" style="height: 350px;"></div>
                    </div>
                </div>
                <!-- Column Types -->
                <div class="col-lg-6">
                    <div class="card-modern p-4 h-100">
                        <h6 class="fw-bold mb-3"><i class="bi bi-pie-chart-fill text-info me-2"></i>Column Data Types</h6>
                        <div id="dtypesChart" style="height: 350px;"></div>
                    </div>
                </div>
            </div>

            <!-- Row 3: Data Preview (FULL WIDTH) -->
            <div class="row g-4 mb-4">
                <div class="col-12">
                    <div class="card-modern p-4 h-100">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="fw-bold mb-0"><i class="bi bi-table text-secondary me-2"></i>Data Preview</h6>
                            <span class="badge bg-light text-muted border">First 5 Rows</span>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover table-sm small mb-0" id="previewTable">
                                <thead class="table-light"><tr id="tableHead"></tr></thead>
                                <tbody id="tableBody"><tr><td class="text-center text-muted p-4">Waiting for upload...</td></tr></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Result Box -->
            <div id="resultSection" class="d-none mb-4">
                <div class="card-modern p-0 border-success border-2 overflow-hidden">
                    <div class="bg-success-subtle p-3 px-4 border-bottom border-success-subtle d-flex justify-content-between align-items-center">
                        <div class="d-flex align-items-center gap-2">
                            <i class="bi bi-check-circle-fill text-success fs-4"></i>
                            <h5 class="fw-bold text-success-emphasis mb-0">Cleaning Complete!</h5>
                        </div>
                        <button id="downloadBtn" class="btn btn-success fw-bold px-4 shadow-sm rounded-pill">
                            <i class="bi bi-download me-2"></i>Download Clean CSV
                        </button>
                    </div>
                    <div class="p-4 bg-dark">
                        <h6 class="text-white-50 text-uppercase small mb-3 fw-bold">System Logs</h6>
                        <div class="console-log" id="logArea"></div>
                    </div>
                </div>
            </div>

            <!-- ================= MACHINE LEARNING STUDIO ================= -->
            <div id="mlStudioSection" class="d-none mb-4">
                <div class="card-modern p-4 border-primary border-2 shadow-sm">
                    <h4 class="fw-bold text-dark mb-4"><i class="bi bi-cpu-fill text-primary me-2"></i>Step 3: Auto-ML Studio</h4>
                    
                    <div class="row g-4">
                        <!-- Left: Settings -->
                        <div class="col-lg-5 border-end pe-4">
                            <label class="fw-bold text-dark mb-2">1. Select Target Variable (Y):</label>
                            <select id="targetColumn" class="form-select mb-4 border-primary border-2 shadow-sm"></select>
                            
                            <label class="fw-bold text-dark mb-2">2. Training Mode:</label>
                            <select id="mlMode" class="form-select mb-3">
                                <option value="auto">🚀 Full AutoML (Recommended)</option>
                                <option value="manual">⚙️ Manual Expert Mode</option>
                            </select>
                            
                            <!-- Manual Options with NEW Models -->
                            <div id="mlManualOptions" class="d-none bg-light p-3 rounded-3 mb-4 border">
                                <label class="small fw-bold mb-2">Select Models to Train:</label>
                                <div class="form-check">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="Linear Regression" checked> <label class="small">Linear / Logistic Regression</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="KNN Regressor" checked> <label class="small">K-Nearest Neighbors (KNN)</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="Decision Tree" checked> <label class="small">Decision Tree</label>
                                </div>
                                <!-- ✅ เพิ่มโมเดลใหม่ -->
                                <div class="form-check">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="Random Forest"> <label class="small">Random Forest</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="SVM"> <label class="small">Support Vector Machine (SVM)</label>
                                </div>
                                <div class="form-check mb-2">
                                    <input class="form-check-input ml-model-cb" type="checkbox" value="Neural Network (MLP)"> <label class="small">Neural Network (MLP)</label>
                                </div>
                                
                                <label class="small fw-bold mt-2 mb-1">Scaling Method:</label>
                                <select id="mlScaling" class="form-select form-select-sm">
                                    <option value="standard">Standard Scaler (Z-Score)</option>
                                    <option value="minmax">Min-Max Scaler (0 to 1)</option>
                                </select>
                            </div>

                            <button onclick="trainModels()" class="btn-ml mt-2" id="trainBtn">
                                <i class="bi bi-play-circle-fill me-2"></i>Train Models
                            </button>
                        </div>
                        
                        <!-- Right: Leaderboard & Action -->
                        <div class="col-lg-7 ps-4">
                            <h6 class="fw-bold text-secondary mb-3"><i class="bi bi-trophy-fill text-warning me-2"></i>Model Leaderboard</h6>
                            
                            <div id="leaderboardWait" class="text-center py-5 bg-light rounded-3 border border-dashed">
                                <i class="bi bi-bar-chart-steps text-muted display-4 mb-3"></i>
                                <p class="text-muted small">Select a target variable and click "Train Models" to see the results.</p>
                            </div>

                            <div id="leaderboardResults" class="d-none">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <span class="badge bg-primary text-white" id="taskTypeBadge">Classification Task</span>
                                </div>
                                <div class="table-responsive">
                                    <table class="table table-hover align-middle border">
                                        <thead class="table-light">
                                            <tr>
                                                <th>Rank</th>
                                                <th>Algorithm</th>
                                                <th>Score</th>
                                            </tr>
                                        </thead>
                                        <tbody id="leaderboardBody"></tbody>
                                    </table>
                                </div>
                                <div class="mt-3 d-flex justify-content-end gap-2">
                                    <!-- ✅ เพิ่มปุ่มขอดู Source Code Python -->
                                    <button class="btn btn-outline-dark rounded-pill px-3 shadow-sm" data-bs-toggle="modal" data-bs-target="#pythonModal">
                                        <i class="bi bi-filetype-py me-1"></i>How to use in Python
                                    </button>
                                    <button id="downloadModelBtn" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                        <i class="bi bi-download me-2"></i>Download .pkl
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </main>
</div>

<!-- ================= MODAL: Python Source Code ================= -->
<div class="modal fade" id="pythonModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-centered">
        <div class="modal-content rounded-4 border-0 shadow-lg">
            <div class="modal-header bg-dark text-white rounded-top-4 border-bottom-0">
                <h5 class="modal-title fw-bold"><i class="bi bi-filetype-py text-warning me-2"></i>Python Integration Guide</h5>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body bg-dark text-light p-4">
                <p class="small text-info mb-3">1. ติดตั้ง Library ที่จำเป็น (ถ้ายังไม่มี): <code>pip install pandas scikit-learn joblib</code><br>2. วางไฟล์โมเดล <code>best_model.pkl</code> ไว้ในโฟลเดอร์เดียวกับโค้ดด้านล่าง</p>
                
                <div class="position-relative">
                    <button class="btn btn-sm btn-outline-light position-absolute top-0 end-0 m-2" onclick="copyPythonCode()">
                        <i class="bi bi-clipboard"></i> Copy
                    </button>
                    <pre class="rounded-3 p-3 bg-black" style="font-family: 'Courier New', Courier, monospace;"><code id="pythonCodeSnippet" class="text-success">import pandas as pd
import joblib

# 1. โหลดโมเดลที่ดาวน์โหลดมา (สมอง AI ที่ถูกฝึกแล้ว)
model = joblib.load('best_model.pkl')

# 2. เตรียมข้อมูลใหม่ที่ต้องการทำนาย (ข้อมูลต้องมีคอลัมน์เหมือนตอนฝึก)
# ตัวอย่างการอ่านจากไฟล์ CSV:
new_data = pd.read_csv('new_customers_to_predict.csv')

# 3. ให้โมเดลทำการพยากรณ์
predictions = model.predict(new_data)

# 4. แสดงผลลัพธ์
new_data['Predicted_Result'] = predictions
print(new_data.head())

# (ถ้าต้องการบันทึกผลลัพธ์ลงไฟล์ใหม่)
# new_data.to_csv('predicted_results.csv', index=False)
</code></pre>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

<script>
    // ✅ ใช้ Ngrok URL ที่คุณให้มา (อย่าลืมตัด / ท้ายสุดออกถ้ามี)
    const NGROK_URL = "https://intermandibular-cohen-nontalkatively.ngrok-free.dev"; 
    
    // ================= STATE =================
    const currentUser = "<?php echo $isLoggedIn ? $username : ''; ?>";
    let currentRawFile = "";
    let currentCleanFile = "";
    let datasetColumns = [];

    function getApiUrl() { 
        let url = NGROK_URL.trim();
        // ถ้า Ngrok ไม่ได้ตั้ง หรือรัน Local ให้ใช้ localhost
        if(url === "" || url.includes("your-ngrok-url")) return "http://127.0.0.1:8000";
        return url.endsWith('/') ? url.slice(0, -1) : url; 
    }

    // ฟังก์ชันดาวน์โหลด
    async function downloadFile(url, filename) {
        try {
            const response = await fetch(url, { headers: { 'ngrok-skip-browser-warning': 'true' } });
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

    // ฟังก์ชัน Copy Python Code
    function copyPythonCode() {
        const codeText = document.getElementById('pythonCodeSnippet').innerText;
        navigator.clipboard.writeText(codeText).then(() => {
            alert("✅ คัดลอกโค้ด Python สำเร็จ! นำไปวางในไฟล์ .py ของคุณได้เลย");
        });
    }

    // Toggle Manual Options
    document.getElementById('cleanModeSelect').addEventListener('change', (e) => {
        const manualOpts = document.getElementById('manualOptions');
        if(e.target.value === 'manual') manualOpts.classList.remove('d-none');
        else manualOpts.classList.add('d-none');
    });

    document.getElementById('mlMode').addEventListener('change', (e) => {
        const manualOpts = document.getElementById('mlManualOptions');
        if(e.target.value === 'manual') manualOpts.classList.remove('d-none');
        else manualOpts.classList.add('d-none');
    });

    // โหลดประวัติไฟล์
    async function loadHistory() {
        if(!currentUser || currentUser === 'Guest') return;
        const list = document.getElementById('historyList');
        list.innerHTML = '<div class="text-center text-muted small py-2"><span class="spinner-border spinner-border-sm"></span> Loading...</div>';
        try {
            const res = await fetch(`${getApiUrl()}/history?username=${currentUser}`, { headers: { "ngrok-skip-browser-warning": "true" } });
            const data = await res.json();
            
            if(data.status === "success" && data.history.length > 0) {
                list.innerHTML = "";
                data.history.forEach(item => {
                    const el = document.createElement('div');
                    el.className = 'history-item';
                    el.onclick = () => {
                        if(confirm(`Do you want to download "${item.cleaned_filename}"?`)) {
                            downloadFile(`${getApiUrl()}/download/${item.cleaned_filename}`, item.cleaned_filename);
                        }
                    };
                    el.innerHTML = `
                        <div class="d-flex align-items-center">
                            <i class="bi bi-file-earmark-spreadsheet text-success me-2"></i>
                            <div class="text-truncate" style="font-size:0.85rem; font-weight:600;" title="${item.filename}">${item.filename}</div>
                        </div>
                        <div class="text-muted ms-4" style="font-size:0.7rem;">${item.date}</div>
                    `;
                    list.appendChild(el);
                });
            } else { list.innerHTML = '<div class="text-muted small text-center py-2">No history found</div>'; }
        } catch(err) { list.innerHTML = '<div class="text-danger small text-center py-2">Failed to load history</div>'; }
    }

    document.addEventListener("DOMContentLoaded", () => {
        if(currentUser) loadHistory();
    });

    // อัปโหลดไฟล์
    async function uploadFile() {
        const fileInput = document.getElementById('csvFile');
        if(fileInput.files.length === 0) return;
        
        document.getElementById('emptyState').classList.add('d-none');
        document.getElementById('loadingState').classList.remove('d-none');
        document.getElementById('dashboardContent').classList.add('d-none');
        document.getElementById('mlStudioSection').classList.add('d-none');
        
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const res = await fetch(`${getApiUrl()}/upload`, { 
                method: "POST", body: formData, headers: { "ngrok-skip-browser-warning": "true" } 
            });

            if (!res.ok) throw new Error("Server connection failed");

            const data = await res.json();
            if(data.status === "success") {
                currentRawFile = data.filename;
                datasetColumns = data.all_columns || Object.keys(data.missing_values);
                
                document.getElementById('statRows').innerText = data.rows.toLocaleString();
                document.getElementById('statCols').innerText = data.columns;
                
                const totalMissing = Object.values(data.missing_values).reduce((a,b)=>a+b, 0);
                const healthBadge = document.getElementById('healthBadge');
                const healthText = document.getElementById('healthText');
                if(totalMissing === 0) {
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

                document.getElementById('loadingState').classList.add('d-none');
                document.getElementById('dashboardContent').classList.remove('d-none');
                document.getElementById('controlPanel').classList.remove('d-none');

                renderCharts(data.missing_values, data.correlation_matrix);
                if(data.dtype_counts) renderDtypesChart(data.dtype_counts);
                if(data.preview_data) renderPreviewTable(data.preview_data);

            } else { alert("Upload Failed: " + data.message); resetUI(); }
        } catch (err) { alert("Error: ไม่สามารถติดต่อ Backend ได้ กรุณาเช็คว่ารัน python api.py หรือยัง"); resetUI(); }
    }

    // ประมวลผลข้อมูล
    async function processData() {
        const mode = document.getElementById('cleanModeSelect').value;
        // ✅ แก้ไข: เลือก Action ให้ถูกต้อง (ถ้า AI ก็ส่ง ai_agent ไปเลย)
        let action = mode;
        if(mode === 'manual') {
            action = document.getElementById('manualAction').value;
        }
        
        document.getElementById('resultSection').classList.remove('d-none');
        document.getElementById('mlStudioSection').classList.add('d-none');
        const logArea = document.getElementById('logArea');
        logArea.innerHTML = "<span class='text-muted'>Initializing AI Engine...</span><br>";
        
        window.scrollTo(0, document.body.scrollHeight);

        const formData = new FormData();
        formData.append("filename", currentRawFile);
        formData.append("action", action);
        if(currentUser) formData.append("username", currentUser);

        try {
            const res = await fetch(`${getApiUrl()}/clean`, { 
                method: "POST", body: formData, headers: { "ngrok-skip-browser-warning": "true" }
            });
            const data = await res.json();

            if(data.status === "success") {
                currentCleanFile = data.clean_filename || `clean_${currentRawFile}`;

                document.getElementById('downloadBtn').onclick = () => {
                    downloadFile(`${getApiUrl()}${data.download_url}`, currentCleanFile);
                };

                if(currentUser) loadHistory();

                if(data.logs) {
                    data.logs.forEach((log, i) => {
                        setTimeout(() => {
                            let color = "text-white";
                            if(log.includes("AI")) color = "text-warning fw-bold"; 
                            if(log.includes("Drop") || log.includes("Removed")) color = "text-danger";
                            if(log.includes("Fill")) color = "text-info";
                            
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
        } catch (err) { logArea.innerHTML += `<div class="text-danger">> ERROR: ${err.message}</div>`; }
    }

    // ================= ML STUDIO (AutoML) =================
    function setupMLStudio() {
        document.getElementById('mlStudioSection').classList.remove('d-none');
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

        const targetSelect = document.getElementById('targetColumn');
        // ✅ เพิ่มตัวเลือก Unsupervised (Clustering)
        targetSelect.innerHTML = '<option value="" disabled selected>-- Select Column --</option>';
        targetSelect.innerHTML += '<option value="NONE" class="text-primary fw-bold">🔍 No Target (Unsupervised / Clustering)</option>'; 
        
        datasetColumns.forEach(col => {
            targetSelect.innerHTML += `<option value="${col}">${col}</option>`;
        });
    }

    async function trainModels() {
        const target = document.getElementById('targetColumn').value;
        if(!target) return alert("Please select a Target Variable first!");

        const btn = document.getElementById('trainBtn');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
        btn.disabled = true;

        document.getElementById('leaderboardWait').classList.add('d-none');
        document.getElementById('leaderboardResults').classList.add('d-none');

        const mode = document.getElementById('mlMode').value;
        const config = { models: [], scaling: document.getElementById('mlScaling').value };
        
        if (mode === 'manual') {
            document.querySelectorAll('.ml-model-cb:checked').forEach(cb => config.models.push(cb.value));
            
            // Map manual options to what backend expects
            if(config.models.includes("KNN Regressor")) config.models.push("KNN Classifier");
            if(config.models.includes("Linear Regression")) config.models.push("Logistic Regression");
            if(config.models.includes("SVM")) {
                 config.models.push("SVM (SVR)");
                 config.models.push("SVM (SVC)");
            }
            if(config.models.includes("Neural Network (MLP)")) {
                 config.models.push("Neural Network (MLP)");
            }
        }

        const formData = new FormData();
        formData.append("filename", currentCleanFile);
        formData.append("target_column", target);
        formData.append("mode", mode);
        formData.append("manual_config", JSON.stringify(config));

        try {
            const res = await fetch(`${getApiUrl()}/train_model`, {
                method: "POST", body: formData, headers: { "ngrok-skip-browser-warning": "true" }
            });
            const data = await res.json();

            if (data.status === "success") {
                document.getElementById('taskTypeBadge').innerText = data.task_type + " Task";
                const tbody = document.getElementById('leaderboardBody');
                tbody.innerHTML = "";
                
                data.leaderboard.forEach((item, index) => {
                    let medal = index === 0 ? "🏆" : (index === 1 ? "🥈" : "🥉");
                    let rankClass = index === 0 ? "fw-bold text-success bg-success-subtle" : "";
                    
                    tbody.innerHTML += `
                        <tr class="leaderboard-row ${rankClass}">
                            <td class="fw-bold">${medal} #${index + 1}</td>
                            <td>${item.model}</td>
                            <td class="fw-bold">${item.score}% <span class="text-muted fw-normal small">(${item.metric})</span></td>
                        </tr>
                    `;
                });

                document.getElementById('leaderboardResults').classList.remove('d-none');

                document.getElementById('downloadModelBtn').onclick = () => {
                    downloadFile(`${getApiUrl()}${data.download_url}`, 'best_model.pkl');
                };
            } else {
                alert("Training Error: " + data.message);
                document.getElementById('leaderboardWait').classList.remove('d-none');
            }
        } catch(err) {
            alert("Connection Error: " + err.message);
            document.getElementById('leaderboardWait').classList.remove('d-none');
        } finally {
            btn.innerHTML = '<i class="bi bi-play-circle-fill me-2"></i>Train Models';
            btn.disabled = false;
        }
    }

    // ================= กราฟทั้งหมด =================
    function renderCharts(missing, corr) {
        const mX = Object.keys(missing);
        const mY = Object.values(missing);
        if(mY.some(v => v > 0)) {
            Plotly.newPlot('missingChart', [{x: mX, y: mY, type: 'bar', marker: {color:'#ef4444'}}], {
                margin: {t:30,l:50,r:30,b:100}, xaxis:{tickangle:-45, automargin: true}, 
                paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
            }, {displayModeBar: false, responsive: true});
        } else {
            document.getElementById('missingChart').innerHTML = "<div class='text-center text-muted py-5'>✅ No missing values</div>";
        }
        
        if(corr) {
            Plotly.newPlot('heatmapChart', [{z: corr.z, x: corr.x, y: corr.y, type: 'heatmap', colorscale:'Viridis'}], {
                margin: {t:10,l:50,r:10,b:50}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
            }, {displayModeBar: false, responsive: true});
        } else {
            document.getElementById('heatmapChart').innerHTML = "<div class='text-center text-muted py-5'>Not enough numeric data</div>";
        }
    }

    function renderDtypesChart(dtypes) {
        if(!dtypes) return;
        const labels = Object.keys(dtypes);
        const values = Object.values(dtypes);
        const data = [{
            values: values, labels: labels, type: 'pie', hole: .4, textinfo: "label+percent",
            marker: { colors: ['#6366f1', '#10b981', '#f59e0b', '#ef4444'] }
        }];
        const layout = { margin: {t:0, l:0, r:0, b:0}, showlegend: true, height: 250, paper_bgcolor: 'rgba(0,0,0,0)' };
        Plotly.newPlot('dtypesChart', data, layout, {displayModeBar: false});
    }

    function renderPreviewTable(preview) {
        if(!preview) return;
        const thead = document.getElementById('tableHead');
        const tbody = document.getElementById('tableBody');
        
        let headHtml = "";
        preview.columns.forEach(col => { headHtml += `<th scope="col" class="fw-bold text-secondary text-nowrap">${col}</th>`; });
        thead.innerHTML = headHtml;
        
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
        document.getElementById('loadingState').classList.add('d-none');
        document.getElementById('emptyState').classList.remove('d-none');
    }
</script>

</body>
</html>