import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import shutil
import os
import json
import sqlite3
from datetime import datetime
from pydantic import BaseModel
from passlib.context import CryptContext
import google.generativeai as genai
import re 
import joblib

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.cluster import KMeans

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
DB_NAME = "smart_cleaner.db"

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
ADMIN_SECRET_KEY = "MY_SECRET_1234"

# 🔴 ใส่ Gemini Key ของคุณที่นี่
GEMINI_API_KEY = "API_KEY" 
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 🛠️ DATABASE UTILS
# ==========================================
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT UNIQUE, password_hash TEXT, role TEXT DEFAULT 'user', plan TEXT DEFAULT 'free', last_login DATETIME, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS file_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, original_filename TEXT, cleaned_filename TEXT, file_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit(); conn.close()

init_db()

# ==========================================
# 🧠 AI & HELPER
# ==========================================
def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^\w\.-]', '', re.sub(r'\s+', '_', filename))

def ask_gemini_strategy(df):
    try:
        buffer = [f"- Columns: {list(df.columns)}", f"- Dtypes: {df.dtypes.astype(str).to_dict()}", f"- Sample:\n{df.head(10).to_string()}"]
        prompt = """Analyze and return JSON strategy: { "standardize_text": [], "fill_unknown": [], "fill_mean": [], "remove_outliers": [], "extract_numbers": [], "drop_duplicates": true }"""
        model = genai.GenerativeModel('gemini-2.0-flash')
        res = model.generate_content("\n".join(buffer) + "\n" + prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end]), None
    except Exception as e: return None, str(e)

# ==========================================
# 📝 AUTH ENDPOINTS
# ==========================================
class RegisterModel(BaseModel):
    username: str; email: str; password: str; admin_key: str = None
class LoginModel(BaseModel):
    email: str; password: str

@app.post("/register")
async def register(u: RegisterModel):
    conn = get_db_connection(); c = conn.cursor()
    try:
        if c.execute("SELECT id FROM users WHERE email=?", (u.email,)).fetchone(): return JSONResponse(400, {"message": "Email exists"})
        role = "admin" if u.admin_key == ADMIN_SECRET_KEY else "user"
        c.execute("INSERT INTO users (username, email, password_hash, role) VALUES (?,?,?,?)", (u.username, u.email, pwd_context.hash(u.password), role))
        conn.commit(); return {"status": "success"}
    finally: conn.close()

@app.post("/login")
async def login(u: LoginModel):
    conn = get_db_connection(); c = conn.cursor()
    try:
        user = c.execute("SELECT * FROM users WHERE email=?", (u.email,)).fetchone()
        if not user or not pwd_context.verify(u.password, user['password_hash']): return JSONResponse(400, {"message": "Invalid login"})
        c.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now(), user['id'])); conn.commit()
        return {"status": "success", "username": user['username'], "role": user['role'], "plan": user['plan']}
    finally: conn.close()

# ==========================================
# 📂 DATA ENDPOINTS (Clean with Fallback)
# ==========================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    fname = sanitize_filename(file.filename); fpath = os.path.join(TEMP_DIR, fname)
    with open(fpath, "wb") as b: shutil.copyfileobj(file.file, b)
    try:
        df = pd.read_csv(fpath); df.replace(['Nan','nan','Null','null','NAN'], np.nan, inplace=True)
        num = df.select_dtypes(include=['number'])
        corr = {"z": num.corr().replace({np.nan:None}).values.tolist(), "x": num.columns.tolist(), "y": num.columns.tolist()} if not num.empty and num.shape[1]>1 else None
        return {"status": "success", "filename": fname, "rows": df.shape[0], "columns": df.shape[1], "missing_values": df.isnull().sum().to_dict(), "correlation_matrix": corr, "preview_data": df.head().replace({np.nan:None}).to_dict(orient='split'), "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(), "all_columns": df.columns.tolist()}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/clean")
async def clean(filename: str=Form(...), action: str=Form(...), username: str=Form(None)):
    fpath = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(fpath): return {"status": "error", "message": "File not found"}
    df = pd.read_csv(fpath); logs = []
    
    # AI Logic with Fallback
    ai_status = "ok"
    ai_error_msg = None
    strategy = None

    if action == "ai_agent":
        logs.append("🧠 Asking Gemini AI...")
        strategy, error = ask_gemini_strategy(df)
        if not strategy: 
            action = "auto_smart"
            ai_status = "failed"
            ai_error_msg = error if error else "Unknown Error"
            logs.append(f"⚠️ AI Failed: {ai_error_msg}")
            logs.append("🔄 Switching to Auto-Smart (Rule-based)...")

    if strategy:
        if strategy.get("drop_duplicates"): df.drop_duplicates(inplace=True); logs.append("AI: Deduped")
        for c in strategy.get("fill_mean", []): 
            if c in df: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].mean()); logs.append(f"AI: Fill Mean {c}")
        for c in strategy.get("fill_unknown", []):
            if c in df: df[c] = df[c].fillna("Unknown"); logs.append(f"AI: Fill Unknown {c}")
    elif action == "auto_smart":
        df.drop_duplicates(inplace=True)
        for c in df.select_dtypes(include='number'): df[c] = df[c].fillna(df[c].median()); logs.append(f"Filled Median {c}")
        for c in df.select_dtypes(include='object'): df[c] = df[c].fillna("Unknown"); logs.append(f"Filled Unknown {c}")

    clean_name = f"clean_{sanitize_filename(filename)}"
    df.to_csv(os.path.join(TEMP_DIR, clean_name), index=False)
    
    if username and username != "Guest":
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if user: 
            conn.execute("INSERT INTO file_history (user_id, original_filename, cleaned_filename, file_path) VALUES (?,?,?,?)", (user['id'], filename, clean_name, fpath))
            conn.commit()
        conn.close()

    return {
        "status": "success", "download_url": f"/download/{clean_name}", 
        "logs": logs, "clean_filename": clean_name,
        "ai_status": ai_status, "ai_error": ai_error_msg
    }

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path): return FileResponse(path, media_type='text/csv', filename=filename)
    return JSONResponse(404, {"message": "File not found"})

@app.get("/history")
async def history(username: str):
    conn = get_db_connection()
    user = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if not user: return {"status": "error"}
    rows = conn.execute("SELECT * FROM file_history WHERE user_id=? ORDER BY created_at DESC LIMIT 10", (user['id'],)).fetchall()
    conn.close()
    return {"status": "success", "history": [{"filename": r['original_filename'], "cleaned_filename": r['cleaned_filename'], "date": str(r['created_at'])[:16]} for r in rows]}

# ==========================================
# 🚀 REAL TRAIN MODEL
# ==========================================
@app.post("/train_model")
async def train_model(filename: str = Form(...), target_column: str = Form(...), mode: str = Form("auto"), manual_config: str = Form("{}")):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path): return {"status": "error", "message": "File not found"}
    df = pd.read_csv(file_path)
    
    is_unsupervised = (target_column == "NONE")
    if is_unsupervised:
        X = df; y = None; task_type = "Clustering"
    else:
        if target_column not in df.columns: return {"status": "error", "message": "Target not found"}
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column]); y = df[target_column]
        if y.dtype == 'object' or y.nunique() <= 20:
            task_type = "Classification"; le = LabelEncoder(); y = le.fit_transform(y)
        else: task_type = "Regression"

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    config = json.loads(manual_config)
    scaler = MinMaxScaler() if mode == "manual" and config.get("scaling") == "minmax" else StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', scaler)]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    models = {}
    if task_type == "Classification":
        models = { "Logistic Regression": LogisticRegression(max_iter=500), "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier(), "KNN": KNeighborsClassifier() }
        metric_name = "Accuracy"
    elif task_type == "Regression":
        models = { "Linear Regression": LinearRegression(), "Decision Tree": DecisionTreeRegressor(), "Random Forest": RandomForestRegressor(), "SVR": SVR() }
        metric_name = "R2 Score"
    else:
        models = { "K-Means (k=3)": KMeans(n_clusters=3), "K-Means (k=5)": KMeans(n_clusters=5) }
        metric_name = "Silhouette"

    if mode == "manual" and "models" in config and config["models"]:
        selected = config["models"]
        models = {k: v for k, v in models.items() if any(s in k for s in selected)}

    leaderboard = []
    best_score = -1; best_pipeline = None; best_model_name = ""
    if task_type != "Clustering": X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else: X_train = X

    for name, model in models.items():
        try:
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            if task_type != "Clustering":
                clf.fit(X_train, y_train); y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred) if task_type == "Classification" else r2_score(y_test, y_pred)
            else:
                clf.fit(X_train); labels = clf.named_steps['model'].labels_
                X_proc = clf.named_steps['preprocessor'].transform(X_train)
                score = silhouette_score(X_proc, labels) if len(set(labels)) > 1 else 0
            
            leaderboard.append({"model": name, "score": round(score * 100, 2), "metric": metric_name})
            if score > best_score: best_score = score; best_pipeline = clf; best_model_name = name
        except Exception as e: print(f"Error {name}: {e}")

    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    model_filename = f"model_{sanitize_filename(filename)}.pkl"
    if best_pipeline: joblib.dump(best_pipeline, os.path.join(TEMP_DIR, model_filename))

    return {"status": "success", "task_type": task_type, "leaderboard": leaderboard, "best_model": best_model_name, "download_url": f"/download_model/{model_filename}"}

@app.get("/download_model/{filename}")
async def download_model(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path): return FileResponse(path, media_type='application/octet-stream', filename=filename)
    return JSONResponse(404, {"message": "Model not found"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)