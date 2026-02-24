import warnings
# ปิด Warning ที่น่ารำคาญ
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
import mysql.connector
from pydantic import BaseModel
from passlib.context import CryptContext
import google.generativeai as genai
import re 
import joblib

# ✅ IMPORT เพิ่มสำหรับ Clustering และ Supervised Models ใหม่
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

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

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",          
    "database": "smart_cleaner_ai",
    "port": 3306             
}

ADMIN_SECRET_KEY = "MY_SECRET_1234"

# 🔴 ใส่ Gemini Key ของคุณที่นี่ (ต้องเป็น KEY ใหม่เท่านั้น)
GEMINI_API_KEY = "AIzaSyAJ9IFve73FmKweh3dE6xdLbKs2yzXmNPU" 
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 🧠 AI LOGIC (FIXED MODEL NAME)
# ==========================================
def ask_gemini_strategy(df):
    if "YOUR_GEMINI_KEY" in GEMINI_API_KEY or "YOUR_NEW_GEMINI_KEY" in GEMINI_API_KEY:
        return None, "API Key Missing"

    try:
        buffer = []
        buffer.append("DATASET SUMMARY:")
        buffer.append(f"- Columns: {list(df.columns)}")
        buffer.append(f"- Data Types: {df.dtypes.astype(str).to_dict()}")
        sample_str = df.head(10).to_string()
        buffer.append(f"- Sample Data (First 10 rows):\n{sample_str}")
        
        prompt = """
        Analyze this dataset and return a JSON cleaning strategy.
        CRITICAL INSTRUCTIONS:
        1. "standardize_text": List categorical columns where values have inconsistent casing (e.g. "Fiction" vs "FICTION").
        2. "fill_unknown": List STRING/OBJECT columns that have missing values.
        3. "fill_mean": List NUMERIC columns that have missing values.
        4. "remove_outliers": List NUMERIC columns that likely have outliers.
        5. "extract_numbers": List columns with units (e.g. "50kg") BUT IGNORE IDs/Dates/Emails.
        
        Return ONLY JSON structure:
        { "standardize_text": [], "fill_unknown": [], "fill_mean": [], "remove_outliers": [], "extract_numbers": [], "drop_duplicates": true }
        """
        
        # ✅ แก้ไข: ใช้ gemini-1.5-flash (ตัวมาตรฐานปัจจุบัน)
        print("👉 Calling gemini-2.5-flash...")
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content("\n".join(buffer) + "\n" + prompt)
        
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        if "{" in cleaned_text:
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}") + 1
            cleaned_text = cleaned_text[start:end]
            
        return json.loads(cleaned_text), None

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return None, str(e)

# ==========================================
# 📝 MODELS & AUTH
# ==========================================
class RegisterModel(BaseModel):
    username: str
    email: str
    password: str
    admin_key: str = None

class LoginModel(BaseModel):
    email: str
    password: str

@app.post("/register")
async def register_user(user: RegisterModel):
    conn = None; cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone(): return JSONResponse(status_code=400, content={"message": "Email already exists"})
        role = "user"; plan = "free"
        if user.admin_key == ADMIN_SECRET_KEY: role = "admin"; plan = "pro"
        hashed = pwd_context.hash(user.password)
        cursor.execute("INSERT INTO users (username, email, password_hash, role, plan) VALUES (%s, %s, %s, %s, %s)", (user.username, user.email, hashed, role, plan))
        conn.commit()
        return {"status": "success", "message": f"User created as {role.upper()}"}
    except Exception as e: return JSONResponse(status_code=500, content={"message": str(e)})
    finally: 
        if cursor: cursor.close()
        if conn: conn.close()

@app.post("/login")
async def login_user(user: LoginModel):
    conn = None; cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()
        if not db_user or not pwd_context.verify(user.password, db_user['password_hash']):
            return JSONResponse(status_code=400, content={"message": "Invalid credentials"})
        cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (db_user['id'],))
        conn.commit()
        return {"status": "success", "username": db_user['username'], "role": db_user['role'], "plan": db_user['plan'], "token": "demo-token"}
    except Exception as e: return JSONResponse(status_code=500, content={"message": str(e)})
    finally: 
        if cursor: cursor.close()
        if conn: conn.close()

@app.get("/history")
async def get_history(username: str):
    conn = None; cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user: return {"status": "error", "message": "User not found"}
        cursor.execute("SELECT original_filename, cleaned_filename, created_at FROM file_history WHERE user_id = %s ORDER BY created_at DESC LIMIT 10", (user['id'],))
        history = [{"filename": h['original_filename'], "cleaned_filename": h['cleaned_filename'], "date": h['created_at'].strftime("%Y-%m-%d %H:%M")} for h in cursor.fetchall()]
        return {"status": "success", "history": history}
    except Exception as e: return JSONResponse(status_code=500, content={"message": str(e)})
    finally: 
        if cursor: cursor.close()
        if conn: conn.close()

def sanitize_filename(filename: str) -> str:
    clean_name = re.sub(r'\s+', '_', filename)
    clean_name = re.sub(r'[^\w\.-]', '', clean_name)
    return clean_name

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    safe_filename = sanitize_filename(file.filename)
    file_path = os.path.join(TEMP_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        df = pd.read_csv(file_path)
        
        # ✅ Pre-clean: แปลงค่า "Nan", "null" ที่เป็น Text ให้เป็น NaN จริงๆ ก่อนส่งไปวิเคราะห์
        df.replace(['Nan', 'nan', 'Null', 'null', 'NAN', 'N/A', 'n/a'], np.nan, inplace=True)

        num_df = df.select_dtypes(include=['number'])
        corr = None
        if not num_df.empty and num_df.shape[1] > 1:
            corr = {"x": num_df.columns.tolist(), "y": num_df.columns.tolist(), "z": num_df.corr().replace({np.nan: None}).values.tolist()}
        
        preview = df.head(5).replace({np.nan: None}).to_dict(orient='split')
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        
        return {
            "status": "success", "filename": safe_filename, 
            "rows": df.shape[0], "columns": df.shape[1], 
            "missing_values": df.isnull().sum().to_dict(), "correlation_matrix": corr, "preview_data": preview, "dtype_counts": dtypes,
            "all_columns": df.columns.tolist()
        }
    except Exception as e: return {"status": "error", "message": str(e)}

# ==========================================
# 🛠️ CLEANING LOGIC
# ==========================================
@app.post("/clean")
async def clean_data(filename: str = Form(...), action: str = Form(...), username: str = Form(None)):
    file_path = os.path.join(TEMP_DIR, filename)
    
    # Fallback: ถ้าหาไฟล์ไม่เจอ ให้ลอง sanitize ชื่อดู
    if not os.path.exists(file_path):
        fallback_name = sanitize_filename(filename)
        file_path = os.path.join(TEMP_DIR, fallback_name)
        
    try: df = pd.read_csv(file_path)
    except: return {"status": "error", "message": f"File not found: {filename}"}

    # STEP 0: NORMALIZE
    df.replace(['Nan', 'nan', 'Null', 'null', 'NAN', 'N/A', 'n/a'], np.nan, inplace=True)
    logs = []
    
    # ✅ HARD RULE: ลบแถวที่หายเยอะเกินไป (>2 ค่า) ทันที
    rows_before_drop = len(df)
    df.dropna(thresh=len(df.columns) - 2, inplace=True)
    if len(df) < rows_before_drop:
        logs.append(f"🗑️ Hard Rule: Dropped {rows_before_drop - len(df)} rows with >2 missing values")

    strategy = None

    if action == "ai_agent":
        logs.append("🧠 Asking Gemini AI...")
        strategy, error_msg = ask_gemini_strategy(df)
        if strategy: logs.append(f"✅ AI Decision Loaded.")
        else: logs.append(f"⚠️ AI Failed: {error_msg}. Fallback to Auto-Smart."); action = "auto_smart"

    # --- EXECUTION ---
    if strategy:
        if strategy.get("drop_duplicates"):
            rows = len(df); df = df.drop_duplicates()
            if len(df) < rows: logs.append("AI: Removed duplicates")

        for col in strategy.get("drop_columns", []):
            if col in df.columns: df = df.drop(columns=[col]); logs.append(f"AI: Dropped '{col}'")

        for col in strategy.get("standardize_text", []):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                df[col].replace('Nan', np.nan, inplace=True)
                logs.append(f"AI: Standardized text in '{col}'")

        for col in strategy.get("extract_numbers", []):
            if col in df.columns:
                # ป้องกันไม่ให้ยุ่งกับ ID/Email
                if any(x in col.lower() for x in ['email', 'id', 'date', 'name', 'phone']): continue
                try:
                    df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logs.append(f"AI: Extracted numbers from '{col}'")
                except: pass

        for col in strategy.get("fill_unknown", []):
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                # เผื่อมีสตริง "Nan" หลงเหลือจากการแปลง
                df[col] = df[col].replace(['nan', 'Nan', 'NaN'], "Unknown")
                logs.append(f"AI: Filled '{col}' with 'Unknown'")

        for col in strategy.get("fill_mean", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                logs.append(f"AI: Filled '{col}' with Mean ({mean_val:.2f})")

        for col in strategy.get("remove_outliers", []):
            if col in df.select_dtypes(include=['number']).columns:
                Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75); IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                logs.append(f"AI: Removed outliers in '{col}'")


    elif action == "auto_smart":
        init_rows = len(df); df = df.drop_duplicates()
        if len(df) < init_rows: logs.append(f"Removed {init_rows - len(df)} duplicate rows")

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip().str.title()
            if any(x in col.lower() for x in ['email', 'id', 'date', 'name', 'phone']): continue
            try:
                extracted = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                numeric_conv = pd.to_numeric(extracted, errors='coerce')
                if numeric_conv.notna().sum() > (len(df) * 0.7):
                    df[col] = numeric_conv
                    logs.append(f"💰 Extracted numbers from '{col}'")
            except: pass

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median()
                    df[col] = df[col].fillna(val)
                    logs.append(f"🔧 Filled '{col}' with Median ({val:.2f})")
                else:
                    df[col] = df[col].fillna("Unknown")
                    logs.append(f"🔧 Filled '{col}' with 'Unknown'")

        for col in df.select_dtypes(include=['number']).columns:
            Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75); IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper); logs.append(f"✂️ Capped '{col}'")

    # Save
    clean_filename = f"clean_{filename}"
    clean_path = os.path.join(TEMP_DIR, clean_filename)
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].round(2)
        if df[col].notna().all() and df[col].apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype('Int64')

    df.to_csv(clean_path, index=False)

    if username:
        conn = None; cursor = None
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            if user:
                cursor.execute("INSERT INTO file_history (user_id, original_filename, cleaned_filename, file_path) VALUES (%s, %s, %s, %s)", (user['id'], filename, clean_filename, file_path))
                conn.commit()
        except: pass
        finally: 
            if cursor: cursor.close()
            if conn: conn.close()

    return {"status": "success", "download_url": f"/download/{clean_filename}", "logs": logs, "clean_filename": clean_filename}

@app.get("/download/{filename}")
async def download_file(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path): return FileResponse(path, media_type='text/csv', filename=filename)
    return {"error": "File not found"}

# ==========================================
# 🤖 TRAIN MODEL (COMPLETE VERSION)
# ==========================================
@app.post("/train_model")
async def train_model(
    filename: str = Form(...), target_column: str = Form(...), mode: str = Form("auto"), manual_config: str = Form("{}")
):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    from sklearn.impute import SimpleImputer
    
    # Imports for Models
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.naive_bayes import GaussianNB
    
    # Metrics
    from sklearn.metrics import accuracy_score, r2_score, silhouette_score
    import joblib

    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Cleaned file not found: {filename}"}
    
    df = pd.read_csv(file_path)
    
    is_unsupervised = (target_column == "NONE")
    
    if not is_unsupervised and target_column not in df.columns:
        return {"status": "error", "message": f"Target column '{target_column}' not found."}
    
    # Prepare X, y
    if not is_unsupervised:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
    
    # Determine Task Type
    task_type = ""
    if is_unsupervised:
        task_type = "Clustering"
    elif y.dtype == 'object' or y.nunique() <= 15:
        task_type = "Classification"
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        task_type = "Regression"
            
    # Config & Feature Engineering
    config = json.loads(manual_config)
    if mode == "manual" and "drop_columns" in config:
        X = X.drop(columns=[c for c in config["drop_columns"] if c in X.columns])
        
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    scaler = StandardScaler()
    if mode == "manual" and config.get("scaling") == "minmax": scaler = MinMaxScaler()
    
    # Encoding Selection
    encoding_strategy = config.get("encoding", "onehot")
    if encoding_strategy == "ordinal":
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    else:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', scaler)])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])
        
    models = {}
    metric_name = ""

    # ✅ Define Models based on Task Type
    if task_type == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "SVM (SVC)": SVC(),
            "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=42)
        }
        metric_name = "Accuracy"
        
    elif task_type == "Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVM (SVR)": SVR(),
            "Neural Network (MLP)": MLPRegressor(max_iter=500, random_state=42)
        }
        metric_name = "R-Square"
        
    elif task_type == "Clustering":
        X_train = X 
        models = {
            "K-Means (k=3)": KMeans(n_clusters=3, random_state=42),
            "K-Means (k=5)": KMeans(n_clusters=5, random_state=42)
        }
        metric_name = "Silhouette Score"

    # Filter models if manual mode
    if mode == "manual" and "models" in config and len(config["models"]) > 0:
        selected = config["models"]
        filtered_models = {}
        for k, v in models.items():
            # Check partial match (e.g. "KNN" matches "KNN Classifier")
            if any(sel in k for sel in selected):
                filtered_models[k] = v
        if filtered_models:
            models = filtered_models

    leaderboard = []
    best_score = -float('inf'); best_pipeline = None; best_model_name = ""
    
    for name, model in models.items():
        try:
            if task_type == "Clustering":
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                clf.fit(X_train)
                # Silhouette Score needs raw X, labels
                X_transformed = clf.named_steps['preprocessor'].transform(X_train)
                labels = clf.named_steps['model'].labels_
                if len(set(labels)) > 1:
                    score = silhouette_score(X_transformed, labels)
                else:
                    score = 0
                metrics = {"Silhouette": score}
                metric = "Silhouette"
            else:
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if task_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    metrics = {
                        "Accuracy": round(score*100, 2),
                        "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2),
                        "Recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2)
                    }
                    metric = "Accuracy"
                else:
                    score = r2_score(y_test, y_pred)
                    metrics = {"R-Square": round(score*100, 2)}
                    metric = "R-Square"

            leaderboard.append({"model": name, "score": round(score*100, 2), "metrics": metrics, "metric": metric})
            
            if score > best_score: 
                best_score = score; best_model_name = name; best_pipeline = clf
        except Exception as e: print(f"⚠️ Error {name}: {e}")
            
    leaderboard = sorted(leaderboard, key=lambda x: x['score'], reverse=True)
    
    model_filename = f"model_{filename.replace('.csv', '')}.pkl"
    model_path = os.path.join(TEMP_DIR, model_filename)
    if best_pipeline: joblib.dump(best_pipeline, model_path)
    
    return {"status": "success", "task_type": task_type, "leaderboard": leaderboard, "best_model": best_model_name, "download_url": f"/download_model/{model_filename}"}

@app.get("/download_model/{filename}")
async def download_model(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path): return FileResponse(path, media_type='application/octet-stream', filename=filename)
    return {"error": "Model not found"}

@app.post("/predict")
async def predict_from_model(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    try:
        m_path = os.path.join(TEMP_DIR, model_file.filename); d_path = os.path.join(TEMP_DIR, data_file.filename)
        with open(m_path, "wb") as b: shutil.copyfileobj(model_file.file, b)
        with open(d_path, "wb") as b: shutil.copyfileobj(data_file.file, b)
        
        model = joblib.load(m_path)
        df = pd.read_csv(d_path)
        
        # 1. พยากรณ์ผล
        predictions = model.predict(df)
        df['Predicted_Result'] = predictions
        
        # 2. บันทึกไฟล์ที่พยากรณ์แล้ว
        res_name = f"pred_{data_file.filename}"
        df.to_csv(os.path.join(TEMP_DIR, res_name), index=False)
        
        # 3. 📊 สรุปข้อมูลสำหรับทำ Dashboard (ใหม่)
        pred_series = pd.Series(predictions)
        summary = {}
        # เช็คว่าเป็น Classification หรือ Regression
        is_classification = pred_series.dtype == 'object' or pred_series.nunique() <= 15
        
        if is_classification:
            counts = pred_series.value_counts().to_dict()
            summary['type'] = 'classification'
            summary['distribution'] = {str(k): int(v) for k, v in counts.items()}
            # หาค่าที่เยอะที่สุด (Majority Class)
            summary['insight_label'] = "Majority Class"
            summary['insight_value'] = str(pred_series.mode()[0])
        else:
            summary['type'] = 'regression'
            summary['distribution'] = {"Mean": float(pred_series.mean()), "Min": float(pred_series.min()), "Max": float(pred_series.max())}
            summary['insight_label'] = "Average Value"
            summary['insight_value'] = f"{pred_series.mean():.2f}"

        # ประเมินคุณภาพข้อมูล (เปอร์เซ็นต์ที่ไม่มีค่าว่าง)
        missing_count = df.isnull().sum().sum()
        total_cells = df.size
        data_quality_match = round(100 - ((missing_count / total_cells) * 100), 2) if total_cells > 0 else 100
        summary['data_quality'] = data_quality_match

        # 4. แปลง NaN ให้เป็น None สำหรับ JSON (แก้บั๊ก 500)
        preview_data = df.head().replace({np.nan: None}).to_dict(orient='split')
        
        return {
            "status": "success", 
            "download_url": f"/download/{res_name}", 
            "preview": preview_data,
            "summary": summary,
            "total_rows": len(df)
        }
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 API Server (Filename Sanitized & Fixed Logic) Started on Port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)