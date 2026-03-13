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
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from pydantic import BaseModel
from passlib.context import CryptContext
import google.generativeai as genai
import re 
import joblib

# ✅ IMPORT เพิ่มสำหรับ Clustering และ Supervised Models ใหม่
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, r2_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
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
# ⚙️ MONGODB CONNECT
# ==========================================
MONGO_URI = "mongodb+srv://gatunyus_db_user:qtJbLfftWqnA13Gx@intelink.w5cucyl.mongodb.net/?appName=Intelink"
try:
    # เชื่อมต่อ MongoDB
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    client.admin.command('ping') # เช็คว่าต่อติดไหม
    print("✅ Successfully connected to MongoDB!")
    
    # เลือกชื่อ Database
    db = client['intelink'] 
    
    # สร้าง Collection (เปรียบเสมือน Table ใน SQL)
    users_collection = db['users']
    history_collection = db['history']
    
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

ADMIN_SECRET_KEY = "MY_SECRET_1234"

# 🔴 ใส่ Gemini Key ของคุณที่นี่ (ต้องเป็น KEY ใหม่เท่านั้น)
GEMINI_API_KEY = "YOUR_GEMINI_KEY" 
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 🧠 AI LOGIC
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
        1. "standardize_text": List categorical columns where values have inconsistent casing.
        2. "fill_unknown": List STRING/OBJECT columns that have missing values.
        3. "fill_mean": List NUMERIC columns that have missing values.
        4. "remove_outliers": List NUMERIC columns that likely have outliers.
        5. "extract_numbers": List columns with units (e.g. "50kg") BUT IGNORE IDs/Dates/Emails.
        
        Return ONLY JSON structure:
        { "standardize_text": [], "fill_unknown": [], "fill_mean": [], "remove_outliers": [], "extract_numbers": [], "drop_duplicates": true }
        """
        
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
# 📝 MODELS & AUTH (แก้ไขให้ชื่อตรงกันแล้ว)
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
async def register(user: RegisterModel): # ✅ แก้จาก UserRegister เป็น RegisterModel
    existing_user = users_collection.find_one({"$or": [{"email": user.email}, {"username": user.username}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or Username already exists")

    hashed_password = pwd_context.hash(user.password)
    plan = "pro" if user.admin_key == ADMIN_SECRET_KEY else "free"

    new_user = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "plan": plan,
        "created_at": datetime.now()
    }
    users_collection.insert_one(new_user)
    return {"status": "success", "message": "User created"}

@app.post("/login")
async def login(req: LoginModel): # ✅ แก้จาก LoginRequest เป็น LoginModel
    user = users_collection.find_one({"email": req.email})
    if not user or not pwd_context.verify(req.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    return {
        "status": "success", 
        "username": user["username"], 
        "plan": user.get("plan", "free")
    }

# ==========================================
# 🗂️ FILE UPLOAD & HISTORY
# ==========================================
@app.get("/history")
async def get_history(username: str):
    try:
        histories = list(history_collection.find(
            {"username": username}, 
            {"_id": 0} 
        ).sort("date", -1).limit(10)) 
        
        return {"status": "success", "history": histories}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
# 🛠️ CLEANING LOGIC (ลบ MySQL ทิ้งและแทนด้วย MongoDB)
# ==========================================
@app.post("/clean")
async def clean_data(filename: str = Form(...), action: str = Form(...), username: str = Form(None)):
    file_path = os.path.join(TEMP_DIR, filename)
    
    if not os.path.exists(file_path):
        fallback_name = sanitize_filename(filename)
        file_path = os.path.join(TEMP_DIR, fallback_name)
        
    try: df = pd.read_csv(file_path)
    except: return {"status": "error", "message": f"File not found: {filename}"}

    df.replace(['Nan', 'nan', 'Null', 'null', 'NAN', 'N/A', 'n/a'], np.nan, inplace=True)
    logs = []
    
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
                if any(x in col.lower() for x in ['email', 'id', 'date', 'name', 'phone']): continue
                try:
                    df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logs.append(f"AI: Extracted numbers from '{col}'")
                except: pass

        for col in strategy.get("fill_unknown", []):
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
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

    clean_filename = f"clean_{filename}"
    clean_path = os.path.join(TEMP_DIR, clean_filename)
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].round(2)
        if df[col].notna().all() and df[col].apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype('Int64')

    df.to_csv(clean_path, index=False)

    # ✅ บันทึก History ลง MongoDB (แก้จาก MySQL)
    if username:
        try:
            history_doc = {
                "username": username,
                "filename": filename,
                "cleaned_filename": clean_filename,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            history_collection.insert_one(history_doc)
        except Exception as e: 
            print(f"MongoDB History Error: {e}")

    return {"status": "success", "download_url": f"/download/{clean_filename}", "logs": logs, "clean_filename": clean_filename}

@app.get("/download/{filename}")
async def download_file(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path): return FileResponse(path, media_type='text/csv', filename=filename)
    return {"error": "File not found"}

# ==========================================
# 🤖 TRAIN MODEL & PREDICT
# ==========================================
@app.post("/train_model")
async def train_model(
    filename: str = Form(...), target_column: str = Form(...), mode: str = Form("auto"), manual_config: str = Form("{}")
):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Cleaned file not found: {filename}"}
    
    df = pd.read_csv(file_path)
    is_unsupervised = (target_column == "NONE")
    
    if not is_unsupervised and target_column not in df.columns:
        return {"status": "error", "message": f"Target column '{target_column}' not found."}
    
    if not is_unsupervised:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df; y = None
    
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
            
    config = json.loads(manual_config)
    if mode == "manual" and "drop_columns" in config:
        X = X.drop(columns=[c for c in config["drop_columns"] if c in X.columns])
        
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    scaler = StandardScaler()
    if mode == "manual" and config.get("scaling") == "minmax": scaler = MinMaxScaler()
    
    encoding_strategy = config.get("encoding", "onehot")
    if encoding_strategy == "ordinal":
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    else:
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', scaler)])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])
        
    models = {}
    metric_name = ""

    if task_type == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000), "KNN Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42), "Naive Bayes": GaussianNB(),
            "SVM (SVC)": SVC(), "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=42)
        }
        metric_name = "Accuracy"
        
    elif task_type == "Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(), "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42), "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42), "SVM (SVR)": SVR(),
            "Neural Network (MLP)": MLPRegressor(max_iter=500, random_state=42)
        }
        metric_name = "R-Square"
        
    elif task_type == "Clustering":
        X_train = X 
        models = {"K-Means (k=3)": KMeans(n_clusters=3, random_state=42), "K-Means (k=5)": KMeans(n_clusters=5, random_state=42)}
        metric_name = "Silhouette Score"

    if mode == "manual" and "models" in config and len(config["models"]) > 0:
        selected = config["models"]
        filtered_models = {k: v for k, v in models.items() if any(sel in k for sel in selected)}
        if filtered_models: models = filtered_models

    leaderboard = []
    best_score = -float('inf'); best_pipeline = None; best_model_name = ""
    
    for name, model in models.items():
        try:
            if task_type == "Clustering":
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                clf.fit(X_train)
                X_transformed = clf.named_steps['preprocessor'].transform(X_train)
                labels = clf.named_steps['model'].labels_
                score = silhouette_score(X_transformed, labels) if len(set(labels)) > 1 else 0
                metrics = {"Silhouette": score}; metric = "Silhouette"
            else:
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if task_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    metrics = {
                        "Accuracy": round(score*100, 2), "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2),
                        "Recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2)
                    }
                    metric = "Accuracy"
                else:
                    score = r2_score(y_test, y_pred)
                    metrics = {
                        "R-Square": round(score*100, 2), "MAE": round(mean_absolute_error(y_test, y_pred), 2), "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
                    }
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
        
        predictions = model.predict(df)
        df['Predicted_Result'] = predictions
        
        res_name = f"pred_{data_file.filename}"
        df.to_csv(os.path.join(TEMP_DIR, res_name), index=False)
        
        pred_series = pd.Series(predictions)
        summary = {}
        is_classification = pred_series.dtype == 'object' or pred_series.nunique() <= 15
        
        if is_classification:
            counts = pred_series.value_counts().to_dict()
            summary['type'] = 'classification'
            summary['distribution'] = {str(k): int(v) for k, v in counts.items()}
            summary['insight_label'] = "Majority Class"
            summary['insight_value'] = str(pred_series.mode()[0])
        else:
            summary['type'] = 'regression'
            summary['distribution'] = {"Mean": float(pred_series.mean()), "Min": float(pred_series.min()), "Max": float(pred_series.max())}
            summary['insight_label'] = "Average Value"
            summary['insight_value'] = f"{pred_series.mean():.2f}"

        missing_count = df.isnull().sum().sum(); total_cells = df.size
        summary['data_quality'] = round(100 - ((missing_count / total_cells) * 100), 2) if total_cells > 0 else 100
        preview_data = df.head().replace({np.nan: None}).to_dict(orient='split')
        
        return { "status": "success", "download_url": f"/download/{res_name}", "preview": preview_data, "summary": summary, "total_rows": len(df) }
    except KeyError as ke: return {"status": "error", "message": f"ไม่พบข้อมูลคอลัมน์ {str(ke)} ในไฟล์ที่อัปโหลด"}
    except ValueError as ve:
        if "feature names" in str(ve).lower() or "number of features" in str(ve).lower(): return {"status": "error", "message": "จำนวนคอลัมน์ไม่ตรงกับตอนที่ใช้เทรนโมเดล"}
        return {"status": "error", "message": str(ve)}
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 API Server with MongoDB Started on Port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)