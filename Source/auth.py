#auth.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import mysql.connector
from pydantic import BaseModel
from passlib.context import CryptContext

# สร้าง Router เพื่อส่งกลับไปให้ api.py
router = APIRouter()

# ==========================================
# ⚙️ AUTH CONFIGURATION
# ==========================================

# ใช้ pbkdf2_sha256 เพื่อเลี่ยงปัญหา 72 bytes limit
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# 🔴 Database Config (MAMP Settings)
# ย้ายมาไว้ที่นี่เพราะมีแค่ระบบ Login ที่ใช้ Database
db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",          # <--- ลองลบคำว่า root ออก ให้เหลือแค่ฟันหนูว่างๆ
    "database": "smart_cleaner_ai",
    "port": 3306             # ลองเปลี่ยนเป็น 3306 ถ้า 8889 ไม่ได้
}

ADMIN_SECRET_KEY = "MY_SECRET_1234"

# ==========================================
# 📝 AUTH MODELS
# ==========================================

class RegisterModel(BaseModel):
    username: str
    email: str
    password: str
    admin_key: str = None

class LoginModel(BaseModel):
    email: str
    password: str

# ==========================================
# 👤 AUTH ENDPOINTS
# ==========================================

@router.post("/register")
async def register_user(user: RegisterModel):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # 1. เช็คอีเมลซ้ำ
        cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            return JSONResponse(status_code=400, content={"message": "Email already exists"})

        # 2. กำหนด Role
        role = "user"
        plan = "free"
        if user.admin_key == ADMIN_SECRET_KEY:
            role = "admin"
            plan = "pro"

        # 3. Hash Password
        hashed_password = pwd_context.hash(user.password)

        # 4. Insert
        sql = """
            INSERT INTO users (username, email, password_hash, role, plan) 
            VALUES (%s, %s, %s, %s, %s)
        """
        vals = (user.username, user.email, hashed_password, role, plan)
        cursor.execute(sql, vals)
        conn.commit()

        return {"status": "success", "message": f"User created as {role.upper()}"}

    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        return JSONResponse(status_code=500, content={"message": f"Database connect failed: {err}"})
    except Exception as e:
        print(f"❌ General Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@router.post("/login")
async def login_user(user: LoginModel):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1. หา User
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()

        if not db_user:
            return JSONResponse(status_code=400, content={"message": "User not found"})

        # 2. เช็ครหัสผ่าน
        if not pwd_context.verify(user.password, db_user['password_hash']):
            return JSONResponse(status_code=400, content={"message": "Wrong password"})
        
        # 3. อัปเดตเวลา Last Login
        update_sql = "UPDATE users SET last_login = NOW() WHERE id = %s"
        cursor.execute(update_sql, (db_user['id'],))
        conn.commit()

        # 4. ส่งค่ากลับ
        return {
            "status": "success",
            "username": db_user['username'],
            "role": db_user['role'],
            "plan": db_user['plan'],
            "token": "demo-token"
        }

    except Exception as e:
        print(f"❌ Login Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        if cursor: cursor.close()
        if conn: conn.close()