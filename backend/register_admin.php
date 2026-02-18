<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Smart AI Data Cleaner</title>
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">

    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --bg-color: #f8fafc;
        }

        body {
            font-family: 'Sarabun', sans-serif;
            background-color: var(--bg-color);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .register-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            overflow: hidden;
            width: 100%;
            max-width: 900px;
            display: flex;
            flex-direction: row;
        }

        .brand-side {
            background: var(--primary-gradient);
            width: 40%;
            padding: 40px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            text-align: center;
        }

        .form-side {
            width: 60%;
            padding: 50px;
        }

        .form-control {
            border-radius: 10px;
            padding: 12px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .form-control:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .btn-register {
            background: var(--primary-gradient);
            border: none;
            padding: 12px;
            border-radius: 10px;
            font-weight: bold;
            color: white;
            width: 100%;
            transition: 0.3s;
        }
        .btn-register:hover { opacity: 0.9; transform: translateY(-2px); color: white; }

        @media (max-width: 768px) {
            .register-card { flex-direction: column; margin: 20px; }
            .brand-side { width: 100%; padding: 30px; }
            .form-side { width: 100%; padding: 30px; }
        }
    </style>
</head>
<body>

<div class="register-card">
    <div class="brand-side">
        <i class="bi bi-shield-lock-fill display-1 mb-3"></i>
        <h2 class="fw-bold" style="font-family: 'Outfit'">Admin Access</h2>
        <p class="opacity-75">Create your account to manage<br>data and users.</p>
    </div>
    
    <div class="form-side">
        <h3 class="fw-bold text-dark mb-4">Create Account</h3>
        
        <form id="registerForm" onsubmit="handleRegister(event)">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label small text-muted">Username</label>
                    <input type="text" class="form-control" id="username" required>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label small text-muted">Email</label>
                    <input type="email" class="form-control" id="email" required>
                </div>
            </div>

            <div class="mb-3">
                <label class="form-label small text-muted">Password</label>
                <input type="password" class="form-control" id="password" required>
            </div>

            <div class="mb-4">
                <label class="form-label small text-muted fw-bold text-danger">
                    <i class="bi bi-key-fill me-1"></i>Admin Secret Key (Optional)
                </label>
                <input type="password" class="form-control border-danger-subtle" id="adminKey" placeholder="Leave empty for normal user">
                <!-- <div class="form-text text-muted" style="font-size: 0.75rem;">
                    * ใส่รหัสลับเฉพาะเพื่อเป็น Admin (เช่น: MY_SECRET_123)
                </div> -->
            </div>

            <button type="submit" class="btn btn-register mb-3">Register Now</button>
        </form>
    </div>
</div>

<script>
    // ⚠️ แก้ไข URL ตามของเครื่องคุณ
    const API_URL = "http://127.0.0.1:8000"; 
    // หรือถ้าใช้ Ngrok: const API_URL = "https://xxxx.ngrok-free.app";

    async function handleRegister(e) {
        e.preventDefault();
        
        const btn = document.querySelector('.btn-register');
        const originalText = btn.innerText;
        btn.innerText = "Creating Account...";
        btn.disabled = true;

        const formData = {
            username: document.getElementById('username').value,
            email: document.getElementById('email').value,
            password: document.getElementById('password').value,
            admin_key: document.getElementById('adminKey').value // ส่ง Secret Key ไปเช็ค
        };

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });

            const data = await res.json();

            if (res.ok) {
                alert("✅ Registration Successful! Please login.");
                window.location.href = "login.php";
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
</script>

</body>
</html>