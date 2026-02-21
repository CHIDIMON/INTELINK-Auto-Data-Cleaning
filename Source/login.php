// login.php
<?php require_once 'session.php'; ?>
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Smart AI Data Cleaner</title>
    
    <!-- Bootstrap 5 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">

    <style>
        :root { --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); --bg-color: #f8fafc; }
        body { font-family: 'Sarabun', sans-serif; background-color: var(--bg-color); height: 100vh; overflow: hidden; }
        h1, h2, h3, h4, h5, .brand-font { font-family: 'Outfit', sans-serif; }
        
        .login-container { height: 100vh; display: flex; }
        
        /* Left Side */
        .brand-side {
            flex: 1; background: var(--primary-gradient); position: relative;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            color: white; overflow: hidden;
        }
        .circle { position: absolute; border-radius: 50%; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); }
        .c1 { width: 300px; height: 300px; top: -50px; left: -50px; }
        .c2 { width: 500px; height: 500px; bottom: -100px; right: -100px; }
        .brand-content { z-index: 10; text-align: center; padding: 2rem; }

        /* Right Side */
        .form-side {
            flex: 1; display: flex; justify-content: center; align-items: center;
            background: white; padding: 2rem; position: relative;
        }
        .login-card { width: 100%; max-width: 400px; padding: 2rem; animation: slideUp 0.5s ease-out; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        .form-floating > .form-control { border-radius: 12px; border: 1px solid #e2e8f0; background-color: #f8fafc; }
        .form-floating > .form-control:focus { border-color: #6366f1; background-color: white; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1); }
        
        .btn-login {
            background: var(--primary-gradient); border: none; border-radius: 12px;
            padding: 14px; font-weight: 600; width: 100%; transition: 0.3s; color: white;
        }
        .btn-login:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3); opacity: 0.95; }
        
        .social-btn {
            border: 1px solid #e2e8f0; background: white; color: #475569; padding: 10px;
            border-radius: 10px; transition: 0.2s;
        }
        .social-btn:hover { background: #f1f5f9; transform: translateY(-1px); }

        @media (max-width: 768px) { .brand-side { display: none; } .form-side { background: var(--bg-color); } }
    </style>
</head>
<body>

<div class="login-container">
    <!-- ✅ เก็บ CSRF Token -->
    <input type="hidden" id="csrfToken" value="<?php echo csrf_token(); ?>">

    <!-- Brand Side -->
    <div class="brand-side">
        <div class="circle c1"></div>
        <div class="circle c2"></div>
        <div class="brand-content">
            <div class="bg-white text-primary rounded-3 p-3 d-inline-flex mb-4 shadow-lg">
                <i class="bi bi-stars display-4"></i>
            </div>
            <h1 class="fw-bold brand-font display-4 mb-2">SmartClean AI</h1>
            <p class="fs-5 opacity-75">Automate your data preprocessing<br>with the power of AI.</p>
        </div>
    </div>

    <!-- Form Side -->
    <div class="form-side">
        <div class="login-card">
            <div class="text-center mb-5">
                <h2 class="fw-bold text-dark brand-font">Welcome Back</h2>
                <p class="text-muted">Please enter your details to sign in.</p>
            </div>

            <form onsubmit="handleLogin(event)">
                <div class="form-floating mb-3">
                    <input type="email" class="form-control" id="emailInput" placeholder="name@example.com" required>
                    <label class="text-muted"><i class="bi bi-envelope me-2"></i>Email Address</label>
                </div>

                <div class="form-floating mb-4">
                    <input type="password" class="form-control" id="passwordInput" placeholder="Password" required>
                    <label class="text-muted"><i class="bi bi-lock me-2"></i>Password</label>
                </div>

                <button type="submit" class="btn btn-login mb-3">Sign In</button>

                <div class="d-flex align-items-center mb-4">
                    <hr class="flex-grow-1 text-muted opacity-25">
                    <span class="px-3 text-muted small">Or continue with</span>
                    <hr class="flex-grow-1 text-muted opacity-25">
                </div>

                <div class="row g-2 mb-4">
                    <div class="col-6">
                        <button type="button" class="btn w-100 social-btn fw-bold" onclick="alert('Coming Soon')">
                            <i class="bi bi-google me-2 text-danger"></i> Google
                        </button>
                    </div>
                    <div class="col-6">
                        <button type="button" class="btn w-100 social-btn fw-bold" onclick="alert('Coming Soon')">
                            <i class="bi bi-github me-2 text-dark"></i> GitHub
                        </button>
                    </div>
                </div>

                <div class="text-center">
                    <p class="text-muted small">Don't have an account yet? 
                        <a href="register.php" class="text-primary fw-bold text-decoration-none">Create an account</a>
                    </p>
                    <!-- <p class="small mt-2"><a href="admin_register.php" class="text-secondary text-decoration-none">Admin Access</a></p> -->
                </div>
            </form>
        </div>
    </div>
</div>

<script>
    // ⚠️ อย่าลืมเช็ค Port (127.0.0.1:8000)
    const API_URL = "http://127.0.0.1:8000"; 

    async function handleLogin(e) {
        e.preventDefault();
        const btn = document.querySelector('.btn-login');
        const originalText = btn.innerText;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Checking...';
        btn.disabled = true;

        const csrfToken = document.getElementById('csrfToken').value;

        const formData = {
            email: document.getElementById('emailInput').value,
            password: document.getElementById('passwordInput').value
        };

        try {
            // 1. เช็ค Login กับ Python
            const res = await fetch(`${API_URL}/login`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });
            const data = await res.json();

            if (res.ok) {
                // 2. ส่งข้อมูลไปฝาก Session ที่ PHP
                const sessionRes = await fetch('login_action.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        csrf_token: csrfToken, 
                        user: data.username,
                        plan: data.plan
                    })
                });
                
                const sessionData = await sessionRes.json();

                if (sessionData.status === 'success') {
                    // จัดกึ่งกลางปุ่ม Success
                    btn.innerHTML = '<div class="d-flex align-items-center justify-content-center w-100"><i class="bi bi-check-lg me-2"></i>Success!</div>';
                    btn.classList.replace('btn-login', 'btn-success');
                    setTimeout(() => {
                        window.location.href = "index.php"; // ไปหน้า Index (PHP จะเช็ค Session เอง)
                    }, 800);
                } else {
                    throw new Error("Session Error: " + sessionData.message);
                }
            } else {
                alert("Login Failed: " + data.message);
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        } catch (err) {
            alert("Connection Error: " + err.message + "\n(เช็คว่ารัน Python api.py หรือยัง?)");
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }
</script>
</body>
</html>