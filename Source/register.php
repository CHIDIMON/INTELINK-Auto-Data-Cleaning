// register.php
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Account - SmartClean AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
    
    <style>
        /* CSS ชุดเดียวกับ Login.php เพื่อให้ธีมเหมือนกัน */
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
        
        .btn-primary-gradient {
            background: var(--primary-gradient); border: none; border-radius: 12px;
            padding: 14px; font-weight: 600; width: 100%; transition: 0.3s; color: white;
        }
        .btn-primary-gradient:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3); opacity: 0.95; }

        @media (max-width: 768px) { .brand-side { display: none; } .form-side { background: var(--bg-color); } }
    </style>
</head>
<body>

<div class="login-container">
    <!-- LEFT SIDE: BRANDING -->
    <div class="brand-side">
        <div class="circle c1"></div>
        <div class="circle c2"></div>
        <div class="brand-content">
            <div class="bg-white text-primary rounded-3 p-3 d-inline-flex mb-4 shadow-lg">
                <i class="bi bi-person-plus-fill display-4"></i>
            </div>
            <h1 class="fw-bold brand-font display-4 mb-2">Join Us</h1>
            <p class="fs-5 opacity-75">Start cleaning your data with AI today.<br>Fast, simple, and secure.</p>
        </div>
    </div>

    <!-- RIGHT SIDE: REGISTER FORM -->
    <div class="form-side">
        <div class="login-card">
            <div class="text-center mb-4">
                <h2 class="fw-bold text-dark brand-font">Create Account</h2>
                <p class="text-muted">Enter your details to get started.</p>
            </div>

            <form onsubmit="handleRegister(event)">
                <!-- Username -->
                <div class="form-floating mb-3">
                    <input type="text" class="form-control" id="usernameInput" placeholder="Username" required>
                    <label class="text-muted"><i class="bi bi-person me-2"></i>Username</label>
                </div>

                <!-- Email -->
                <div class="form-floating mb-3">
                    <input type="email" class="form-control" id="emailInput" placeholder="name@example.com" required>
                    <label class="text-muted"><i class="bi bi-envelope me-2"></i>Email Address</label>
                </div>

                <!-- Password -->
                <div class="form-floating mb-4">
                    <input type="password" class="form-control" id="passwordInput" placeholder="Password" required>
                    <label class="text-muted"><i class="bi bi-lock me-2"></i>Password</label>
                </div>

                <button type="submit" class="btn btn-primary-gradient mb-3">Sign Up</button>

                <div class="text-center">
                    <p class="text-muted small">Already have an account? 
                        <a href="login.php" class="text-primary fw-bold text-decoration-none">Sign In</a>
                    </p>
                </div>
            </form>
        </div>
    </div>
</div>

<script>
    const API_URL = "http://127.0.0.1:8000"; 

    async function handleRegister(e) {
        e.preventDefault();
        const btn = document.querySelector('button[type="submit"]');
        const originalText = btn.innerText;
        btn.innerText = "Creating...";
        btn.disabled = true;

        const formData = {
            username: document.getElementById('usernameInput').value,
            email: document.getElementById('emailInput').value,
            password: document.getElementById('passwordInput').value,
            admin_key: "" // User ทั่วไปไม่ต้องส่ง Admin Key
        };

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });
            const data = await res.json();

            if (res.ok) {
                alert("✅ Account created successfully!");
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