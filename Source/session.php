// session.php
<?php
// ตั้งค่า Session Timeout เป็น 3 ชั่วโมง (10800 วินาที)
ini_set('session.gc_maxlifetime', 10800);
session_set_cookie_params(10800);

session_start();

// 1. ฟังก์ชันตรวจสอบ Timeout (3 ชั่วโมง)
function check_session_timeout() {
    $timeout_duration = 10800; // 3 ชั่วโมง

    if (isset($_SESSION['last_activity']) && (time() - $_SESSION['last_activity']) > $timeout_duration) {
        // ถ้าเกินเวลา ให้เคลียร์ Session ทิ้ง
        session_unset();
        session_destroy();
        header("Location: login.php?timeout=true");
        exit();
    }
    // อัปเดตเวลาล่าสุดที่มีการใช้งาน
    $_SESSION['last_activity'] = time();
}

// 2. ฟังก์ชันสร้าง CSRF Token
function csrf_token(): string {
    if (empty($_SESSION['csrf'])) {
        try {
            $_SESSION['csrf'] = bin2hex(random_bytes(32));
        } catch (Exception $e) {
            $_SESSION['csrf'] = bin2hex(openssl_random_pseudo_bytes(32));
        }
    }
    return $_SESSION['csrf'];
}

// 3. ฟังก์ชันตรวจ CSRF Token
function csrf_check($token): bool {
    if (!isset($_SESSION['csrf']) || empty($token)) {
        return false;
    }
    return hash_equals($_SESSION['csrf'], $token);
}
?>