// login_action.php
<?php
require_once 'session.php';

header('Content-Type: application/json');

// รับข้อมูล JSON จาก JavaScript
$data = json_decode(file_get_contents('php://input'), true);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $token = $data['csrf_token'] ?? '';
    $userData = $data['user'] ?? null;
    $plan = $data['plan'] ?? 'free';

    // 1. ตรวจสอบ CSRF Token
    if (!csrf_check($token)) {
        http_response_code(403);
        echo json_encode(['status' => 'error', 'message' => 'CSRF Token Invalid']);
        exit;
    }

    // 2. สร้าง Session จริงๆ ฝั่ง PHP
    if ($userData) {
        $_SESSION['user_logged_in'] = true;
        $_SESSION['username'] = $userData;
        $_SESSION['plan'] = $plan;
        $_SESSION['last_activity'] = time(); // เริ่มนับเวลา

        echo json_encode(['status' => 'success']);
    } else {
        echo json_encode(['status' => 'error', 'message' => 'No user data']);
    }
}
?>