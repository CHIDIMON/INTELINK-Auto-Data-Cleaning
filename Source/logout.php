// logout.php
<?php
// 1. ต้องประกาศ session_start() ก่อนเสมอ ไม่งั้นจะลบไม่ได้!
session_start();

// 2. ล้างตัวแปร Session ทั้งหมด
$_SESSION = array();

// 3. ลบ Session Cookie (เพื่อความสะอาดหมดจด)
if (ini_get("session.use_cookies")) {
    $params = session_get_cookie_params();
    setcookie(session_name(), '', time() - 42000,
        $params["path"], $params["domain"],
        $params["secure"], $params["httponly"]
    );
}

// 4. ทำลาย Session ทิ้ง
session_destroy();

// 5. ส่งกลับไปหน้า index.php (จะเป็น Guest Mode ทันที)
header("Location: index.php");
exit();
?>