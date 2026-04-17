"""
全域 debug print 攔截器。
所有模組 import 此模組後，使用 `print = debug_print` 將輸出重定向到 crops/debug_log.txt。
"""
import os
import builtins

_DEBUG_LOG_PATH = "crops/debug_log.txt"
_initialized = False

def _ensure_init():
    global _initialized
    if _initialized:
        return
    _initialized = True
    if os.path.exists(_DEBUG_LOG_PATH):
        try:
            os.remove(_DEBUG_LOG_PATH)
        except Exception:
            pass

def debug_print(*args, **kwargs):
    _ensure_init()
    os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
    with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        builtins.print(*args, **kwargs, file=f)
