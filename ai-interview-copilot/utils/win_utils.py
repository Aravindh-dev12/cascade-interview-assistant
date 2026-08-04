import sys
import ctypes

# Windows API Constants
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10, version 2004 and above (transparent in capture)

def set_window_invisible_to_capture(hwnd_id: int, invisible: bool = True) -> bool:
    """
    Sets the window's display affinity on Windows so it is excluded from screen capture
    (like MS Teams, Zoom, Slack, Meet, screenshots, OBS, etc.) while remaining fully 
    visible and interactive to the user on their screen.
    
    Args:
        hwnd_id (int): The window handle (HWND) e.g. from widget.winId() in PySide6.
        invisible (bool): True to make it invisible to capture, False to make it visible.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if sys.platform != "win32":
        print("[win_utils] Non-windows platform. Window protection not supported.")
        return False
        
    try:
        user32 = ctypes.windll.user32
        affinity = WDA_EXCLUDEFROMCAPTURE if invisible else WDA_NONE
        
        if hasattr(user32, "SetWindowDisplayAffinity"):
            # Set display affinity
            result = user32.SetWindowDisplayAffinity(hwnd_id, affinity)
            if result:
                print(f"[win_utils] Successfully set display affinity to {hex(affinity)} for HWND {hwnd_id}")
                return True
            else:
                # Fallback to WDA_MONITOR (which shows a black/blank box instead of fully transparent on older Windows versions)
                fallback_affinity = WDA_MONITOR if invisible else WDA_NONE
                result = user32.SetWindowDisplayAffinity(hwnd_id, fallback_affinity)
                if result:
                    print(f"[win_utils] Fallback: set display affinity to {hex(fallback_affinity)} for HWND {hwnd_id}")
                    return True
                else:
                    error_code = ctypes.windll.kernel32.GetLastError()
                    print(f"[win_utils] Failed to set display affinity. HWND: {hwnd_id}, Error Code: {error_code}")
                    return False
        else:
            print("[win_utils] SetWindowDisplayAffinity API not found in user32.dll")
            return False
    except Exception as e:
        print(f"[win_utils] Exception while setting display affinity: {e}")
        return False
