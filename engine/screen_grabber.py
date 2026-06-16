import io
import mss
from PIL import Image

def get_monitors_info():
    """
    Returns a list of connected monitors and their geometry.
    """
    with mss.mss() as sct:
        return sct.monitors

def capture_screen(region: dict = None) -> Image.Image:
    """
    Captures the screen or a custom region using mss.
    
    Args:
        region (dict, optional): Dict containing dimensions:
            {"top": int, "left": int, "width": int, "height": int}.
            If None, captures the primary monitor.
            
    Returns:
        PIL.Image: Captured screen image.
    """
    with mss.mss() as sct:
        # If no region, use the primary monitor (index 1 is primary monitor)
        if region is None:
            if len(sct.monitors) > 1:
                monitor = sct.monitors[1]
            else:
                monitor = sct.monitors[0]
        else:
            # Ensure correct format for mss
            monitor = {
                "top": int(region.get("top", 0)),
                "left": int(region.get("left", 0)),
                "width": int(region.get("width", 800)),
                "height": int(region.get("height", 600)),
            }
            
        # Capture the screen
        sct_img = sct.grab(monitor)
        
        # Convert mss object to PIL Image
        # mss raw data is BGRA
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        return img

def get_image_bytes(image: Image.Image, format: str = "PNG", quality: int = 80) -> bytes:
    """
    Converts a PIL Image to raw bytes in memory.
    
    Args:
        image (PIL.Image): Image to convert.
        format (str): Image format (e.g., 'PNG' or 'JPEG').
        quality (int): Compression quality (only relevant for JPEG).
        
    Returns:
        bytes: Compressed image bytes.
    """
    img_byte_arr = io.BytesIO()
    if format.upper() == "JPEG":
        image.save(img_byte_arr, format=format, quality=quality)
    else:
        image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()
