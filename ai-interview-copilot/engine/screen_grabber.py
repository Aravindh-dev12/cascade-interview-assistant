import io
import mss
import numpy as np
from PIL import Image, ImageChops, ImageDraw

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

def images_are_near_duplicates(first: Image.Image, second: Image.Image) -> bool:
    """Detect repeated captures while tolerating cursors and small animations."""
    a = first.convert("RGB").resize((160, 90))
    b = second.convert("RGB").resize((160, 90))
    difference = np.asarray(ImageChops.difference(a, b), dtype=np.float32)
    return float(np.sqrt(np.mean(difference ** 2))) < 3.0

def combine_scroll_captures(images) -> Image.Image:
    """Stack consecutive scroll positions in reading order for one vision request."""
    if len(images) == 1:
        return images[0]

    prepared = []
    max_width = min(1800, max(image.width for image in images))
    for image in images:
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize(
                (max_width, int(image.height * ratio)), Image.Resampling.LANCZOS
            )
        prepared.append(image.convert("RGB"))

    header_height = 30
    total_height = sum(image.height + header_height for image in prepared)
    combined = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(combined)
    y = 0
    for index, image in enumerate(prepared, start=1):
        draw.rectangle((0, y, max_width, y + header_height), fill=(20, 24, 32))
        draw.text(
            (12, y + 7),
            f"SCROLL VIEW {index} OF {len(prepared)} - SAME QUESTION",
            fill="white",
        )
        y += header_height
        combined.paste(image, (0, y))
        y += image.height
    return combined
