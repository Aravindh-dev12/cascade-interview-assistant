import io

import mss
from PIL import Image


def get_monitors_info():
    with mss.mss() as sct:
        return sct.monitors


def monitor_for_point(x: int, y: int) -> dict:
    """Return the physical monitor containing a global screen point."""
    with mss.mss() as sct:
        physical = sct.monitors[1:] if len(sct.monitors) > 1 else sct.monitors
        for monitor in physical:
            left = int(monitor["left"])
            top = int(monitor["top"])
            right = left + int(monitor["width"])
            bottom = top + int(monitor["height"])
            if left <= x < right and top <= y < bottom:
                return dict(monitor)
        return dict(physical[0])


def capture_screen(region: dict = None, point: tuple[int, int] | None = None) -> Image.Image:
    """Capture a custom region or the monitor containing ``point``.

    When no region/point is supplied, capture the primary monitor. ``point`` is
    useful for the overlay: pass the overlay center so capture follows whichever
    monitor the user is currently working on.
    """
    with mss.mss() as sct:
        if region is not None:
            monitor = {
                "top": int(region.get("top", 0)),
                "left": int(region.get("left", 0)),
                "width": max(1, int(region.get("width", 800))),
                "height": max(1, int(region.get("height", 600))),
            }
        elif point is not None:
            x, y = point
            physical = sct.monitors[1:] if len(sct.monitors) > 1 else sct.monitors
            monitor = None
            for candidate in physical:
                left = int(candidate["left"])
                top = int(candidate["top"])
                if (
                    left <= x < left + int(candidate["width"])
                    and top <= y < top + int(candidate["height"])
                ):
                    monitor = candidate
                    break
            monitor = monitor or physical[0]
        else:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def get_image_bytes(image: Image.Image, format: str = "PNG", quality: int = 80) -> bytes:
    img_byte_arr = io.BytesIO()
    if format.upper() == "JPEG":
        image.save(img_byte_arr, format=format, quality=quality, optimize=True)
    else:
        image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()
