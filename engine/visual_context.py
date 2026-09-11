import io

from PIL import Image, ImageDraw


def _open_rgb(payload):
    return Image.open(io.BytesIO(payload)).convert("RGB")


def combine_visual_context(items, max_panel_width=1100, quality=82):
    """Combine labeled JPEG/PNG payloads into one compact multimodal context image.

    ``items`` is an iterable of ``(label, bytes)`` pairs. If only one usable image
    exists, its original bytes are returned to avoid unnecessary re-encoding.
    """
    valid = [(str(label or "IMAGE"), payload) for label, payload in items if payload]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][1]

    panels = []
    for label, payload in valid:
        image = _open_rgb(payload)
        if image.width > max_panel_width:
            ratio = max_panel_width / image.width
            image = image.resize(
                (max_panel_width, max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        panels.append((label.upper(), image))

    label_height = 30
    gap = 8
    width = max(image.width for _, image in panels)
    height = sum(image.height + label_height for _, image in panels) + gap * (
        len(panels) - 1
    )

    canvas = Image.new("RGB", (width, height), (12, 17, 26))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for index, (label, image) in enumerate(panels):
        draw.rectangle((0, y, width, y + label_height), fill=(23, 37, 58))
        draw.text((10, y + 8), label, fill=(226, 232, 240))
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
        if index < len(panels) - 1:
            y += gap

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=int(quality), optimize=True)
    return output.getvalue()
