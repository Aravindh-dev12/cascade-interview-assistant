import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from engine.nvidia_kimi import NvidiaKimiClient


def make_test_image():
    image = Image.new("RGB", (480, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 90), "VISION TEST 42", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def collect(stream, limit=4000):
    pieces = []
    total = 0
    for piece in stream:
        print(piece, end="", flush=True)
        pieces.append(piece)
        total += len(piece)
        if total >= limit:
            break
    print()
    return "".join(pieces)


def main():
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key or key.startswith("nvapi-REPLACE"):
        raise SystemExit(
            "NVIDIA_API_KEY is missing. Rotate the exposed key, place the new key in .env, then rerun this test."
        )

    client = NvidiaKimiClient(api_key=key)
    print(f"Model: {client.model}")
    print("Key loaded: yes (value intentionally hidden)")

    print("\n[1/2] Text streaming test")
    text = collect(
        client.chat_stream(
            "Reply with exactly: READY",
            max_tokens=32,
            system_prompt="Follow the user's formatting instruction exactly.",
        )
    )
    if "READY" not in text.upper():
        print("WARNING: text endpoint responded, but did not contain READY.")

    print("\n[2/2] Vision streaming test")
    vision = collect(
        client.chat_stream(
            "Read the large text in the supplied image. What number is visible? Answer with only the number.",
            max_tokens=32,
            image_bytes_list=[make_test_image()],
            system_prompt="Read the image carefully and answer concisely.",
        )
    )
    if "42" not in vision:
        print("WARNING: vision endpoint responded, but did not identify 42.")

    print("\nKimi K3 smoke test finished.")


if __name__ == "__main__":
    main()
