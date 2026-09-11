"""Small on-device coding + latency benchmark for Ollama models.

Examples:
    python tools/benchmark_local_models.py qwen3.5:4b
    python tools/benchmark_local_models.py qwen3.5:4b gemma4:e4b

This is intentionally small. It is not a replacement for LiveCodeBench; it answers the
more practical question: which installed model is fastest and reliably solves a few
basic coding tasks on this machine?
"""

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"

TASKS = [
    {
        "name": "two_sum",
        "prompt": (
            "Write only one Python code block defining function two_sum(nums, target) that returns the indices "
            "of two distinct elements whose sum equals target. O(n) expected."
        ),
        "tests": """
assert two_sum([2,7,11,15], 9) in ([0,1],[1,0])
r = two_sum([3,2,4], 6); assert set(r) == {1,2}
r = two_sum([3,3], 6); assert set(r) == {0,1}
""",
    },
    {
        "name": "valid_parentheses",
        "prompt": (
            "Write only one Python code block defining function is_valid(s) that returns whether (), [], and {} "
            "are correctly balanced and nested."
        ),
        "tests": """
assert is_valid('()[]{}') is True
assert is_valid('(]') is False
assert is_valid('([{}])') is True
assert is_valid('(((') is False
""",
    },
    {
        "name": "binary_search",
        "prompt": (
            "Write only one Python code block defining function binary_search(nums, target) returning the index "
            "of target in ascending nums, or -1. Use O(log n) time."
        ),
        "tests": """
assert binary_search([-1,0,3,5,9,12], 9) == 4
assert binary_search([-1,0,3,5,9,12], 2) == -1
assert binary_search([5], 5) == 0
assert binary_search([], 1) == -1
""",
    },
]


def stream_chat(model, prompt):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise expert coding assistant. Follow the output format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 8192,
            "num_predict": 700,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    pieces = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            item = json.loads(raw_line.decode("utf-8"))
            content = (item.get("message") or {}).get("content") or ""
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(content)
            if item.get("done"):
                break
    ended = time.perf_counter()
    return "".join(pieces), (first_token_at or ended) - started, ended - started


def extract_python(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def run_tests(code, tests):
    source = code + "\n\n" + tests + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory(prefix="local-llm-bench-") as temp_dir:
        path = Path(temp_dir) / "solution.py"
        path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=temp_dir,
                text=True,
                capture_output=True,
                timeout=4,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return completed.returncode == 0 and "PASS" in completed.stdout, (
            completed.stderr.strip() or completed.stdout.strip()
        )


def benchmark_model(model):
    rows = []
    for task in TASKS:
        try:
            answer, ttft, total = stream_chat(model, task["prompt"])
            code = extract_python(answer)
            passed, detail = run_tests(code, task["tests"])
            rows.append((task["name"], passed, ttft, total, detail[:160]))
        except Exception as exc:
            rows.append((task["name"], False, float("nan"), float("nan"), str(exc)[:160]))
    return rows


def main():
    models = sys.argv[1:] or ["qwen3.5:4b"]
    print("Local coding benchmark (Ollama)")
    print(f"Endpoint: {OLLAMA_URL}\n")
    for model in models:
        print(f"=== {model} ===")
        rows = benchmark_model(model)
        passed = sum(1 for _, ok, *_ in rows if ok)
        ttfts = [ttft for _, _, ttft, _, _ in rows if ttft == ttft]
        totals = [total for _, _, _, total, _ in rows if total == total]
        for name, ok, ttft, total, detail in rows:
            ttft_text = "n/a" if ttft != ttft else f"{ttft:.2f}s"
            total_text = "n/a" if total != total else f"{total:.2f}s"
            print(f"  {name:20} {'PASS' if ok else 'FAIL':4}  TTFT={ttft_text:>7}  total={total_text:>7}")
            if not ok and detail:
                print(f"    {detail}")
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
        avg_total = sum(totals) / len(totals) if totals else float("nan")
        print(f"  score: {passed}/{len(rows)} | avg TTFT: {avg_ttft:.2f}s | avg total: {avg_total:.2f}s\n")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama is not reachable at {OLLAMA_URL}: {exc}")
