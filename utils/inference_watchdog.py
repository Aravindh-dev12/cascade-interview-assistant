import os
import queue
import threading
import time


def _float_env(name, default, minimum, maximum):
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def install_inference_watchdog():
    """Prevent a stalled Kimi/Qwen request from blocking the desktop answer queue.

    The original hybrid router waited forever if a provider connected but never
    emitted its first token. Visual requests can also arrive immediately at
    startup while an Ollama warmup is still loading the model. This patch makes
    warmup opt-in, clamps stalled Ollama sockets, and gives every hybrid request
    a bounded first-token deadline.
    """
    from engine import copilot_ai as ai_module

    CopilotAI = ai_module.CopilotAI
    OllamaClient = ai_module._OllamaClient
    if getattr(CopilotAI, "_inference_watchdog_installed", False):
        return

    original_warmup = OllamaClient.warmup
    original_request = OllamaClient._request

    def warmup(client):
        enabled = os.environ.get("OLLAMA_WARMUP", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not enabled:
            print("[ollama] Startup warmup skipped; first real request will load the model.")
            return False
        return original_warmup(client)

    def bounded_request(client, path, payload=None, timeout=2.0):
        if path == "/api/chat" and payload and payload.get("stream"):
            socket_timeout = _float_env(
                "OLLAMA_STREAM_SOCKET_TIMEOUT_SECONDS", 35.0, 8.0, 120.0
            )
            timeout = min(float(timeout), socket_timeout)
        return original_request(client, path, payload=payload, timeout=timeout)

    OllamaClient.warmup = warmup
    OllamaClient._request = bounded_request

    def run_provider(name, factory, events):
        try:
            produced = False
            for piece in factory():
                if not piece:
                    continue
                produced = True
                events.put((name, "chunk", piece))
            events.put((name, "done", None if produced else "no answer tokens"))
        except Exception as exc:
            events.put((name, "error", exc))

    def hybrid_stream(ai, prompt, max_tokens, image_bytes_list=None):
        visual = bool(image_bytes_list)
        if not ai.kimi.available():
            yield from ai._qwen_stream(prompt, max_tokens, image_bytes_list)
            return

        started = time.monotonic()
        events = queue.Queue()
        errors = {}
        done = set()
        winner = None
        qwen_started = False

        if visual:
            hedge_seconds = _float_env("HYBRID_VISION_HEDGE_SECONDS", 0.15, 0.0, 4.0)
            first_token_timeout = _float_env(
                "VISION_FIRST_TOKEN_TIMEOUT_SECONDS", 25.0, 4.0, 90.0
            )
        else:
            hedge_seconds = _float_env("HYBRID_HEDGE_SECONDS", 1.25, 0.0, 4.0)
            first_token_timeout = _float_env(
                "TEXT_FIRST_TOKEN_TIMEOUT_SECONDS", 15.0, 3.0, 60.0
            )
        stream_idle_timeout = _float_env(
            "PROVIDER_STREAM_IDLE_TIMEOUT_SECONDS", 30.0, 5.0, 120.0
        )

        total_image_bytes = sum(len(item) for item in (image_bytes_list or []) if item)
        print(
            f"[hybrid] Request started · visual={visual} · images={len(image_bytes_list or [])} · "
            f"image_bytes={total_image_bytes} · Kimi head-start={hedge_seconds:.2f}s · "
            f"first-token deadline={first_token_timeout:.1f}s"
        )

        threading.Thread(
            target=run_provider,
            args=("kimi", lambda: ai._kimi_stream(prompt, max_tokens, image_bytes_list), events),
            daemon=True,
            name="hybrid-kimi",
        ).start()

        deadline = started + first_token_timeout
        while winner is None:
            elapsed = time.monotonic() - started
            if not qwen_started and elapsed >= hedge_seconds:
                qwen_started = True
                print("[hybrid] Starting local Qwen hedge.")
                threading.Thread(
                    target=run_provider,
                    args=("qwen", lambda: ai._qwen_stream(prompt, max_tokens, image_bytes_list), events),
                    daemon=True,
                    name="hybrid-qwen",
                ).start()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending = []
                if "kimi" not in errors and "kimi" not in done:
                    pending.append("Kimi: no first token")
                if qwen_started and "qwen" not in errors and "qwen" not in done:
                    pending.append("Qwen: no first token")
                details = "; ".join(
                    [f"{name}: {err}" for name, err in errors.items()] + pending
                ) or "no provider produced output"
                raise RuntimeError(
                    f"Hybrid first-token timeout after {first_token_timeout:.1f}s: {details}"
                )

            try:
                name, event_type, payload = events.get(timeout=min(0.10, remaining))
            except queue.Empty:
                continue

            if event_type == "chunk":
                winner = name
                ai._active_provider = "nvidia" if name == "kimi" else "ollama"
                print(
                    f"[hybrid] {name.upper()} won first token in "
                    f"{time.monotonic() - started:.2f}s"
                )
                yield payload
                break
            if event_type == "error":
                errors[name] = payload
                print(f"[hybrid] {name} failed before first token: {payload}")
                if name == "kimi" and not qwen_started:
                    qwen_started = True
                    print("[hybrid] Kimi failed; starting local Qwen immediately.")
                    threading.Thread(
                        target=run_provider,
                        args=("qwen", lambda: ai._qwen_stream(prompt, max_tokens, image_bytes_list), events),
                        daemon=True,
                        name="hybrid-qwen",
                    ).start()
            elif event_type == "done":
                done.add(name)
                if payload:
                    errors[name] = RuntimeError(str(payload))

            expected = {"kimi", "qwen"} if qwen_started else {"kimi"}
            finished = set(errors) | done
            if expected.issubset(finished) and winner is None:
                details = "; ".join(
                    f"{name}: {errors.get(name, 'no answer tokens')}" for name in sorted(expected)
                )
                raise RuntimeError(f"Hybrid inference failed: {details}")

        while True:
            try:
                name, event_type, payload = events.get(timeout=stream_idle_timeout)
            except queue.Empty:
                print(
                    f"[hybrid] {winner} stream idle for {stream_idle_timeout:.1f}s; "
                    "ending the request so later questions are not blocked."
                )
                return
            if name != winner:
                continue
            if event_type == "chunk":
                yield payload
            elif event_type == "done":
                return
            elif event_type == "error":
                print(f"[hybrid] {winner} stream ended after partial output: {payload}")
                return

    CopilotAI._run_provider = staticmethod(run_provider)
    CopilotAI._hybrid_stream = hybrid_stream
    CopilotAI._inference_watchdog_installed = True
