# quntumnintent

A Windows desktop **practice assistant** for low-latency technical interview coaching. It combines real-time microphone/system-audio transcription, continuous screen-change capture, local or cloud multimodal reasoning, hot-plug device detection, and a compact streaming-answer overlay.

> Automatic coaching is intended only for mock interviews, practice sessions, or environments where AI assistance is explicitly permitted. `PRACTICE_MODE` is off by default.

## Highlights

- **Streaming answers:** answer chunks appear immediately instead of waiting for the full response.
- **Local-first AI:** `auto` mode prefers Ollama and falls back to Gemini only when the selected local model is unavailable.
- **Recommended local model:** `qwen3.5:4b` for the best coding + vision balance around the 4B class.
- **Fast speech endpointing:** NVIDIA Riva/Nemotron finalizes an utterance after configurable silence (default `0.40s`).
- **Bluetooth/USB microphone hot-plug:** a newly connected microphone/headset can become the active mic automatically, even while listening.
- **Camera hot-plug tracking:** USB, virtual, and supported Bluetooth camera topology changes are detected without opening the camera stream.
- **Continuous screen watcher:** frames are processed locally and emitted only after a meaningful visual change stabilizes.
- **Speech-first scheduling:** screen context never blocks a spoken-question answer.
- **Bounded queues/buffers:** audio and AI work queues cannot grow indefinitely.
- **No hover tooltips:** Qt tooltips are globally suppressed so scrollable settings/forms stay clean.
- **Optional Windows capture exclusion:** available but off by default.

## Why Qwen3.5 4B is the default local model

For this app, the important mix is **coding quality + image/screen understanding + low local latency**. Published 2026 benchmark tables are not perfectly comparable because different teams use different prompts/sampling settings, but the practical trade-off is clear:

| Model | Local footprint / class | Coding | Agent/tool use | Vision | Best use here |
|---|---|---|---|---|---|
| **Qwen3.5 4B** | ~3.4 GB Q4_K_M in Ollama | Strong | Strong | Yes | **Default**: coding + screenshot questions |
| LFM2.5 2.6B | ~1.7 GB Q4_K_M | Good, but behind the best small coding models | Excellent | No | Fast CPU / agentic fallback |
| Gemma 4 E4B | ~9.6 GB default Ollama package | Strong | Good | Yes | Good if you have more RAM/VRAM |

If your machine is resource constrained and raw speed matters more than coding quality, LFM2.5-2.6B is a useful alternative. If you have substantially more memory, benchmark Gemma 4 E4B on your own hardware.

## Target latency

For voice-only conceptual questions, the design target is roughly:

1. speaker stops
2. ~350–450 ms endpoint silence
3. final streaming ASR result
4. local question check
5. already-warmed local model request
6. first answer chunks rendered immediately

Treat ~1–2 seconds as a **time-to-first-answer target**, not a guarantee. First launch can be slower while Ollama loads model weights; the app warms the selected model in the background and keeps it alive for later questions.

## Architecture

```text
Mic / system audio
        |
        v
AudioRecorder (bounded queues)
        |
        v
NVIDIA Riva/Nemotron streaming ASR
        |
  partial + final transcript
        |
        +------> question detector
        |                 |
        |                 v
        |           speech request (highest priority)
        |                 |
        v                 v
recent transcript ---> local Ollama stream ---> overlay chunks
        ^                    |                       |
        |                    | fallback              |
ScreenWatcher ---------------+------> Gemini --------+

USB/Bluetooth mic ---------> AudioDeviceMonitor -> hot-swap active stream
Camera connect/disconnect -> CameraDeviceMonitor -> update selected camera
```

## Requirements

- Windows 10/11 recommended.
- Python 3.9+.
- NVIDIA API key for hosted Riva/Nemotron streaming ASR.
- **Ollama** for local reasoning (recommended), or a Gemini API key for cloud fallback.
- An input-capable Windows loopback device (or Stereo Mix) is needed to transcribe remote/system audio. Microphone-only mode also works.

## Install

```bash
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.template .env
```

### Install the recommended local model

Install Ollama for Windows, then:

```bash
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
```

You can close the interactive `ollama run` session afterward; keep the Ollama service running. The desktop app talks to `http://127.0.0.1:11434` and warms the model automatically.

For a permitted practice session:

```env
PRACTICE_MODE=1
AI_PROVIDER=auto
OLLAMA_MODEL=qwen3.5:4b
```

`AI_PROVIDER` accepts:

- `auto` — local Ollama first, then Gemini fallback.
- `ollama` — local-only.
- `gemini` — cloud-only.

## Recommended low-latency config

```env
ASR_ENDPOINT_SECONDS=0.40
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=30

AI_PROVIDER=auto
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_NUM_CTX=8192
OLLAMA_KEEP_ALIVE=30m
OLLAMA_TEMPERATURE=0.15
```

Avoid endpoint silence much below ~0.30s because natural pauses inside a sentence can otherwise be split into separate utterances.

## Automatic microphone and camera detection

The app continuously watches audio topology while running. When a new USB/Bluetooth/headset microphone appears and **Switch automatically to a newly connected microphone/headset** is enabled, it hot-swaps to that microphone and restarts the active stream automatically.

Camera topology is tracked through Qt Multimedia without opening the camera. Newly connected cameras can become the selected camera automatically. The current visual-answer path still uses the screen watcher; camera detection is maintained as device state for camera-based capture workflows.

## Real-time screen behavior

The screen watcher does **not** upload frames continuously. It computes a tiny local grayscale fingerprint and waits for a visual change to stabilize before producing a JPEG context frame.

While live listening is active:

- screen changes update the latest visual context;
- they do not occupy the speech AI worker;
- a spoken question referring to visible code/error/question can attach the recent frame.

Qwen3.5 4B supports image input in Ollama, so screen-based coding questions can remain fully local.

## Settings

The Settings window now includes:

- AI provider: Auto / local Ollama / Gemini.
- editable local model name and Ollama URL.
- local context size.
- Gemini fallback model.
- microphone/system-audio routing.
- automatic audio hot-plug and auto-switch behavior.
- camera device selection and camera hot-plug behavior.
- continuous screen watcher controls.
- screen capture interval and stability delay.
- overlay opacity/font/always-on-top.

Hover tooltips are disabled globally.

## Benchmark local models on your machine

Published benchmarks are useful, but local latency depends heavily on your CPU/GPU and quantization. A small benchmark runner is included:

```bash
python tools/benchmark_local_models.py qwen3.5:4b
```

Compare several installed models:

```bash
python tools/benchmark_local_models.py qwen3.5:4b gemma4:e4b oamazonasgabriel/lfm2.5-2.6b:q4_k_m-8gbGPU
```

It reports:

- time to first token (TTFT),
- total generation time,
- pass/fail for a few executable Python coding tasks.

This is not a replacement for LiveCodeBench; it is meant to answer which model is actually best for **this app on your machine**.

## Controls

- `Ctrl+Shift+A` — toggle live listening.
- `Ctrl+Shift+S` — manual screen capture and answer.
- **Listen** — start/stop streaming transcription.
- **Capture screen** — capture the configured region or the monitor containing the overlay.
- **Clear** — clear transcript/image context and invalidate stale AI results.

## Security

Credentials load from a project-local env file and are never stored in the application settings JSON. Supported local env filenames are ignored by Git. Never commit real credentials; rotate any credential that has ever appeared in Git history.

## Run

```bash
python main.py
```

## Tests

```bash
python -m unittest tests.test_question_detector
```
