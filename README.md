# quntumnintent

A Windows desktop **practice assistant** for low-latency technical interview coaching. It combines microphone + system-audio transcription, screen context, live camera frames, local Qwen vision/reasoning, and a streaming answer overlay.

> Automatic coaching is intended only for mock interviews, practice sessions, or environments where AI assistance is explicitly permitted. `PRACTICE_MODE` is off by default.

## Real-time pipeline

```text
Bluetooth / USB / built-in microphone ─┐
                                       ├─> NVIDIA Riva/Nemotron streaming ASR
Browser / YouTube / Teams / Meet audio ┘       |
        ^                                      | final interviewer question
        |                                      v
Windows default-speaker loopback           question detector
                                                |
Screen watcher ───────┐                         |
                      ├─> latest visual context ├─> Qwen3.5 4B (Ollama)
Live camera frames ───┘                         |      |
                                                |      └─ streaming tokens
                                                v
                                             overlay
```

## Implemented

- **Local Qwen:** `qwen3.5:4b` through Ollama with streamed answer chunks.
- **Gemini fallback:** optional when a Gemini key is configured.
- **Local-only mode:** `AI_PROVIDER=ollama` works without a Gemini key.
- **Bluetooth/USB microphone hot-plug:** new microphones/headsets can become active while the app is running.
- **Windows system audio:** the preferred `Default speaker loopback` source captures the current Windows output using WASAPI/SoundCard, so browser videos and supported meeting apps do not require Stereo Mix or a virtual cable.
- **Camera hot-plug + live frames:** Qt `QVideoSink` samples the selected/default camera. Only the most recent JPEG is kept.
- **Camera → Qwen:** press **Camera** to analyze the current frame, or ask a camera-aware spoken/typed question such as “what am I holding?” and the latest camera context is attached automatically.
- **Screen + camera context:** when both are available, they are combined into a labeled vision image so Qwen can distinguish them.
- **Speech-first scheduling:** visual context does not continuously occupy Qwen; ordinary voice questions stay text-only and high priority.
- **No hover tooltips:** Qt tooltips are globally suppressed.

## Latency target

For a warm local Qwen model, the target path is:

1. speaker stops;
2. about 350 ms endpoint silence;
3. NVIDIA returns the final transcript;
4. local question detection runs immediately;
5. Qwen starts streaming;
6. first answer tokens appear in the overlay.

The engineering target is roughly **1–2 seconds to first answer tokens** for voice-only questions. It is not a hard guarantee: ASR network latency, CPU/GPU speed, model load state, and image processing can increase it. Camera/screen images are attached only when relevant so normal voice questions stay on the fastest path.

## Install

```powershell
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
git checkout main

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.template .env
```

Install Ollama for Windows and pull the local model:

```powershell
ollama pull qwen3.5:4b
```

## Recommended `.env`

```env
PRACTICE_MODE=1

AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_NUM_CTX=8192
OLLAMA_KEEP_ALIVE=30m

NVIDIA_API_KEY=YOUR_NVIDIA_KEY
ASR_ENDPOINT_SECONDS=0.35
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=30

WASAPI_LOOPBACK_SAMPLE_RATE=48000
```

Gemini is optional in `ollama` mode. To allow cloud fallback, use `AI_PROVIDER=auto` and add a valid `GEMINI_API_KEY`.

## Audio behavior

In Settings, choose a microphone and use **Default speaker loopback** for system audio when available. Microphone audio is transcribed as `Candidate`; default-speaker/browser/meeting audio is transcribed as `Interviewer`.

The default-speaker source is designed for audio played through the current Windows output, including browser/video players and common meeting applications. When Windows audio topology changes, the monitor refreshes the active streams; newly connected microphones can also be selected automatically.

## Camera behavior

Choose the camera in Settings. In practice mode, live camera capture starts automatically using the selected camera or Windows default camera. Frames are sampled at about 450 ms by default, compressed, and kept only as the latest local context.

Frames are **not** continuously submitted to Qwen. They are used when:

- you press **Camera**;
- a spoken/typed prompt refers to the camera, an object being shown, or asks “what do you see?”;
- the request otherwise needs the combined visual context.

This keeps continuous camera capture from competing with voice-answer latency.

## Controls

- `Ctrl+Shift+A` — start/stop microphone + system-audio listening.
- `Ctrl+Shift+S` — capture screen and answer.
- **Camera** — analyze the latest live camera frame.
- **Capture screen** — analyze the current screen/selected region.
- **Clear** — clear transcript and answer context.

## Run

```powershell
python main.py
```

## Tests

```powershell
python -m compileall -q .
python -m unittest -q tests.test_question_detector
```

For Windows hardware verification, test microphone transcription, YouTube/system-audio transcription, Bluetooth hot-swap, the **Camera** button, and a spoken camera prompt such as “what am I holding in front of the camera?” separately.
