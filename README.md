# quntumnintent

A Windows desktop **practice assistant** for low-latency technical interview coaching. It combines microphone + system-audio transcription, screen context, live camera frames, NVIDIA Kimi-K3 reasoning/vision, local Qwen fallback, and a streaming answer overlay.

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
                      ├─> latest visual context ├─> NVIDIA Kimi-K3
Live camera frames ───┘                         |       |
                                                |       ├─ streamed answer
                                                |       |
                                                |       └─ local Qwen hedge/fallback
                                                v
                                             overlay
```

## AI backends

There are only three runtime modes:

- `AI_PROVIDER=hybrid` — NVIDIA Kimi-K3 gets a short head start; local Qwen starts if cloud first-token latency is high. First answer stream wins.
- `AI_PROVIDER=nvidia` — NVIDIA Kimi-K3 only.
- `AI_PROVIDER=ollama` — local Qwen/Ollama only.

`NVIDIA_API_KEY` is read automatically from the project `.env` file. There is no API-key field in Settings and credentials are not written to the settings JSON file.

The same NVIDIA key is used for:

- Kimi-K3 text/image inference through NVIDIA API Catalog;
- NVIDIA Riva/Nemotron streaming speech recognition.

## Implemented

- **NVIDIA Kimi-K3:** streamed text + image reasoning for chat, screenshots, and camera frames.
- **Local Qwen:** `qwen3.5:4b` through Ollama with streamed text/vision output.
- **Hybrid first-token race:** Kimi starts first; Qwen hedges after `HYBRID_HEDGE_SECONDS` when needed.
- **Bluetooth/USB microphone hot-plug:** new microphones/headsets can become active while the app is running.
- **Windows system audio:** `Default speaker loopback` captures current Windows output using WASAPI/SoundCard.
- **Camera hot-plug + live frames:** Qt `QVideoSink` samples the selected/default camera and keeps only the latest compressed frame locally.
- **Camera vision:** press **Camera**, or ask a visual question such as “what am I holding?” to attach recent camera context.
- **Screen + camera context:** when both are available, they are combined into a labeled vision image.
- **Speech-first scheduling:** ordinary voice questions stay text-only unless visual context is relevant.

## Install

```powershell
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
git checkout main

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.template .env
```

Install Ollama for Windows and pull the local fallback model:

```powershell
ollama pull qwen3.5:4b
```

## Recommended `.env`

```env
PRACTICE_MODE=1

AI_PROVIDER=hybrid

NVIDIA_API_KEY=YOUR_NVIDIA_KEY
NVIDIA_KIMI_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_KIMI_MODEL=moonshotai/kimi-k3
NVIDIA_KIMI_REASONING_EFFORT=low
NVIDIA_KIMI_TEMPERATURE=1.0
NVIDIA_KIMI_MAX_TOKENS=1200
NVIDIA_KIMI_TIMEOUT_SECONDS=8.0
HYBRID_HEDGE_SECONDS=1.25

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_NUM_CTX=8192
OLLAMA_KEEP_ALIVE=30m

NVIDIA_RIVA_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_RIVA_FUNCTION_ID=bb0837de-8c7b-481f-9ec8-ef5663e9c1fa
NVIDIA_RIVA_LANGUAGE=en-US
ASR_ENDPOINT_SECONDS=0.35
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=30

WASAPI_LOOPBACK_SAMPLE_RATE=48000
```

## Audio behavior

In Settings, choose a microphone and use **Default speaker loopback** for system audio when available. Microphone audio is transcribed as `Candidate`; default-speaker/browser/meeting audio is transcribed as `Interviewer`.

Windows/PortAudio numeric device indices can change after reboot, Bluetooth reconnect, docking, or driver changes. The app validates saved indices before listening and attempts to recover to the current microphone/default-speaker loopback automatically.

## Camera behavior

Choose the camera in Settings. In practice mode, live camera capture starts automatically using the selected camera or Windows default camera. Frames are sampled at about 450 ms by default, compressed, and kept only as the latest local context.

Frames are not continuously submitted to an AI provider. They are attached when:

- you press **Camera**;
- a spoken/typed prompt refers to the camera, an object being shown, or asks “what do you see?”;
- the request otherwise needs the combined visual context.

## Controls

- `Ctrl+Shift+A` — start/stop microphone + system-audio listening.
- `Ctrl+Shift+S` — capture screen and answer.
- **Camera** — analyze the latest live camera frame.
- **Clear** — clear transcript and answer context.

## Run

```powershell
python main.py
```

Expected startup includes:

```text
[env] NVIDIA_API_KEY loaded: True
[env] PRACTICE_MODE enabled: True
[env] AI_PROVIDER: hybrid
AI: NVIDIA Kimi-K3 + local Qwen/Ollama only.
NVIDIA_API_KEY is loaded only from the project .env file.
```

## Tests

```powershell
python -m compileall -q .
python -m unittest -q tests.test_question_detector
```

For Windows hardware verification, test typed chat, microphone transcription, YouTube/system-audio transcription, the **Camera** button, and a spoken camera prompt separately.
