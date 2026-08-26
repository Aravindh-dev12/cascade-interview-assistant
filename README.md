# quntumnintent

A Windows desktop practice assistant with a lightweight overlay, real-time microphone/system-audio transcription, and screen-aware technical Q&A.

## AI stack

This project now uses only two AI services:

- **Google Gemini** for all text and vision reasoning. Default model: `gemini-2.5-flash`.
- **NVIDIA Nemotron streaming ASR** for real-time speech-to-text. The hosted Riva endpoint uses NVIDIA's `nemotron-asr-streaming` NIM and supports interim streaming transcripts.

There is no OpenAI provider or OpenAI SDK path in the application.

## Features

- Real-time microphone capture for the candidate.
- Windows system/loopback audio capture for the interviewer when a compatible input is available.
- Low-latency NVIDIA streaming ASR with interim and final transcripts.
- Gemini text answers from transcript context.
- Gemini vision analysis for a selected screen region.
- Global shortcuts:
  - `Ctrl+Shift+A` toggles listening.
  - `Ctrl+Shift+S` captures/analyzes the configured screen region.
- Configurable always-on-top/translucent overlay.
- Automatic coaching is disabled by default; enable `PRACTICE_MODE=1` only for mock interviews or environments where AI assistance is permitted.

## Requirements

- Windows 10/11 is recommended for system-audio loopback and protected overlay features.
- Python 3.9+.
- A Gemini API key.
- An NVIDIA API key for the hosted Nemotron/Riva speech endpoint.

## Install

```bash
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

Copy the environment template:

```bash
copy .env.template .env
```

Then set your own keys in `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_RIVA_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_RIVA_FUNCTION_ID=bb0837de-8c7b-481f-9ec8-ef5663e9c1fa
NVIDIA_RIVA_LANGUAGE=en-US
```

The NVIDIA function ID above is the hosted `nemotron-asr-streaming` endpoint documented by NVIDIA. You can also paste the Gemini and NVIDIA keys in the Settings dialog instead of storing them in `.env`.

Never commit a real `.env` file or real credentials. If a key has ever been committed to Git history, revoke/rotate it even after replacing the current file with placeholders.

## Real-time voice flow

1. `AudioRecorder` captures 16 kHz microphone audio and, when available, Windows loopback/system audio.
2. `STTWorker` opens an NVIDIA Riva gRPC streaming session for each active speaker utterance.
3. Interim hypotheses are emitted immediately to the overlay as a live line.
4. Final transcripts are added to the conversation context.
5. In practice mode, substantive interviewer transcripts can trigger a Gemini answer.

Latency can be tuned with:

```env
ASR_ENDPOINT_SECONDS=0.50
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=20
```

## Run

```bash
python main.py
```

Open Settings and choose your microphone and system/loopback device. The UI shows separate fields for the Gemini API key and NVIDIA API key.

## Security note

`.env.template` must contain placeholders only. Keep `.env` ignored by Git and rotate any credential that was previously committed.
