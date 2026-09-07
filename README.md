# quntumnintent

A Windows desktop **practice assistant** for low-latency technical interview coaching. It combines real-time microphone/system-audio transcription, continuous screen-change capture, Gemini text/vision reasoning, and a compact streaming-answer overlay.

> Automatic coaching is intended only for mock interviews, practice sessions, or environments where AI assistance is explicitly permitted. `PRACTICE_MODE` is off by default.

## What changed in the real-time build

- **Streaming answers:** Gemini chunks are rendered as they arrive instead of waiting for the full response.
- **Fast speech endpointing:** NVIDIA Riva/Nemotron finalizes an utterance after configurable silence (default `0.40s`).
- **Question detector:** acknowledgements such as “okay” or “great” do not trigger an AI request; substantive interview prompts do.
- **Continuous screen watcher:** captures locally every ~650 ms, but sends a frame only after a meaningful visual change stabilizes.
- **Speech-first scheduling:** screen watching updates context while live listening is active and does not block the speech answer path.
- **Screen-aware speech:** questions such as “fix this code” or “solve this question” can automatically attach the latest recent screen frame.
- **Bounded queues/buffers:** microphone/system audio and AI request queues no longer grow without bound.
- **Stale request handling:** a new spoken question supersedes stale background screen work; Clear invalidates in-flight results.
- **Safer capture behavior:** overlay capture exclusion is available but is **off by default**.

## Target latency

For voice-only conceptual questions, the design target is roughly:

1. speaker stops
2. ~350–450 ms endpoint silence
3. final streaming ASR result
4. local question check
5. Gemini request
6. first answer chunks rendered immediately

Actual latency depends on the network and provider load. Treat ~1–2 seconds as a time-to-first-answer target, not a hard guarantee. Vision/coding requests can take longer.

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
        +------> local question detector
        |                 |
        |                 v
        |           speech request (highest priority)
        |                 |
        v                 v
recent transcript ----> Gemini stream ----> overlay chunks
        ^                    ^
        |                    |
ScreenWatcher ---- latest stable screen context
        |
        +---- auto screen-only answer when live listening is not active
```

## Requirements

- Windows 10/11 recommended for audio loopback and overlay features.
- Python 3.9+.
- Gemini API key.
- NVIDIA API key for the hosted Riva/Nemotron streaming ASR endpoint.
- An input-capable Windows loopback device (or Stereo Mix) is needed to transcribe remote/system audio. The microphone can still be used alone.

## Install

```bash
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template:

```bash
copy .env.template .env
```

Then add your credentials and, for a permitted practice session, explicitly enable:

```env
PRACTICE_MODE=1
```

## Recommended low-latency config

```env
ASR_ENDPOINT_SECONDS=0.40
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=30
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_BUDGET=0
GEMINI_TEXT_MAX_TOKENS=700
GEMINI_VISION_MAX_TOKENS=1600
```

Avoid setting the endpoint much below ~0.30s because ordinary pauses inside a sentence can otherwise be mistaken for the end of the question.

## Real-time screen behavior

The screen watcher does **not** upload frames continuously. It computes a tiny grayscale fingerprint locally and waits for a visual change to stabilize before producing a JPEG context frame.

While live listening is active:

- screen changes update the latest visual context;
- they do not automatically occupy the AI worker;
- a spoken question referring to visible code/error/question can attach the recent frame.

When listening is off, stable screen changes can automatically trigger a vision answer if **Auto-answer screen** is enabled in Settings. Manual `Capture screen` works at any time.

## Controls

- `Ctrl+Shift+A` — toggle live listening.
- `Ctrl+Shift+S` — manual screen capture and answer.
- **Listen** — start/stop streaming transcription.
- **Capture screen** — capture the configured region, or the monitor containing the overlay.
- **Clear** — clear transcript/image context and invalidate stale AI results.

## Settings

The Settings window includes:

- Gemini model.
- microphone and system-audio routing.
- automatic listening / speech answer behavior.
- continuous screen watcher and screen-only auto-answer toggles.
- screen capture interval and stability delay.
- optional selected capture region.
- overlay opacity/font/always-on-top.
- optional Windows capture exclusion (off by default).

## Security

Credentials are loaded from a project-local env file and never stored in the application settings JSON. Supported local env filenames are ignored by Git. Never commit real API credentials; rotate any credential that has ever appeared in Git history.

## Run

```bash
python main.py
```

## Tests

The local detector has a small standard-library test suite:

```bash
python -m unittest tests.test_question_detector
```
