# quntumnintent

A Windows desktop **practice assistant** for low-latency technical interview coaching. It uses NVIDIA Parakeet for microphone/system-audio speech-to-text, NVIDIA Nemotron Omni for first-pass visual extraction, and Google Gemini 2.5 Flash for every final answer.

> Use this assistant only for mock interviews, practice sessions, or environments where AI assistance is explicitly permitted. `PRACTICE_MODE` is off by default.

## Runtime pipeline

```text
Microphone + Windows system audio
        |
        v
NVIDIA Parakeet CTC English streaming ASR
        |
        v
Transcript / detected question
        |
        v
Google Gemini 2.5 Flash
        |
        v
Streaming overlay answer
```

Manual visual capture:

```text
Capture screen / Camera
        |
        v
NVIDIA Nemotron 3 Nano Omni image-to-text
        |
        v
Google Gemini 2.5 Flash
        |
        v
Streaming overlay answer
```

If NVIDIA visual extraction is unavailable, rate-limited, overloaded, or returns unusable text, the original image is sent directly to Gemini 2.5 Flash. There is **no Ollama/local-model fallback**.

## Models

- Final answer model: `gemini-2.5-flash`
- NVIDIA speech model: Parakeet CTC 1.1B English via function ID `1598d209-5e27-4d3c-8079-4751568b1081`
- NVIDIA visual extractor: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

## Setup

```powershell
cd C:\Users\ADMIN\Desktop\cascade-interview-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.template .env
notepad .env
```

Set your own credentials in `.env`:

```env
PRACTICE_MODE=1
AI_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_KEY
NVIDIA_API_KEY=YOUR_NVIDIA_KEY
```

Never commit real API keys. If a key has been pasted into chat, logs, screenshots, or source control, revoke it and create a new one.

Run:

```powershell
python main.py
```

## Manual controls

- **Listen**: starts microphone + Windows system-audio capture and NVIDIA Parakeet transcription.
- **Stop**: stops audio capture.
- **Capture screen** / `Ctrl+Shift+S`: captures one screenshot and answers the visible practice question.
- **Send**: sends a typed question to Gemini 2.5 Flash.
- **Camera**: analyzes the latest live camera frame manually.
- `Ctrl+Shift+A`: toggles Listen/Stop.

Nothing starts listening automatically, and there is no background screenshot inference.

## Expected logs

Audio question:

```text
[audio] Mic stream started.
[audio] Default-speaker WASAPI loopback started @ 48000 Hz.
[stt] Interviewer: Tell me about yourself
[pipeline] Parakeet transcript/chat -> Gemini 2.5 Flash
[gemini] First text answer token in ...s · model=gemini-2.5-flash
```

Screen capture:

```text
[capture] Manual screenshot captured · ...
[pipeline] Manual image -> NVIDIA Omni extraction · bytes=...
[vision-nvidia] Extracted visual context in ...s · chars=...
[pipeline] NVIDIA visual text -> Gemini 2.5 Flash · ...
[gemini] First text answer token in ...s · model=gemini-2.5-flash
```

NVIDIA visual fallback:

```text
[vision] NVIDIA extraction unusable/unavailable: ...
[vision] Falling back to direct Gemini 2.5 Flash image analysis for this capture.
[gemini] First vision answer token in ...s · model=gemini-2.5-flash
```

## Candidate context

Optional private candidate/resume/project context can be stored at:

```text
data/candidate_context.local.md
```

The file is git-ignored. General technical/coding questions do not require it.
