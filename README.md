# quntumnintent

A Windows desktop practice assistant with a lightweight overlay, real-time microphone/system-audio transcription, chat, and screen-aware technical Q&A.

## AI stack

- **Google Gemini** for text, chat, and vision reasoning. Default model: `gemini-2.5-flash`.
- **NVIDIA Nemotron streaming ASR** for low-latency real-time speech-to-text through NVIDIA Riva.

There is no OpenAI provider or OpenAI SDK path in the application.

## Main workflow

- API credentials load automatically from the project-local `.env` when the app starts.
- Settings never store or display Gemini/NVIDIA API keys.
- In practice mode, listening can start automatically at launch.
- Interviewer speech is transcribed in real time and can automatically trigger a concise Gemini answer.
- Typed chat questions are answered using recent transcript context.
- Screen capture sends the configured screen region plus recent transcript context to Gemini Vision.
- Speech, chat, and screen requests share an internal queue, so requests are not discarded while another answer is running.

Automatic speech coaching is intended only for mock interviews, practice sessions, or environments where AI assistance is explicitly permitted.

## Requirements

- Windows 10/11 recommended for system-audio loopback and protected overlay features.
- Python 3.9+.
- Gemini API key.
- NVIDIA API key for the hosted Nemotron/Riva speech endpoint.

## Install

```bash
git clone https://github.com/Aravindh-dev12/cascade-interview-assistant.git
cd cascade-interview-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Configure `.env`

Copy the template once:

```bash
copy .env.template .env
```

Put your credentials in `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_RIVA_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_RIVA_FUNCTION_ID=bb0837de-8c7b-481f-9ec8-ef5663e9c1fa
NVIDIA_RIVA_LANGUAGE=en-US

# Enable hands-free transcript -> answer only for permitted practice use.
PRACTICE_MODE=1
```

No API-key entry is required inside the Settings window. On startup, Settings reports whether Gemini and NVIDIA credentials were detected from `.env`.

Never commit a real `.env` file or credentials. If a key has ever been committed to Git history, revoke/rotate it even after replacing the current file with placeholders.

## Real-time voice flow

1. `AudioRecorder` captures 16 kHz microphone audio and, when available, Windows loopback/system audio.
2. `STTWorker` opens NVIDIA Riva streaming sessions and emits interim transcripts immediately.
3. Final transcript lines are added to Gemini conversation context.
4. With `PRACTICE_MODE=1` and **Auto-answer** enabled, substantive interviewer speech queues a Gemini response automatically.
5. New speech, chat, and screen requests wait in the same answer queue instead of being dropped.

Latency can be tuned in `.env`:

```env
ASR_ENDPOINT_SECONDS=0.50
ASR_VAD_THRESHOLD=0.005
ASR_MAX_UTTERANCE_SECONDS=20
GEMINI_FAST_MIN_CHARS=120
GEMINI_FAST_MAX_CHARS=700
```

## Chat and screen answers

Use the chat field in the overlay and press **Send** (or Enter) for a text question based on recent context.

Use **Capture screen** or `Ctrl+Shift+S` to capture the configured region and ask Gemini to solve or explain the visible question, code, diagram, or MCQ.

## Shortcuts

- `Ctrl+Shift+A` — toggle live listening.
- `Ctrl+Shift+S` — capture screen region and answer.

## Run

```bash
python main.py
```

If `PRACTICE_MODE=1`, `NVIDIA_API_KEY` is available, and **Start listening automatically** is enabled in Settings, the voice listener starts automatically shortly after launch.

## Security

`.env.template` must contain placeholders only. Real credentials belong only in the ignored local `.env`; they are not copied into the app settings JSON.
