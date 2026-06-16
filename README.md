# 🤖 quntumnintent (Realtime Overlay)

An ultra-modern, real-time desktop helper application designed for coding and technical interviews. It captures audio (both candidate voice and interviewer voice via Zoom/Teams/Slack) and screen regions, processes them with advanced AI (GPT-4o, Claude 3.5 Sonnet), and displays highly readable answers in under 5 seconds.

### 🌟 Core Feature: The "Invisible" Overlay (Anti-Capture Screen Share Protection)
This application implements a Windows-exclusive system-level display protection using the native `SetWindowDisplayAffinity` Win32 API. 

* **What it does:** When you share your screen in **Microsoft Teams, Zoom, Google Meet, Slack, OBS, Discord, or take screenshots**, this floating overlay window becomes **completely transparent/invisible** on the shared screen or recording!
* **How you see it:** On your physical monitor, the overlay remains **100% visible, fully interactive, and translucent**. You can read solutions, code snippets, and talking points in real-time, while everyone else in the meeting sees absolutely nothing but your clean background!
* **Verification:** The window features a green `🔒 PRIVACY ACTIVE` badge on the top title bar indicating that capture protection is fully functional.

---

## 🛠️ Key Features
1. **Real-Time Speech-to-Text (STT):** Continuous background audio capture using an energy-based Voice Activity Detector (VAD). It records:
   * **Your Voice** (via your native Microphone)
   * **Interviewer's Voice** (via Windows WASAPI Loopback, intercepting speech from Meet, Zoom, Teams, or Slack)
2. **Real-Time Transcription Feed:** Shows a rolling history of the spoken dialogue, color-coded by speaker (Candidate vs. Interviewer).
3. **Smart Vision Screen Capture:** Select a custom bounding box (e.g. over LeetCode, system design diagrams, or slides). Pressing a hotkey grabs that region and sends it with the spoken context to the AI's vision engine.
4. **Multi-Model Support (Focus on Privacy & Free High-Tier Models):**
   * **OpenRouter (Default)** - Configure keys to run high-tier models for free. Defaults to **GLM 4.5 Air (free)** (`zhipuai/glm-4-5-air:free`)!
   * **Google Gemini** - Highly capable, excellent vision, cost-effective API.
   * **Ollama (100% Local & Offline)** - Run open-source models (Llama 3, Mistral, CodeLlama, Phi-3, Gemma 2, and Llava for vision) completely locally, with **absolute privacy and zero API costs**!
5. **Interview-Optimized UI:** Floating, draggable, resizable, borderless panel. Answers are rendered in beautifully formatted, high-contrast Markdown, focused on **instant scanability** (talking points, optimal code blocks, and complexities).
6. **Global Keyboard Hotkeys:** Activate capabilities from anywhere in the OS without having to click on the overlay window:
   * `Ctrl+Shift+S` : Triggers screen-region snapshot analysis.
   * `Ctrl+Shift+A` : Toggles audio recording (Start/Stop listening).

---

## ⚙️ Installation & Setup

### Prerequisites
* Windows 10 (version 2004 and above) or Windows 11.
* Python 3.9 - 3.11.
* C compiler or visual build tools (sometimes required for compiling `sounddevice` or sound utilities if wheel is unavailable, though pip wheels are pre-compiled).

### 1. Clone & Set Up Directory
Open your terminal (PowerShell or Command Prompt) and install dependencies:
```bash
# Navigate to project directory
cd C:\Users\ADMIN\CascadeProjects\ai-interview-copilot

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Create a `.env` file in the project root based on `.env.template` to load your API keys automatically, or you can paste your API keys directly into the UI Settings panel.

### 🐳 3. Running Local Models with Ollama (100% Offline & Private)
If you want to use open-source models completely locally (no internet, no logging, zero cost):
1. Download and install Ollama from [ollama.com](https://ollama.com/).
2. Run your desired model in PowerShell or Command Prompt. For example:
   * **For optimal coding/text answers:**
     ```bash
     ollama run llama3
     ```
   * **For screen capture/vision answers (Highly Recommended):**
     To support analyzing code/diagrams from screen grabs locally, pull and run a vision-capable multimodal model:
     ```bash
     ollama run llava
     ```
3. Keep Ollama running in the background. The Interview Copilot will automatically detect all downloaded models on your machine!

---

## 🚀 How to Run & Use

### 1. Launch the Application
Run the entry point file:
```bash
python main.py
```
You will see the floating, semi-transparent overlay appear on your desktop.

### 2. Configure Settings (Crucial First Step)
1. Click the **⚙️ (Gear)** button in the bottom corner of the overlay.
2. **AI Provider:** Select `openrouter`, `gemini`, or `ollama` (default: `openrouter`).
   * **If using OpenRouter:** Paste your API Key or set it in `.env`. It will default to the free, ultra-capable **`zhipuai/glm-4-5-air:free`** model!
   * **If using Gemini:** Paste your API Key or set it in `.env`.
   * **If using Ollama:** No API Key is required! Ensure the Ollama app is running locally. The **Model Selection** drop-down will automatically query and list all downloaded models on your machine (e.g. `llama3:latest`, `llava:latest`).
3. **Your Voice (Mic):** Choose your primary microphone device.
4. **Interviewer (System Output):** Select your primary system audio device (e.g. Headset, Speakers, or Virtual Cable) listed as **`[Loopback]`**. This is required to capture the interviewer's voice.
5. **Select Screen Capture Region:**
   * Click this button. The screen will dim.
   * **Click and drag a box** around your browser's coding editor (like LeetCode), slides, or whiteboard area.
   * On release, the coordinates are saved and the settings dialog returns.
6. **Overlay Opacity & Font:** Customize transparency levels and font sizing to match your screen resolution and visual comfort.
7. Click **Save Settings**.

### 3. Start an Interview Simulation
1. Click **🎤 Start Listening** or press `Ctrl+Shift+A` globally.
2. Speak into your microphone and play some system audio (e.g., a YouTube coding tutorial or a mock call). You will see live transcribing rolling in the lower panel:
   * `Candidate: [Your transcribed speech]`
   * `Interviewer: [System transcribed speech]`
3. When you run into a coding problem or system design slide, press `Ctrl+Shift+S` globally or click **📸 Capture Screen**.
4. The display panel will show `● THINKING...` and output a structured answer in 3-5 seconds!

---

## 🛡️ Interview Cheat-Sheet Tips
* **🗣️ "What to Say" Section:** Look at the bottom of the AI output. There is always a section called `🗣️ WHAT TO SAY`. Read this out loud immediately to buy yourself time or start explaining the concept while you read through the full code snippet!
* **Readability:** Set the opacity to ~85% to 90%. It is translucent enough to see text underneath, but opaque enough to read code hints clearly.
* **Keep On Top:** Keep "Always on Top" enabled in settings so the window doesn't get pushed behind your browser or IDE when you type!
