import base64
import os
import io
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Minimal token-optimized prompt
SYSTEM_PROMPT = """You are a technical interview co-pilot. Provide the optimal code solution with a brief explanation. No headings, no layouts, no filler. Just code and 1-2 sentences explaining the approach. For complexity, add it inline as a comment."""

class CopilotAI:
    def __init__(self, provider="gemini", model="gemini-2.5-flash", api_key=None):
        self.provider = "gemini"
        self.model = "gemini-2.5-flash"
        
        # Load API key with environmental fallbacks
        self.api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        
        # Transcript history storage: list of dicts: {"speaker": str, "text": str}
        self.transcript_history = []
        self.max_transcript_history = 30

    def set_config(self, provider, model, api_key):
        """
        Maintains API compatibility with the UI overlay settings dialog.
        Always locks provider to gemini and model to gemini-2.5-flash.
        """
        self.provider = "gemini"
        self.model = "gemini-2.5-flash"
        self.api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    def add_transcript_line(self, speaker, text):
        """
        Adds a spoken line from Candidate or Interviewer to the running conversation history.
        """
        self.transcript_history.append({
            "speaker": speaker,
            "text": text
        })
        if len(self.transcript_history) > self.max_transcript_history:
            self.transcript_history.pop(0)

    def clear_history(self):
        self.transcript_history.clear()

    def get_formatted_transcript(self):
        """
        Formats the current transcript history as a string.
        """
        if not self.transcript_history:
            return "[No conversation recorded yet]"
            
        formatted_lines = []
        for item in self.transcript_history:
            formatted_lines.append(f"{item['speaker']}: {item['text']}")
        return "\n".join(formatted_lines)

    def generate_answer(self, image_bytes=None, custom_query=None):
        """
        Generates an answer from Gemini based on transcript history and an optional screen capture.
        """
        transcript = self.get_formatted_transcript()
        user_prompt = f"--- CONVERSATION TRANSCRIPT ---\n{transcript}\n\n"
        
        if custom_query:
            user_prompt += f"--- CANDIDATE SPECIFIC QUERY ---\n{custom_query}\n"
        else:
            user_prompt += "--- TASK ---\nAnalyze the conversation and any provided image. Address the latest spoken question or the coding problem shown on screen. Provide optimal solution details.\n"

        # Ensure we have resolved some API key
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
            
        if not self.api_key:
            return "### 🔑 Gemini API Key Missing\n\nPlease add your `GEMINI_API_KEY` inside your `.env` file to start receiving sub-second co-pilot answers."

        try:
            return self._call_gemini(user_prompt, image_bytes)
        except Exception as e:
            return f"### ❌ Error calling Google Gemini API\n\nDetails:\n`{str(e)}`"

    def _call_gemini(self, prompt, image_bytes=None):
        """
        Executes sub-second vision and text queries directly with Google's modern SDK (google-genai).
        Includes progressive backoff retries exclusively on gemini-2.5-flash to bypass temporary 503/429 spikes.
        """
        # Initialize Google GenAI client
        client = genai.Client(api_key=self.api_key)
        
        contents = [prompt]
        if image_bytes:
            from PIL import Image
            # Gemini SDK handles PIL Images directly
            pil_img = Image.open(io.BytesIO(image_bytes))
            contents.append(pil_img)
            
        # Try up to 5 retries on gemini-2.5-flash with progressive backoff on 503/429 overloads
        max_tries = 5
        last_error = None
        for attempt in range(max_tries):
            try:
                print(f"[gemini] Querying model: gemini-2.5-flash (Attempt {attempt+1}/{max_tries})...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=2048
                    )
                )
                # Success!
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check for temporary Google-side overloads (503) or rate-limits (429)
                if "503" in error_str or "unavail" in error_str or "429" in error_str or "limit" in error_str or "exhaust" in error_str:
                    if attempt < max_tries - 1:
                        import time
                        # Progressive backoff delay: 0.4s, 0.8s, 1.2s, 1.6s
                        delay = 0.4 * (attempt + 1)
                        print(f"[gemini] Model is temporarily busy (503/429). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[gemini] Model exhausted all {max_tries} retries.")
                else:
                    # Non-temporary errors (authentication, bad API key), raise immediately
                    raise e
                    
        if last_error:
            raise last_error
