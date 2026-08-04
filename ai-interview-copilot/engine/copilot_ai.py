import base64
import os
import io
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

def build_system_prompt(language="python"):
    return f"""You are an elite technical interview co-pilot. Answer ALL question types: coding, MCQ, theoretical, system design, SQL.

Previous Q&A in this session is provided as chat history. For follow-ups, continue the conversation — don't restart.

Question types & handling:
- CODING: Complete solution, exact function signature, handle edge cases, optimal complexity. Use {language}.
- MATH/QUANTITATIVE: Transcribe every symbol, number, exponent, fraction, and diagram label first.
  Solve independently, substitute the result back into the original problem, and only then answer.
  If any character is unreadable or cropped, say what is ambiguous instead of guessing.
- MCQ: State correct option letter + text. Explain why it's right and others are wrong.
- THEORETICAL: Clear, concise explanation with brief example.
- SYSTEM DESIGN/SQL: Structured answer with trade-offs.

Output format:
**Answer:** [correct option / code / explanation]
**Explanation:** [brief reasoning]
**🗣️ Say:** [a natural first-person answer the candidate can speak immediately]

The Say section must sound like a thoughtful human in an interview: use contractions,
short sentences, and natural transitions. Never mention AI, screenshots, transcripts,
prompts, or hidden context. Avoid robotic headings inside that section.

CRITICAL: Code must be production-ready. Always state MCQ answer clearly. Be accurate,
natural, and concise."""

class CopilotAI:
    def __init__(self, provider="openai", model="gpt-5.6-luna", api_key=None):
        self.provider = provider
        self.model = model
        self.code_language = "python"
        
        # Load API key with environmental fallbacks
        self.api_key = api_key or self._environment_key(provider)
        
        # Transcript history storage: list of dicts: {"speaker": str, "text": str}
        self.transcript_history = []
        self.max_transcript_history = 30
        
        # Chat conversation history: list of dicts: {"role": "user"/"assistant", "text": str}
        self.chat_history = []
        self.max_chat_history = 20

    def set_config(self, provider, model, api_key):
        """
        Maintains API compatibility with the UI overlay settings dialog.
        """
        self.provider = provider or "openai"
        default_model = "gpt-5.6-luna" if self.provider == "openai" else "gemini-2.0-flash"
        self.model = model or default_model
        self.api_key = api_key or self._environment_key(self.provider)

    @staticmethod
    def _environment_key(provider):
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    def set_language(self, language):
        """Sets the preferred programming language for code solutions."""
        self.code_language = language

    def add_chat_exchange(self, role, text):
        """Stores a Q&A exchange for continuous conversation context."""
        self.chat_history.append({"role": role, "text": text})
        if len(self.chat_history) > self.max_chat_history * 2:
            self.chat_history = self.chat_history[-self.max_chat_history * 2:]

    def clear_chat(self):
        """Clears the conversation chat history for a fresh start."""
        self.chat_history.clear()

    def get_formatted_chat_history(self):
        """Formats the chat history as context for the AI."""
        if not self.chat_history:
            return "[No previous conversation in this session]"
        lines = []
        for item in self.chat_history:
            label = "Candidate" if item["role"] == "user" else "Co-pilot"
            lines.append(f"{label}: {item['text']}")
        return "\n".join(lines)

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

    def generate_answer(
        self, image_bytes=None, custom_query=None, on_delta=None, spoken_only=False
    ):
        """
        Generates an answer from Gemini based on chat history, transcript, and an optional screen capture.
        Maintains continuous conversation context across follow-up questions.
        """
        chat_ctx = self.get_formatted_chat_history()
        transcript = self.get_formatted_transcript()
        
        # Trim context for speed: only last 5 chat exchanges and last 5 transcript lines
        chat_lines = chat_ctx.split("\n")
        if len(chat_lines) > 10:
            chat_lines = chat_lines[-10:]
        transcript_lines = transcript.split("\n")
        if len(transcript_lines) > 5:
            transcript_lines = transcript_lines[-5:]
        
        user_prompt = f"--- CHAT HISTORY ---\n{chr(10).join(chat_lines)}\n\n"
        user_prompt += f"--- TRANSCRIPT ---\n{chr(10).join(transcript_lines)}\n\n"
        
        if spoken_only:
            user_prompt += (
                "--- TASK ---\n"
                "Reply to the interviewer's latest real question as the candidate. "
                "Output only the exact natural words to say aloud. Use one short "
                "paragraph, usually 2-4 sentences. Start directly with the answer. "
                "Do not use headings, labels, markdown, bullet points, explanations "
                "about your process, or phrases such as 'I would say'.\n"
            )
        elif custom_query:
            user_prompt += f"--- CANDIDATE FOLLOW-UP QUERY ---\n{custom_query}\n"
            user_prompt += "--- TASK ---\nAnswer the candidate's follow-up question in the context of the previous chat history. If it's related to a previous question, reference and build upon the previous answer. Do NOT start a new problem.\n"
        else:
            user_prompt += "--- TASK ---\nThe image may contain up to three numbered SCROLL VIEW sections from the SAME webpage/question. Reconstruct one complete question in top-to-bottom reading order and merge overlapping repeated text only once. Ignore browser navigation, ads, chat controls, timers, score panels, and editor boilerplate.\nFIRST transcribe the exact problem statement, notation, examples, constraints, function signature, input/output format, and choices visible across all views. Do not guess from a familiar opening pattern; distinguish the problem using its exact constraints and examples.\n- For CODING: derive the algorithm from the complete reconstructed statement, obey the platform function signature, handle all stated constraints and edge cases, and provide runnable selected-language code with time/space complexity.\n- For MATH: calculate, verify independently, and compare every option. Never invent a cropped or blurry symbol.\n- For MCQ, identify the correct option and explain why.\n- For THEORETICAL questions, give a clear concise explanation.\n- If essential information remains below the fold or unreadable, state exactly what is missing and ask for one more scroll capture instead of guessing.\nAnswer ALL question types.\n"

        # Credentials are intentionally loaded only from .env/environment.
        if not self.api_key:
            self.api_key = self._environment_key(self.provider)
            
        if not self.api_key:
            variable = "OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"
            return f"### API Key Missing\n\nAdd `{variable}` to the project `.env` file."

        try:
            if self.provider == "openai":
                answer = self._call_openai(
                    user_prompt,
                    image_bytes,
                    on_delta=on_delta,
                    spoken_only=spoken_only,
                )
            else:
                answer = self._call_gemini(user_prompt, image_bytes)
            # Store this exchange in chat history for continuous conversation
            query_ref = custom_query if custom_query else "[Screen capture analysis]"
            self.add_chat_exchange("user", query_ref)
            self.add_chat_exchange("assistant", answer)
            return answer
        except Exception as e:
            return f"### AI request failed\n\nDetails:\n`{str(e)}`"

    def _call_openai(
        self, prompt, image_bytes=None, on_delta=None, spoken_only=False
    ):
        """Use the OpenAI Responses API for text and high-detail vision."""
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", self.api_key))
        content = [{"type": "input_text", "text": prompt}]
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{encoded}",
                "detail": "high",
            })

        model = os.environ.get("OPENAI_MODEL", self.model or "gpt-5.6-luna")
        print(f"[openai] Querying Responses API model: {model}...")
        try:
            request = dict(
                model=model,
                instructions=build_system_prompt(self.code_language),
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "none"},
                max_output_tokens=(
                    3000 if image_bytes else (320 if spoken_only else 1400)
                ),
            )
            if on_delta:
                chunks = []
                with client.responses.stream(**request) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            chunks.append(event.delta)
                            on_delta("".join(chunks))
                    response = stream.get_final_response()
                return response.output_text or "".join(chunks)
            response = client.responses.create(**request)
            return response.output_text
        except Exception as openai_error:
            gemini_key = os.environ.get(
                "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")
            )
            if not gemini_key:
                raise
            print(f"[openai] Request failed; using Gemini fallback: {str(openai_error)[:160]}")
            previous_key = self.api_key
            try:
                self.api_key = gemini_key
                return self._call_gemini(prompt, image_bytes)
            finally:
                self.api_key = previous_key

    def _call_gemini(self, prompt, image_bytes=None):
        """
        Executes fast vision and text queries with Google Gemini.
        Uses gemini-2.0-flash-lite for text (fastest) and gemini-2.0-flash for images.
        Falls back to OpenAI GPT-4o-mini on 429 quota errors.
        """
        client = genai.Client(api_key=self.api_key)
        
        contents = [prompt]
        if image_bytes:
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_bytes))
            contents.append(pil_img)
            
        system_prompt = build_system_prompt(self.code_language)
        
        # Use flash-lite for text-only (faster), flash for images (vision capable)
        model_name = "gemini-2.0-flash-lite" if not image_bytes else "gemini-2.0-flash"
        max_tokens = 2048 if not image_bytes else 4096
        
        # Try Gemini (high free tier quota: 1500 RPD)
        try:
            print(f"[gemini] Querying model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as gemini_err:
            error_str = str(gemini_err).lower()
            print(f"[gemini] gemini-2.0-flash failed: {str(gemini_err)[:200]}")
            
            # If it's a 429 quota error, try OpenAI fallback
            if "429" in error_str or "exhaust" in error_str or "quota" in error_str:
                openai_key = os.environ.get("OPENAI_API_KEY", "")
                if openai_key:
                    print("[gemini] Quota exceeded. Falling back to OpenAI GPT-4o-mini...")
                    fallback = self._call_openai_fallback(prompt, image_bytes, openai_key)
                    if fallback:
                        return fallback
                
                # If no OpenAI fallback, try gemini-1.5-flash as last resort
                try:
                    print("[gemini] Trying gemini-1.5-flash as last resort...")
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=max_tokens
                        )
                    )
                    return response.text
                except Exception as e2:
                    print(f"[gemini] gemini-1.5-flash also failed: {str(e2)[:200]}")
                    raise gemini_err
            else:
                # Non-quota error (503, auth, etc.) — retry once with 0.5s delay
                import time
                time.sleep(0.5)
                try:
                    print(f"[gemini] Retrying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=max_tokens
                        )
                    )
                    return response.text
                except Exception as retry_err:
                    # Try OpenAI as final fallback
                    openai_key = os.environ.get("OPENAI_API_KEY", "")
                    if openai_key:
                        print("[gemini] Retry failed. Falling back to OpenAI...")
                        fallback = self._call_openai_fallback(prompt, image_bytes, openai_key)
                        if fallback:
                            return fallback
                    raise retry_err

    def _call_openai_fallback(self, prompt, image_bytes=None, api_key=None):
        """Fallback to OpenAI GPT-4o-mini when Gemini quota is exhausted."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            system_prompt = build_system_prompt(self.code_language)
            
            if image_bytes:
                import base64
                b64_img = base64.b64encode(image_bytes).decode("utf-8")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            print("[openai] Querying GPT-4o-mini fallback...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=4096
            )
            print("[openai] Fallback successful.")
            return response.choices[0].message.content
        except Exception as e:
            print(f"[openai] Fallback failed: {str(e)[:200]}")
            return None
