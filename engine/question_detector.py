import re

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "which", "who", "whose", "can you",
    "could you", "would you", "do you", "did you", "have you", "are you", "is it",
    "what would", "how would", "explain", "describe", "compare", "differentiate",
    "implement", "write", "solve", "design", "debug", "optimize", "define", "discuss",
    "outline", "share", "walk me through", "walk us through", "take me through",
    "tell me about", "tell us about", "talk about", "give me", "give us",
    "please explain", "please describe", "please tell", "suppose", "imagine", "assume",
)

INTERVIEW_TASK_TERMS = (
    "difference between", "time complexity", "space complexity", "tradeoff", "trade-off",
    "algorithm", "data structure", "system design", "architecture", "database", "sql",
    "api", "thread", "process", "memory", "cache", "network", "code", "function",
    "class", "object", "error", "bug", "output", "query", "approach", "solution",
    "experience", "project", "challenge", "conflict", "leadership", "strength",
    "weakness", "responsibility", "role", "resume", "team", "decision", "failure",
    "success", "deadline", "priority", "customer", "stakeholder", "debugging",
    "performance", "scalability", "availability", "security", "testing", "deployment",
)

INTERVIEW_INTENT_PHRASES = (
    "your experience", "your approach", "your thought", "your opinion", "your role",
    "your responsibility", "your biggest", "your strongest", "your weakness",
    "a time when", "an example of", "you handled", "you solved", "you designed",
    "you implemented", "you optimized", "you debugged", "you would", "would you",
    "can you", "could you", "tell me", "tell us", "walk me", "walk us", "take me",
)

SCREEN_REFERENCE_TERMS = (
    "this code", "this error", "this output", "this question", "on the screen", "shown here",
    "shown on", "above code", "below code", "diagram", "screenshot", "visible", "fix this",
    "solve this", "what is wrong here", "what's wrong here",
)

CAMERA_REFERENCE_TERMS = (
    "camera", "webcam", "look at this", "look at me", "what am i holding",
    "what i'm holding", "what i am holding", "what do you see", "showing you",
    "in front of the camera", "this object", "this item", "this device",
)

ACK_ONLY = re.compile(
    r"^(okay|ok|right|great|good|nice|sure|thanks|thank you|interesting|got it|alright|cool|yes|no|perfect|awesome|understood|makes sense)[.! ]*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def is_substantive_question(text: str) -> bool:
    """Detect answer-worthy interviewer turns from imperfect streaming ASR.

    Streaming ASR often omits a trailing question mark or slightly damages the
    first word of a question. Treat explicit questions, common interview
    imperatives, technical/behavioral prompts, and sufficiently clear
    second-person interrogatives as answer-worthy while filtering short
    acknowledgements.
    """
    clean = normalize_text(text)
    if len(clean) < 9 or ACK_ONLY.match(clean):
        return False

    lower = clean.lower()
    words = re.findall(r"[a-z0-9']+", lower)

    if "?" in clean:
        return True
    if any(lower.startswith(prefix) for prefix in QUESTION_STARTERS):
        return True
    if any(phrase in lower for phrase in INTERVIEW_INTENT_PHRASES):
        return True
    if any(term in lower for term in INTERVIEW_TASK_TERMS) and len(words) >= 3:
        return True

    # Recover questions where ASR dropped/mangled the opening word but retained
    # an interrogative later in the utterance, e.g. "for this role how would you...".
    interrogatives = {"what", "why", "how", "when", "where", "which", "who"}
    if len(words) >= 5 and interrogatives.intersection(words) and ({"you", "your"} & set(words)):
        return True

    return False


def should_attach_camera(text: str) -> bool:
    lower = normalize_text(text).lower()
    return any(term in lower for term in CAMERA_REFERENCE_TERMS)


def should_attach_screen(text: str) -> bool:
    """Return true when the existing overlay should attach visual context.

    The overlay historically knows only a single ``latest_screen_bytes`` slot. The
    real-time multimodal bridge publishes a labeled SCREEN+CAMERA composite there,
    so camera-referential prompts use the same proven vision-request path.
    """
    lower = normalize_text(text).lower()
    return any(term in lower for term in SCREEN_REFERENCE_TERMS) or any(
        term in lower for term in CAMERA_REFERENCE_TERMS
    )
