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

# These phrases mean the answer depends on what is visually present. Keep this
# deliberately narrower than general technical vocabulary so behavioral/conceptual
# questions such as "tell me about yourself" stay on the fast text path.
SCREEN_REFERENCE_TERMS = (
    "this code", "this error", "this output", "this question", "this problem",
    "this function", "this query", "this diagram", "this screenshot", "on the screen",
    "shown here", "shown on", "above code", "below code", "visible", "fix this",
    "solve this", "debug this", "read this", "look at this", "take a look",
    "what is wrong here", "what's wrong here", "what will be the output",
    "what is the output", "output of this", "which option", "choose the correct",
    "multiple choice", "mcq", "coding problem", "problem statement", "given array",
    "given string", "given matrix", "given linked list", "given tree",
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
    """Detect answer-worthy interviewer turns from imperfect streaming ASR."""
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

    interrogatives = {"what", "why", "how", "when", "where", "which", "who"}
    if len(words) >= 5 and interrogatives.intersection(words) and ({"you", "your"} & set(words)):
        return True

    return False


def should_attach_camera(text: str) -> bool:
    lower = normalize_text(text).lower()
    return any(term in lower for term in CAMERA_REFERENCE_TERMS)


def should_attach_screen(text: str) -> bool:
    """Return true only when the spoken/text prompt actually depends on visuals."""
    lower = normalize_text(text).lower()
    return any(term in lower for term in SCREEN_REFERENCE_TERMS) or any(
        term in lower for term in CAMERA_REFERENCE_TERMS
    )
