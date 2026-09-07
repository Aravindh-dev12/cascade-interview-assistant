import re

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "which", "who", "whose", "can you",
    "could you", "would you", "do you", "did you", "have you", "are you", "is it",
    "explain", "describe", "compare", "differentiate", "implement", "write", "solve",
    "design", "debug", "optimize", "walk me through", "tell me about", "give me",
)

INTERVIEW_TASK_TERMS = (
    "difference between", "time complexity", "space complexity", "tradeoff", "trade-off",
    "algorithm", "data structure", "system design", "architecture", "database", "sql",
    "api", "thread", "process", "memory", "cache", "network", "code", "function",
    "class", "object", "error", "bug", "output", "query", "approach", "solution",
)

SCREEN_REFERENCE_TERMS = (
    "this code", "this error", "this output", "this question", "on the screen", "shown here",
    "shown on", "above code", "below code", "diagram", "screenshot", "visible", "what do you see",
    "fix this", "solve this", "what is wrong here", "what's wrong here",
)

ACK_ONLY = re.compile(
    r"^(okay|ok|right|great|good|nice|sure|thanks|thank you|interesting|got it|alright|cool|yes|no)[.! ]*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def is_substantive_question(text: str) -> bool:
    """Cheap local detector used before spending an LLM request.

    It intentionally accepts imperative interview prompts such as "Explain dependency injection"
    even when ASR punctuation does not include a question mark.
    """
    clean = normalize_text(text)
    if len(clean) < 9 or ACK_ONLY.match(clean):
        return False

    lower = clean.lower()
    if "?" in clean:
        return True
    if any(lower.startswith(prefix) for prefix in QUESTION_STARTERS):
        return True
    if any(term in lower for term in INTERVIEW_TASK_TERMS) and len(clean) >= 18:
        return True
    return False


def should_attach_screen(text: str) -> bool:
    lower = normalize_text(text).lower()
    return any(term in lower for term in SCREEN_REFERENCE_TERMS)
