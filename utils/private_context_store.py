import ctypes
import getpass
import hashlib
import os
import re
import sqlite3
import time
from ctypes import wintypes
from pathlib import Path


APP_NAME = "quntumnintent"


def _default_private_dir():
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if root:
            return Path(root) / APP_NAME
    return Path.home() / f".{APP_NAME}"


def private_db_path():
    configured = os.environ.get("PRIVATE_CONTEXT_DB", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        return path
    return _default_private_dir() / "candidate_context.local.db"


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _to_blob(data):
    buffer = ctypes.create_string_buffer(data)
    return _DATA_BLOB(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)),
    ), buffer


def _dpapi_protect(data):
    raw = bytes(data or b"")
    if os.name != "nt":
        return b"P0" + raw

    in_blob, in_buffer = _to_blob(raw)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    CRYPTPROTECT_UI_FORBIDDEN = 0x1

    ok = crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "quntumnintent private candidate context",
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    _ = in_buffer
    if not ok:
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return b"D1" + encrypted
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(data):
    raw = bytes(data or b"")
    if raw.startswith(b"P0"):
        return raw[2:]
    if not raw.startswith(b"D1"):
        return raw
    if os.name != "nt":
        raise RuntimeError("Windows DPAPI-encrypted context cannot be decrypted on this OS")

    payload = raw[2:]
    in_blob, in_buffer = _to_blob(payload)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    CRYPTPROTECT_UI_FORBIDDEN = 0x1

    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    _ = in_buffer
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _connect():
    path = private_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS context_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            position INTEGER NOT NULL,
            checksum TEXT NOT NULL UNIQUE,
            content BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_context_source ON context_chunks(source)"
    )
    connection.commit()
    return connection


def _clean_title(value):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:240] or "Candidate context"


def _chunk_markdown(text, max_chars=3600):
    text = str(text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    current_title = "Candidate context"
    current = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        body = "\n".join(current).strip()
        if body:
            chunks.append((_clean_title(current_title), body))
        current = []
        current_len = 0

    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_len >= max_chars // 2:
                flush()
            current_title = heading.group(1).strip()

        incoming = len(line) + 1
        if current and current_len + incoming > max_chars:
            flush()
        current.append(line)
        current_len += incoming
    flush()
    return chunks


def import_text(text, source="manual", replace_source=True):
    chunks = _chunk_markdown(text)
    if not chunks:
        return 0

    connection = _connect()
    try:
        if replace_source:
            connection.execute("DELETE FROM context_chunks WHERE source = ?", (source,))
        inserted = 0
        now = time.time()
        for position, (title, body) in enumerate(chunks):
            digest = hashlib.sha256(
                (source + "\n" + title + "\n" + body).encode("utf-8")
            ).hexdigest()
            encrypted = _dpapi_protect(body.encode("utf-8"))
            connection.execute(
                """
                INSERT OR REPLACE INTO context_chunks
                (source, title, position, checksum, content, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, title, position, digest, encrypted, now),
            )
            inserted += 1
        connection.commit()
        return inserted
    finally:
        connection.close()


def import_file(path, source=None):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    source_name = source or str(path)
    return import_text(text, source=source_name, replace_source=True)


def _tokens(text):
    stop = {
        "about", "after", "again", "also", "answer", "because", "before", "could",
        "from", "have", "into", "just", "latest", "more", "question", "should",
        "that", "their", "there", "these", "they", "this", "using", "what", "when",
        "where", "which", "with", "would", "your",
    }
    values = re.findall(r"[a-zA-Z0-9_+.#-]{3,}", str(text or "").lower())
    return [value for value in values if value not in stop]


def search_context(query, limit=8, max_chars=16000):
    try:
        connection = _connect()
    except Exception as exc:
        print(f"[context-db] Could not open private context store: {exc}")
        return ""

    try:
        rows = connection.execute(
            "SELECT source, title, position, content, updated_at FROM context_chunks"
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        return ""

    query_tokens = _tokens(query)
    ranked = []
    for source, title, position, encrypted, updated_at in rows:
        try:
            body = _dpapi_unprotect(encrypted).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"[context-db] Could not decrypt one context chunk: {exc}")
            continue

        haystack = f"{title}\n{body}".lower()
        if query_tokens:
            score = 0
            title_lower = title.lower()
            for token in query_tokens:
                if token in title_lower:
                    score += 8
                occurrences = haystack.count(token)
                score += min(occurrences, 6)
            if score <= 0:
                continue
        else:
            score = max(1, 1000 - int(position))
        ranked.append((score, updated_at, source, title, position, body))

    if not ranked and rows:
        for source, title, position, encrypted, updated_at in rows[: max(1, limit)]:
            try:
                body = _dpapi_unprotect(encrypted).decode("utf-8", errors="replace")
            except Exception:
                continue
            ranked.append((1, updated_at, source, title, position, body))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    pieces = []
    total = 0
    for _, _, source, title, _, body in ranked[: max(1, int(limit))]:
        block = f"SOURCE: {source}\nSECTION: {title}\n{body.strip()}"
        if pieces and total + len(block) > max_chars:
            break
        if len(block) > max_chars and not pieces:
            block = block[:max_chars]
        pieces.append(block)
        total += len(block)
    return "\n\n---\n\n".join(pieces)


def context_store_status():
    path = private_db_path()
    count = 0
    try:
        connection = _connect()
        try:
            count = int(
                connection.execute("SELECT COUNT(*) FROM context_chunks").fetchone()[0]
            )
        finally:
            connection.close()
    except Exception:
        pass
    protection = "Windows DPAPI" if os.name == "nt" else "local file permissions"
    return {
        "path": path,
        "chunks": count,
        "protection": protection,
        "user": getpass.getuser(),
    }
