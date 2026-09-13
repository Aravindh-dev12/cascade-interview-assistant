import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.private_context_store import context_store_status, import_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import local resume/project markdown or text into quntumnintent's private context store. "
            "On Windows, stored chunks are protected with the current user's DPAPI credentials."
        )
    )
    parser.add_argument("files", nargs="+", help="Markdown/text files to import")
    args = parser.parse_args()

    total = 0
    for raw in args.files:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        count = import_file(path, source=str(path))
        total += count
        print(f"[context-import] {path} -> {count} encrypted chunks")

    status = context_store_status()
    print(
        f"[context-import] Done · imported={total} · store={status['path']} · "
        f"total_chunks={status['chunks']} · protection={status['protection']}"
    )


if __name__ == "__main__":
    main()
