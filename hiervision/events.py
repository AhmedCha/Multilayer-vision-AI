# Handles writing and reading live progress events as JSON lines.
# Uses only the standard library so it can be safely imported by any tool.

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_event_file: Optional[Path] = None


def set_file(path: Optional[str | Path]) -> None:
    """Sets the global file path where events will be appended."""
    global _event_file
    if path is None:
        _event_file = None
    else:
        _event_file = Path(path)
        # Ensure the directory exists
        _event_file.parent.mkdir(parents=True, exist_ok=True)


def emit(kind: str, **fields: Any) -> None:
    """Appends an event as a JSON line and flushes the file. Does nothing if no file is set."""
    if _event_file is None:
        return

    event = {"type": kind, "time": time.time()}
    event.update(fields)

    # Append and flush immediately so the app can read it live
    with open(_event_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()


def read_events(path: str | Path) -> List[Dict[str, Any]]:
    """Reads all events, skipping a potentially incomplete JSON line at the end."""
    path = Path(path)
    if not path.exists():
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # If it's the last line, it might be mid-write. Ignore it.
            if i == len(lines) - 1:
                pass
            else:
                # If a middle line is corrupted, re-raise.
                raise
    return events