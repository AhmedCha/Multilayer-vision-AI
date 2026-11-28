# hiervision/runner.py
# Starts and tracks background processes like training and downloading.
# Saves active state so the UI can reconnect to runs, and cleanly reaps children.

import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import config
from hiervision import events

# Keeps track of spawned Popen objects so we can check their status directly.
_active_processes: Dict[int, subprocess.Popen] = {}


def current() -> Optional[Dict]:
    """Returns the details of the currently tracked run, or None."""
    current_file = config.REPORTS_DIR / "runs" / "current.json"
    if not current_file.exists():
        return None
    with open(current_file, "r", encoding="utf-8") as f:
        return json.load(f)


def is_running() -> bool:
    """Checks if the current run's process is still alive."""
    curr = current()
    if not curr:
        return False

    pid = curr["pid"]

    # If we started it in this exact Python session, poll the Popen object
    if pid in _active_processes:
        proc = _active_processes[pid]
        return proc.poll() is None

    # Otherwise, check the OS process table
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(kind: str, command: List[str]) -> Dict:
    """Starts a command in the background, creating a new run directory."""
    if is_running():
        raise RuntimeError("A run is already currently running.")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.REPORTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.txt"
    events_path = run_dir / "events.jsonl"

    log_file = open(log_path, "w", encoding="utf-8")

    # Environment needs to be unbuffered so logs flow in real-time
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Start the child in a new process group so we can cleanly kill it later
    proc = subprocess.Popen(
        command,
        cwd=config.BASE_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True
    )

    _active_processes[proc.pid] = proc

    run_info = {
        "run_id": run_id,
        "kind": kind,
        "pid": proc.pid,
        "events": str(events_path),
        "log": str(log_path),
        "started": time.time(),
        "command": command
    }

    current_file = config.REPORTS_DIR / "runs" / "current.json"
    current_file.parent.mkdir(parents=True, exist_ok=True)
    with open(current_file, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    return run_info


def start_training(nodes: List[str]) -> Dict:
    """Helper to start the training script with the correct event parameters."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = config.REPORTS_DIR / "runs" / run_id / "events.jsonl"

    command = [sys.executable, "train.py", "--node"] + nodes + ["--events", str(events_path)]
    return start("train", command)


def start_download(prefixes: List[str]) -> Dict:
    """Helper to start the download script with the correct event parameters."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = config.REPORTS_DIR / "runs" / run_id / "events.jsonl"

    command = [sys.executable, "tools/download_openimages.py", "--events", str(events_path)]
    if prefixes:
        command.extend(["--only"] + prefixes)
    return start("download", command)


def stop() -> None:
    """Stops the currently running process and its entire process group."""
    curr = current()
    if not curr or not is_running():
        return

    try:
        # SIGTERM to the whole process group ensures children like dataloaders die too
        os.killpg(os.getpgid(curr["pid"]), signal.SIGTERM)
    except OSError:
        pass


def get_events() -> List[Dict]:
    """Returns all parsed events for the current run."""
    curr = current()
    if not curr:
        return []
    return events.read_events(curr["events"])


def log_tail(lines: int = 30) -> str:
    """Returns the last N lines of the current run's log file."""
    curr = current()
    if not curr:
        return ""

    log_path = Path(curr["log"])
    if not log_path.exists():
        return ""

    # Read entire file for simplicity since logs aren't massively huge in this scope
    with open(log_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    return "".join(all_lines[-lines:])