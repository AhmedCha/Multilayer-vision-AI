# Manages the targets.json configuration for downloading Open Images data.
# Loads and validates quotas, ensures taxonomy rules, and handles file naming conventions.

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys

# Ensure config can be imported securely when running as a script or tool
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

@dataclass
class Target:
    folder: str
    cls: str
    quota: int
    min_area: float = 0.25
    note: str = ""

def load_targets(path: Path | str = config.TARGETS_FILE) -> list[Target]:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list.")

    targets = []
    for i, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} is not a valid JSON object.")
        if not all(k in entry for k in ("folder", "class", "quota")):
            raise ValueError(f"Entry {i} is missing required keys (folder, class, quota).")

        folder = entry["folder"]
        if not folder or Path(folder).is_absolute() or ".." in Path(folder).parts:
            raise ValueError(f"Entry {i} has an invalid, empty, absolute, or '..' folder: '{folder}'.")

        quota = entry["quota"]
        if not isinstance(quota, int) or quota < 1:
            raise ValueError(f"Entry {i} has an invalid quota (must be integer >= 1).")

        min_area = entry.get("min_area", 0.25)
        if not isinstance(min_area, (int, float)) or not (0 < min_area <= 1):
            raise ValueError(f"Entry {i} has an invalid min_area (must be strictly between 0 and 1).")

        targets.append(Target(
            folder=folder,
            cls=entry["class"],
            quota=quota,
            min_area=float(min_area),
            note=entry.get("note", "")
        ))
    return targets

def save_targets(targets: list[Target], path: Path | str = config.TARGETS_FILE) -> None:
    path = Path(path)
    if path.exists():
        shutil.copy(path, path.with_name(path.name + ".bak"))

    data = [
        {"folder": t.folder, "class": t.cls, "quota": t.quota, "min_area": t.min_area, "note": t.note}
        for t in targets
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_problems(targets: list[Target]) -> list[str]:
    problems = []
    seen = set()
    folders = set()

    for t in targets:
        if (t.folder, t.cls) in seen:
            problems.append(f"Duplicate target configuration: folder '{t.folder}', class '{t.cls}'.")
        seen.add((t.folder, t.cls))
        folders.add(t.folder)

    folder_paths = [Path(f) for f in folders]

    for p1 in folder_paths:
        for p2 in folder_paths:
            if p1 != p2 and p1 in p2.parents:
                problems.append(f"Target '{p1}' is a parent of '{p2}'. A node cannot receive images.")

    for p in folder_paths:
        dest_path = config.DATA_DIR / p
        if dest_path.exists():
            for item in dest_path.iterdir():
                if item.is_dir() and not item.name.startswith((".", "_")):
                    problems.append(f"Folder '{p}' already has subfolders on disk. Nodes cannot contain images.")

        parent = dest_path.parent
        while parent.is_relative_to(config.DATA_DIR) and parent != config.DATA_DIR:
            if parent.exists():
                for item in parent.iterdir():
                    if item.is_file() and not item.name.startswith((".", "_")):
                        problems.append(f"Parent folder '{parent.relative_to(config.DATA_DIR)}' contains loose files. Nodes cannot contain images.")
                        break
            parent = parent.parent

    return problems

def class_slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '-', name.lower())

def image_id(filename: str) -> str:
    return Path(filename).stem.split('_')[-1]

def source_tag(filename: str) -> str:
    if not filename.startswith("oi_"):
        return "other source"
    
    parts = Path(filename).stem.split('_')
    if len(parts) >= 3:
        # Reconstruct slug from oi_SLUG_ID pattern
        return "_".join(parts[1:-1])
    return "untagged"