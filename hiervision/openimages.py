# Manages the Open Images catalogue and bounding box statistics without needing external libraries.
# Downloads annotation CSVs, streams them to prevent huge memory spikes, and tracks box areas.

import csv
import datetime
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
import sys

# Ensure config can be imported when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

BASE = "https://storage.googleapis.com/openimages/v5/"
DESCRIPTION_URLS = [
    BASE + "class-descriptions-boxable.csv",
    "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
]
SPLIT_FILES = {
    "validation": "validation-annotations-bbox.csv",
    "test": "test-annotations-bbox.csv"
}

CACHE_DIR = config.PROJECT_ROOT / "downloads" / "oi_annotations"
CATALOGUE_FILE = CACHE_DIR / "catalogue.json"


def _fetch(urls: list[str], name: str) -> Path:
    # Download a file once into the cache folder, trying each URL in turn.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists():
        return dest
    for url in urls:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"  failed: {e}")
            dest.unlink(missing_ok=True)  # remove a partial file
    sys.exit(f"Could not download {name}")


def _largest_boxes(path: Path, ids: dict[str, str], tracker: dict[str, dict[str, float]]) -> None:
    # Stream the CSV row by row and update the tracker with the max box area for each image/class.
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = ids.get(row["LabelName"])
            if name is None:
                continue
            width = float(row["XMax"]) - float(row["XMin"])
            height = float(row["YMax"]) - float(row["YMin"])
            area = width * height
            image = row["ImageID"]

            if area > tracker[name].get(image, 0.0):
                tracker[name][image] = area


def build_catalogue() -> dict:
    # Downloads annotations, maps classes, and counts box size distributions for every boxable class.
    desc_path = _fetch(DESCRIPTION_URLS, "class-descriptions-boxable.csv")

    ids = {}
    with open(desc_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                ids[row[0]] = row[1]

    # Structure: tracker[split][class_name][image_id] = max_area
    val_tracker = defaultdict(dict)
    test_tracker = defaultdict(dict)

    val_path = _fetch([BASE + SPLIT_FILES["validation"]], SPLIT_FILES["validation"])
    _largest_boxes(val_path, ids, val_tracker)

    test_path = _fetch([BASE + SPLIT_FILES["test"]], SPLIT_FILES["test"])
    _largest_boxes(test_path, ids, test_tracker)

    # Process counts for thresholds: 0.05, 0.10, ..., 0.50
    classes_dict = {}
    all_class_names = set(val_tracker.keys()) | set(test_tracker.keys())

    for cls in all_class_names:
        val_areas = list(val_tracker[cls].values())
        test_areas = list(test_tracker[cls].values())
        all_areas = val_areas + test_areas

        counts = []
        for i in range(1, 11):
            threshold = i * 0.05
            # Add small epsilon to float comparison to prevent precision errors
            counts.append(sum(1 for a in all_areas if a >= (threshold - 1e-6)))

        classes_dict[cls] = {
            "val": len(val_areas),
            "test": len(test_areas),
            "total": len(val_areas) + len(test_areas),
            "counts": counts
        }

    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "classes": classes_dict
    }


def save_catalogue(catalogue: dict) -> None:
    # Safely save the generated catalogue.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOGUE_FILE, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2)


def load_catalogue() -> dict | None:
    # Load the catalogue if it exists.
    if not CATALOGUE_FILE.exists():
        return None
    try:
        with open(CATALOGUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def available(entry: dict, min_area: float) -> int:
    # Return the image count for the next 0.05 threshold at or above min_area.
    idx = int(math.ceil(min_area / 0.05)) - 1
    idx = max(0, idx)
    if idx > 9:
        return 0
    return entry["counts"][idx]


def class_names(catalogue: dict) -> list[str]:
    # A cleanly sorted list of available classes in the catalogue.
    return sorted(catalogue["classes"].keys())