# Counts, for each Open Images class, how many validation and test images contain it and how
# many pass a minimum box-size filter. It reads only the small annotation files (about 90 MB)
# and downloads no photos, so you can plan quotas before the big download.

import argparse
import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

BASE = "https://storage.googleapis.com/openimages/v5/"
DESCRIPTION_URLS = [
    BASE + "class-descriptions-boxable.csv",
    "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
]
SPLIT_FILES = {"validation": "validation-annotations-bbox.csv", "test": "test-annotations-bbox.csv"}
AREAS = [0.25, 0.15, 0.10]  # minimum box size as a fraction of the image
CACHE = config.PROJECT_ROOT / "downloads" / "oi_annotations"


def fetch(urls: list[str], name: str) -> Path:
    # Download a file once into the cache folder, trying each URL in turn.
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
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


def wanted_classes(cli_classes: list[str] | None) -> list[str]:
    # Use the classes given on the command line, otherwise every class in TARGETS.
    if cli_classes:
        return cli_classes
    from download_openimages import TARGETS  # lives in the same folder as this script
    return sorted({t[1] for t in TARGETS})


def load_ids(path: Path, names: set[str]) -> dict[str, str]:
    # Map Open Images label ids (like /m/0k4j) to the class names we care about.
    ids = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] in names:
                ids[row[0]] = row[1]
    return ids


def largest_boxes(path: Path, ids: dict[str, str]) -> dict[str, dict[str, float]]:
    # For every class, the area (fraction of the image) of its biggest box in each image.
    best: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = ids.get(row["LabelName"])
            if name is None:
                continue
            width = float(row["XMax"]) - float(row["XMin"])
            height = float(row["YMax"]) - float(row["YMin"])
            image = row["ImageID"]
            if width * height > best[name].get(image, 0.0):
                best[name][image] = width * height
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Count available Open Images per class.")
    parser.add_argument("--classes", nargs="+", help="Class names (default: all classes in TARGETS)")
    args = parser.parse_args()

    names = wanted_classes(args.classes)
    ids = load_ids(fetch(DESCRIPTION_URLS, "class-descriptions-boxable.csv"), set(names))
    missing = sorted(set(names) - set(ids.values()))
    for name in missing:
        print(f"WARNING: '{name}' is not an Open Images boxable class.")

    per_split = {split: largest_boxes(fetch([BASE + file], file), ids) for split, file in SPLIT_FILES.items()}

    header = f"{'class':16s}{'val':>6s}{'test':>7s}{'total':>7s}"
    for area in AREAS:
        header += f"{'>=' + format(area, '.0%'):>8s}"
    print("\n" + header)
    for name in sorted(set(names) - set(missing)):
        val = per_split["validation"].get(name, {})
        test = per_split["test"].get(name, {})
        areas = list(val.values()) + list(test.values())
        row = f"{name:16s}{len(val):6d}{len(test):7d}{len(areas):7d}"
        for area in AREAS:
            row += f"{sum(a >= area for a in areas):8d}"
        print(row)


if __name__ == "__main__":
    main()