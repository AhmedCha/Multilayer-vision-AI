# Downloads Open Images V7 datasets using the fiftyone library.
# Filters images to ensure the primary subject takes up at least 25% of the frame.
# Processes one class at a time to minimize RAM and disk usage.

import os
import sys
import shutil
import argparse
import gc
import socket
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Tuple, Set

os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.openimages import get_classes

# Add project root to sys.path to import config since this runs as a standalone script
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

SPLITS = ["validation", "test"]

# List of (destination folder, Open Images class name, target max images)
TARGETS = [
    # (folder, Open Images class, quota, minimum box area as a fraction of the image)

    # ---------- vehicle ----------
    ("vehicle/car", "Car", 600, 0.25),
    ("vehicle/truck", "Truck", 700, 0.15),
    ("vehicle/bus", "Bus", 316, 0.10),
    ("vehicle/train", "Train", 326, 0.10),
    ("vehicle/bicycle", "Bicycle", 650, 0.10),
    ("vehicle/plane", "Airplane", 600, 0.25),
    ("vehicle/motorcycle", "Motorcycle", 486, 0.10),
    ("vehicle/other", "Boat", 150, 0.25),
    ("vehicle/other", "Helicopter", 100, 0.25),
    ("vehicle/other", "Tank", 100, 0.15),
    ("vehicle/other", "Snowmobile", 46, 0.10),
    ("vehicle/other", "Golf cart", 33, 0.10),

    # ---------- animal / mammal ----------
    ("animal/mammal/cat", "Cat", 600, 0.25),
    ("animal/mammal/dog", "Dog", 600, 0.25),
    ("animal/mammal/elephant", "Elephant", 163, 0.10),
    ("animal/mammal/lion", "Lion", 192, 0.10),
    ("animal/mammal/other", "Horse", 50, 0.25),
    ("animal/mammal/other", "Sheep", 40, 0.25),
    ("animal/mammal/other", "Cattle", 40, 0.25),
    ("animal/mammal/other", "Giraffe", 30, 0.10),
    ("animal/mammal/other", "Zebra", 30, 0.10),
    ("animal/mammal/other", "Bear", 40, 0.15),
    ("animal/mammal/other", "Fox", 30, 0.15),
    ("animal/mammal/other", "Deer", 30, 0.15),
    ("animal/mammal/other", "Rabbit", 30, 0.15),
    ("animal/mammal/other", "Squirrel", 30, 0.15),
    ("animal/mammal/other", "Monkey", 40, 0.15),
    ("animal/mammal/other", "Camel", 25, 0.15),
    ("animal/mammal/other", "Kangaroo", 25, 0.15),
    ("animal/mammal/other", "Tiger", 30, 0.15),      # looks like a lion: a useful hard negative

    # ---------- animal / bird (crow and pigeon are not in Open Images) ----------
    ("animal/bird/duck", "Duck", 224, 0.10),
    ("animal/bird/eagle", "Eagle", 244, 0.10),
    ("animal/bird/canary", "Canary", 78, 0.10),
    ("animal/bird/other", "Parrot", 40, 0.15),
    ("animal/bird/other", "Owl", 30, 0.15),
    ("animal/bird/other", "Penguin", 30, 0.15),
    ("animal/bird/other", "Sparrow", 30, 0.15),
    ("animal/bird/other", "Swan", 30, 0.15),         # hard negative for duck
    ("animal/bird/other", "Goose", 30, 0.15),        # hard negative for duck
    ("animal/bird/other", "Chicken", 30, 0.15),
    ("animal/bird/other", "Turkey", 20, 0.15),
    ("animal/bird/other", "Ostrich", 25, 0.15),
    ("animal/bird/other", "Falcon", 30, 0.15),       # hard negative for eagle
    ("animal/bird/other", "Woodpecker", 20, 0.15),

    # ---------- building (leaf class directly under data/) ----------
    ("building", "Building", 400, 0.25),
    ("building", "House", 250, 0.25),
    ("building", "Tower", 100, 0.25),
    ("building", "Skyscraper", 100, 0.15),
    ("building", "Castle", 60, 0.15),
    ("building", "Lighthouse", 30, 0.10),

    # ---------- other (root level: things that are none of the above) ----------
    ("other", "Person", 150, 0.25),
    ("other", "Tree", 100, 0.25),
    ("other", "Flower", 100, 0.25),
    ("other", "Plant", 60, 0.25),
    ("other", "Chair", 60, 0.15),
    ("other", "Table", 60, 0.15),
    ("other", "Couch", 40, 0.10),
    ("other", "Bed", 40, 0.25),
    ("other", "Laptop", 50, 0.15),
    ("other", "Mobile phone", 50, 0.25),
    ("other", "Television", 40, 0.15),
    ("other", "Clock", 20, 0.10),
    ("other", "Lamp", 30, 0.15),
    ("other", "Bottle", 50, 0.15),
    ("other", "Coffee cup", 40, 0.25),
    ("other", "Pizza", 40, 0.25),
    ("other", "Cake", 40, 0.25),
    ("other", "Apple", 30, 0.15),
    ("other", "Banana", 30, 0.15),
    ("other", "Hat", 40, 0.10),
    ("other", "Footwear", 40, 0.10),
    ("other", "Handbag", 30, 0.25),
    ("other", "Book", 40, 0.15),
    ("other", "Guitar", 30, 0.25),
    ("other", "Toy", 40, 0.25),
]

# Attempt to limit thread pool workers if this FiftyOne version supports it
try:
    fo.config.max_thread_pool_workers = 4
except AttributeError:
    pass

def get_existing_images() -> Set[str]:
    """Finds all existing oi_ files anywhere in the data directory."""
    if not config.DATA_DIR.exists():
        return set()
    return {p.name for p in config.DATA_DIR.rglob("oi_*")}

def check_safety(active_paths: List[Path]) -> None:
    """Ensures nodes do not receive images and destination folders are clean."""
    for p1 in active_paths:
        for p2 in active_paths:
            if p1 != p2 and p1 in p2.parents:
                raise ValueError(f"Target '{p1}' is a parent of '{p2}'. A node cannot receive images.")

    for p in active_paths:
        dest_path = config.DATA_DIR / p
        if dest_path.exists():
            for item in dest_path.iterdir():
                if item.is_dir() and not item.name.startswith((".", "_")):
                    raise ValueError(f"{dest_path} has subfolders. Nodes cannot contain images.")

        parent = dest_path.parent
        while parent.is_relative_to(config.DATA_DIR) and parent != config.DATA_DIR:
            if parent.exists() and any(i.is_file() for i in parent.iterdir()):
                raise ValueError(f"Parent folder {parent} contains files. Nodes cannot contain images.")
            parent = parent.parent

def process_class(cls: str, cls_targets: List[Tuple[str, str, int]], existing_files: Set[str]) -> None:
    """Loads and processes a single class, filtering by bounding box area."""
    # Track quotas strictly per (folder, class) tuple
    quotas = {(t[0], t[1]): {"max": t[2], "current": 0} for t in cls_targets}
    total_quota = sum(t[2] for t in cls_targets)
    
    print(f"Loading '{cls}' (max {3 * total_quota} samples)...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7", splits=SPLITS, label_types=["detections"],
        classes=[cls], max_samples=3 * total_quota, shuffle=True, seed=config.SEED
    )

    processed = 0
    for sample in dataset:
        if all(q["current"] >= q["max"] for q in quotas.values()):
            break
        
        filename = f"oi_{Path(sample.filepath).name}"
        if filename in existing_files:
            continue

        largest_area = 0.0
        detections = sample.ground_truth.detections if sample.ground_truth else []
        for det in detections:
            if det.label == cls:
                _, _, w, h = det.bounding_box
                if (w * h) > largest_area:
                    largest_area = w * h

        if largest_area >= 0.25:
            for (folder, _), q in quotas.items():
                if q["current"] < q["max"]:
                    dest = config.DATA_DIR / folder / filename
                    shutil.copy(sample.filepath, dest)
                    q["current"] += 1
                    existing_files.add(filename)
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Processed {processed} images for '{cls}'...")
                    break  # One image only satisfies one quota slot to prevent duplicates

    total_found = sum(q["current"] for q in quotas.values())
    if total_found < total_quota:
        print(f"{cls}: found {total_found} of {total_quota} quota, use another source")

    # Clean up the single-class dataset to save memory and disk space
    fo.delete_dataset(dataset.name)
    gc.collect()

def main() -> None:
    # Keep the dataset directory fixed and persistent
    fo.config.dataset_zoo_dir = str(config.PROJECT_ROOT / "downloads")

    parser = argparse.ArgumentParser(description="Download and filter Open Images.")
    parser.add_argument("--list-classes", action="store_true", help="Print all valid classes.")
    parser.add_argument("--only", type=str, help="Only process targets matching PREFIX.")
    parser.add_argument("--dry-run", action="store_true", help="Print classes and quotas, then exit.")
    args = parser.parse_args()

    if not (args.list_classes or args.dry_run):
        uri = os.environ["FIFTYONE_DATABASE_URI"]
        parsed_uri = urlparse(uri)
        host = parsed_uri.hostname or "localhost"
        port = parsed_uri.port or 27017
        try:
            socket.create_connection((host, port), timeout=3.0).close()
        except OSError:
            print(f"Cannot reach MongoDB at {uri}. Start it with:\ndocker start fiftyone-mongo")
            sys.exit(1)

    valid_classes = get_classes()
    if args.list_classes:
        print("\n".join(valid_classes))
        return

    active_targets = [t for t in TARGETS if not args.only or t[0].startswith(args.only)]
    
    # Check targets against valid classes at startup
    for t in active_targets:
        if t[1] not in valid_classes:
            print(f"WARNING: Class '{t[1]}' is not a valid Open Images V7 class.")

    if args.dry_run:
        for t in active_targets:
            print(f"Folder: {t[0]}, Class: {t[1]}, Quota: {t[2]}")
        return

    check_safety([Path(t[0]) for t in active_targets])
    
    # Ensure destination directories exist
    for t in active_targets:
        (config.DATA_DIR / t[0]).mkdir(parents=True, exist_ok=True)

    existing = get_existing_images()
    unique_classes = list(set(t[1] for t in active_targets))
    failed_classes = []
    
    for cls in unique_classes:
        try:
            process_class(cls, [t for t in active_targets if t[1] == cls], existing)
        except Exception as e:
            print(f"ERROR in {cls}: {e}")
            failed_classes.append(cls)
            continue

    if failed_classes:
        print("\nThe following classes failed to process:")
        for fc in failed_classes:
            print(f" - {fc}")

if __name__ == "__main__":
    main()