# tools/download_openimages.py
# Downloads Open Images V7 datasets using the fiftyone library.
# Reads targets from targets.json and filters images by bounding box area.
# Processes one class at a time to minimize RAM and disk usage.

import os
import sys
import shutil
import argparse
import gc
import socket
import signal
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Set

os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.openimages import get_classes

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from hiervision import events
from hiervision.targets import load_targets, find_problems, class_slug, image_id, Target

SPLITS = ["validation", "test"]

try:
    fo.config.max_thread_pool_workers = 4
except AttributeError:
    pass


def handle_sigterm(signum, frame):
    """Gracefully handles termination signals from the runner."""
    events.emit("run_end", status="stopped", message="Stopped by user")
    sys.exit(0)


def get_existing_image_ids() -> Set[str]:
    """Finds all existing oi_ image IDs anywhere in the data directory."""
    if not config.DATA_DIR.exists():
        return set()
    return {image_id(p.name) for p in config.DATA_DIR.rglob("oi_*")}


def process_class(cls: str, cls_targets: List[Target], existing_ids: Set[str]) -> None:
    """Loads and processes a single class, evaluating multiple quota targets."""
    slug = class_slug(cls)
    quotas = [{"target": t, "max": t.quota, "current": 0} for t in cls_targets]
    total_quota = sum(t.quota for t in cls_targets)
    
    events.emit("class_start", cls=cls, quota=total_quota)
    print(f"Loading '{cls}' (max {3 * total_quota} samples)...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7", splits=SPLITS, label_types=["detections"],
        classes=[cls], max_samples=3 * total_quota, shuffle=True, seed=config.SEED
    )

    processed = 0
    for sample in dataset:
        if all(q["current"] >= q["max"] for q in quotas):
            break
        
        img_id = image_id(Path(sample.filepath).name)
        if img_id in existing_ids:
            continue

        largest_area = 0.0
        detections = sample.ground_truth.detections if sample.ground_truth else []
        for det in detections:
            if det.label == cls:
                _, _, w, h = det.bounding_box
                if (w * h) > largest_area:
                    largest_area = w * h

        for q in quotas:
            target = q["target"]
            if q["current"] < q["max"] and largest_area >= target.min_area:
                filename = f"oi_{slug}_{img_id}.jpg"
                dest = config.DATA_DIR / target.folder / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(sample.filepath, dest)
                
                q["current"] += 1
                existing_ids.add(img_id)
                processed += 1
                
                if processed % 10 == 0:
                    current_found = sum(q_inner["current"] for q_inner in quotas)
                    events.emit("class_progress", cls=cls, added=current_found, quota=total_quota)
                    
                if processed % 50 == 0:
                    print(f"Processed {processed} images for '{cls}'...")
                break

    total_found = sum(q["current"] for q in quotas)
    if total_found < total_quota:
        print(f"{cls}: found {total_found} of {total_quota} quota, use another source")
        
    events.emit("class_end", cls=cls, added=total_found, quota=total_quota, short=(total_found < total_quota))

    fo.delete_dataset(dataset.name)
    gc.collect()


def main() -> None:
    signal.signal(signal.SIGTERM, handle_sigterm)
    fo.config.dataset_zoo_dir = str(config.PROJECT_ROOT / "downloads")

    parser = argparse.ArgumentParser(description="Download and filter Open Images.")
    parser.add_argument("--targets", type=Path, default=config.TARGETS_FILE, help="Path to targets.json")
    parser.add_argument("--list-classes", action="store_true", help="Print all valid classes.")
    parser.add_argument("--only", nargs="+", help="Only process targets matching PREFIX(es).")
    parser.add_argument("--dry-run", action="store_true", help="Print classes and quotas, then exit.")
    parser.add_argument("--events", type=Path, help="Path to events.jsonl for background communication.")
    args = parser.parse_args()

    if args.events:
        events.set_file(args.events)

    try:
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

        all_targets = load_targets(args.targets)
        active_targets = []
        for t in all_targets:
            if not args.only or any(t.folder.startswith(prefix) for prefix in args.only):
                active_targets.append(t)
        
        for t in active_targets:
            if t.cls not in valid_classes:
                print(f"WARNING: Class '{t.cls}' is not a valid Open Images V7 class.")

        if args.dry_run:
            for t in active_targets:
                print(f"Folder: {t.folder}, Class: {t.cls}, Quota: {t.quota}, Min Area: {t.min_area}")
            return

        problems = find_problems(active_targets)
        if problems:
            print("Cannot proceed due to target problems:")
            for p in problems:
                print(f" - {p}")
            sys.exit(1)
            
        folders_to_process = list(set(t.folder for t in active_targets))
        events.emit("run_start", folders=folders_to_process)
        
        existing_ids = get_existing_image_ids()
        unique_classes = list(dict.fromkeys(t.cls for t in active_targets))
        failed_classes = []
        
        for cls in unique_classes:
            try:
                cls_targets = [t for t in active_targets if t.cls == cls]
                process_class(cls, cls_targets, existing_ids)
            except Exception as e:
                print(f"ERROR in {cls}: {e}")
                failed_classes.append(cls)
                continue

        if failed_classes:
            print("\nThe following classes failed to process:")
            for fc in failed_classes:
                print(f" - {fc}")

        print("\nFinal image counts per folder:")
        folders = sorted(list(set(t.folder for t in active_targets)))
        for folder in folders:
            folder_path = config.DATA_DIR / folder
            if folder_path.exists():
                count = sum(1 for item in folder_path.iterdir() if item.is_file() and not item.name.startswith("."))
                print(f"{folder}: {count} files")
            else:
                print(f"{folder}: 0 files")
                
        events.emit("run_end", status="done", message="Completed successfully")

    except Exception as e:
        events.emit("run_end", status="error", message=str(e))
        raise e


if __name__ == "__main__":
    main()