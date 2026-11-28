# Counts available Open Images per class based on the cached catalogue.
# Refactored as a lightweight CLI over hiervision.openimages.

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from hiervision import openimages
from hiervision.targets import load_targets

def main() -> None:
    parser = argparse.ArgumentParser(description="Count available Open Images per class.")
    parser.add_argument("--classes", nargs="+", help="Class names (default: targets.json or all if --all)")
    parser.add_argument("--all", action="store_true", help="Show all classes in the catalogue.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the catalogue from CSVs.")
    args = parser.parse_args()

    if args.refresh:
        print("Rebuilding catalogue...")
        cat = openimages.build_catalogue()
        openimages.save_catalogue(cat)
    else:
        cat = openimages.load_catalogue()
        if cat is None:
            print("Catalogue not found. Building for the first time...")
            cat = openimages.build_catalogue()
            openimages.save_catalogue(cat)

    if args.all:
        names = openimages.class_names(cat)
    elif args.classes:
        names = args.classes
    else:
        targets = load_targets()
        names = sorted({t.cls for t in targets})

    missing = [n for n in names if n not in cat["classes"]]
    for name in missing:
        print(f"WARNING: '{name}' is not an Open Images boxable class or has no boxes.")

    names = [n for n in names if n in cat["classes"]]

    header = f"{'class':16s}{'val':>6s}{'test':>7s}{'total':>7s}{'>=25%':>8s}{'>=15%':>8s}{'>=10%':>8s}"
    print("\n" + header)

    for name in sorted(names):
        entry = cat["classes"][name]
        val = entry["val"]
        test = entry["test"]
        total = entry["total"]
        c_25 = openimages.available(entry, 0.25)
        c_15 = openimages.available(entry, 0.15)
        c_10 = openimages.available(entry, 0.10)

        print(f"{name:16s}{val:6d}{test:7d}{total:7d}{c_25:8d}{c_15:8d}{c_10:8d}")


if __name__ == "__main__":
    main()