# Reports byte-identical images under data/. The same photo in train and test inflates accuracy.

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from hiervision import data


def main() -> None:
    groups = defaultdict(list)
    for path in data.list_images(config.DATA_DIR):
        groups[hashlib.md5(path.read_bytes()).hexdigest()].append(path)

    found = 0
    for paths in groups.values():
        if len(paths) > 1:
            found += 1
            same_folder = len({p.parent for p in paths}) == 1
            note = "duplicate" if same_folder else "LABEL CONFLICT (different folders)"
            print(f"\n{note}:")
            for p in sorted(paths):
                print(f"  {p}")
    print(f"\n{found} duplicate groups found.")


if __name__ == "__main__":
    main()