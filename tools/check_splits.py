# Prints how many images of each class fall into train / val / test for every node.
# Run it after downloading data and before training, to spot classes with too few images.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from hiervision import data


def main() -> None:
    for node in data.find_nodes(config.DATA_DIR):
        print(f"\nNode '{node or 'root'}'")
        for cls in data.list_classes(node):
            counts = {"train": 0, "val": 0, "test": 0}
            for path in data.list_images(config.DATA_DIR / node / cls):
                rel = path.relative_to(config.DATA_DIR).as_posix()
                counts[data.get_split(rel)] += 1
            total = max(sum(counts.values()), 1)
            flag = "  <-- too few for val/test" if counts["val"] < 30 or counts["test"] < 30 else ""
            print(
                f"  {cls:14s} total={sum(counts.values()):5d}  "
                f"train={counts['train']:5d} ({counts['train'] / total:.0%})  "
                f"val={counts['val']:4d} ({counts['val'] / total:.0%})  "
                f"test={counts['test']:4d} ({counts['test'] / total:.0%}){flag}"
            )


if __name__ == "__main__":
    main()