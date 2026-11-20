# Data loading and validation for the hierarchical image classifier.
# Handles taxonomy traversal, deterministic hashing for data splits, PyTorch datasets,
# and sanity checking to ensure folders do not mix images and subfolders.

import os
import hashlib
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


def list_images(folder: Path) -> list[Path]:
    # Recursively find images since leaf classes might contain subfolders of sorted images.
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in extensions]


def find_nodes(data_dir: Path) -> list[str]:
    # A node is any directory that contains at least one subdirectory.
    nodes = []
    for root, dirs, files in os.walk(data_dir):
        if dirs:
            rel = os.path.relpath(root, data_dir)
            # Use "" for the root node to keep CLI commands simple and consistent.
            nodes.append("" if rel == "." else rel.replace("\\", "/"))
    return nodes


def list_classes(node_path: str) -> list[str]:
    # Classes are simply the direct subdirectories of a node, sorted alphabetically for deterministic indexing.
    full_path = config.DATA_DIR / node_path # using config.DATA_DIR
    return sorted([d.name for d in full_path.iterdir() if d.is_dir()])


def get_split(rel_path: str) -> str:
    # Hash the relative path so the same image falls in the exact same split in every node.
    # This completely prevents data leakage across the hierarchy without needing a split file.
    md5_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    # Use modulo on the first few hex digits to assign a percentage bucket (0-99).
    bucket = int(md5_hash[:8], 16) % 100

    # Split proportions: 80% train, 10% val, 10% test.
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"


def make_transforms(train: bool) -> transforms.Compose:
    # Use standard ImageNet normalization because the backbone is a pre-trained VGG16.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        # Mild augmentation to prevent overfitting on small datasets.
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(config.IMG_SIZE), # 224 pixels
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Deterministic crops for validation and testing.
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE), # 224 pixels
            transforms.ToTensor(),
            normalize
        ])


class NodeDataset(Dataset):
    def __init__(self, node_path: str, split: str):
        # A torch Dataset allowed by design rules.
        self.transform = make_transforms(train=(split == "train"))
        self.classes = list_classes(node_path)
        self.samples = []

        # Gather all matching images for this node's classes
        for class_idx, class_name in enumerate(self.classes):
            class_dir = config.DATA_DIR / node_path / class_name # using config.DATA_DIR
            for img_path in list_images(class_dir):
                # Use standard posix relative paths from the overall data root for stable hashing.
                rel_img_path = img_path.relative_to(config.DATA_DIR).as_posix() # using config.DATA_DIR
                if get_split(rel_img_path) == split:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        # Convert to RGB to ensure grayscale and RGBA images don't crash the convolutions.
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), class_idx


def get_loader(node_path: str, split: str) -> DataLoader:
    # Encapsulate the dataset into a DataLoader for batching.
    dataset = NodeDataset(node_path, split)
    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE, # hardware-optimized batch size
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS # hardware-optimized worker count
    )


def class_weights(node_path: str) -> torch.Tensor:
    # Calculate inverse frequency weights so rare classes impact the loss more heavily.
    # Ensures nodes train correctly even if image counts between subfolders are severely unbalanced.
    classes = list_classes(node_path)
    counts = []

    for class_name in classes:
        class_dir = config.DATA_DIR / node_path / class_name # using config.DATA_DIR
        counts.append(len(list_images(class_dir)))

    total = sum(counts)
    # Add a tiny epsilon to prevent division by zero in empty classes.
    weights = [total / (c + 1e-6) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def check_data() -> None:
    # Validates directory structure, image integrity, and minimum class sizes.
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Structural check
    for root, dirs, files in os.walk(config.DATA_DIR): # using config.DATA_DIR
        has_subfolders = len(dirs) > 0
        img_files = [f for f in files if Path(f).suffix.lower() in extensions]
        has_images = len(img_files) > 0

        # A folder can be a node or a leaf class, but never contains both.
        if has_subfolders and has_images:
            raise ValueError(f"Folder mixes images and subfolders: {root}")

        # Corruption check
        for f in img_files:
            img_path = Path(root) / f
            try:
                # verify() reads headers without loading full pixel data.
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Corrupt image found: {img_path} ({e})")

    # Size and composition report
    nodes = find_nodes(config.DATA_DIR) # using config.DATA_DIR
    for node in nodes:
        print(f"Node: '{node}'")
        for cls in list_classes(node):
            cls_dir = config.DATA_DIR / node / cls # using config.DATA_DIR
            count = len(list_images(cls_dir))
            print(f"  - {cls}: {count} images")
            if count < 20:
                print(f"  WARNING: Class '{cls}' in node '{node}' has fewer than 20 images.")
