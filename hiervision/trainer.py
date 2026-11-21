# Trains the classification heads for individual nodes in the hierarchy.
# Uses a frozen VGG16 backbone and trains a linear layer with AMP for speed.
# Evaluates on the test split, producing a classification report and confusion matrix.

import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.optim import AdamW

import config
from hiervision import data, model


def set_seed(seed: int) -> None:
    # Set seeds to ensure deterministic behavior across runs using the config seed.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    head: nn.Module, backbone: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, list[int], list[int]]:
    # Run inference over a dataset to compute accuracy and collect predictions for evaluation.
    head.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    # Disable gradient tracking since we are only evaluating.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # The backbone is completely frozen and used only for feature extraction.
            features = backbone(images)

            # Use Automatic Mixed Precision (AMP) for faster inference.
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                outputs = head(features)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    return acc, y_true, y_pred


def save_confusion_matrix(node_path: str, y_true: list[int], y_pred: list[int], classes: list[str]) -> None:
    # Generate and save a confusion matrix plot to visualize misclassifications.
    out_dir = config.REPORTS_DIR / node_path
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    # Use sklearn's built-in display to format the grid automatically with our class names.
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, xticks_rotation='vertical', ax=ax, cmap='Blues'
    )
    plt.tight_layout()

    out_file = out_dir / "confusion_matrix.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_file}")


def train_node(node_path: str) -> None:
    # Orchestrate the training of a single node's classification head.
    classes = data.list_classes(node_path)

    # Skip nodes with fewer than 2 classes since there's nothing to distinguish.
    if len(classes) < 2:
        print(f"Skipping '{node_path}': needs at least 2 classes, found {len(classes)}.")
        return

    print(f"\n--- Training node: '{node_path}' ---")
    device = model.get_device()
    backbone = model.load_backbone()
    head = model.make_head(len(classes))

    # Initialize DataLoaders utilizing the hash-based deterministic split.
    train_loader = data.get_loader(node_path, "train")
    val_loader = data.get_loader(node_path, "val")
    test_loader = data.get_loader(node_path, "test")

    # Setup loss with inverse frequency weights to handle imbalanced folders.
    weights = data.class_weights(node_path).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Train only the head using AdamW.
    optimizer = AdamW(head.parameters(), lr=config.LR)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None

    best_val_acc = -1.0
    best_state = None

    for epoch in range(config.EPOCHS):
        head.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Extract features from the frozen VGG16 backbone without tracking gradients.
            with torch.no_grad():
                features = backbone(images)

            # Mixed precision (AMP) for faster training and lower VRAM usage on the RTX 3050.
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                outputs = head(features)
                loss = criterion(outputs, labels)

            # Scale loss and step if AMP is enabled, otherwise standard backward pass.
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate on the validation set to track the best model.
        val_acc, _, _ = evaluate(head, backbone, val_loader, device)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Keep track of the highest validation accuracy head.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Copy the state dict to memory so we don't accidentally save an overfit later epoch.
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # Restore the best head for testing and saving.
    if best_state is not None:
        head.load_state_dict(best_state)

    model.save_head(node_path, head, classes)
    print(f"Saved best head for '{node_path}' with Val Acc: {best_val_acc:.4f}")

    # Final evaluation on the unseen test split.
    test_acc, y_true, y_pred = evaluate(head, backbone, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")

    # zero_division=0 prevents warnings for classes that might not appear in small test sets.
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    save_confusion_matrix(node_path, y_true, y_pred, classes)


def main() -> None:
    # Parses command line arguments to train either one specific node or all valid nodes in the hierarchy.
    parser = argparse.ArgumentParser(description="Train HierVision node classification heads.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--node", type=str, help="Path of the node to train (e.g. '', 'animal/mammal')")
    group.add_argument("--all", action="store_true", help="Train all nodes found in the dataset")
    args = parser.parse_args()

    set_seed(config.SEED)

    if args.all:
        # Find all valid node directories in the taxonomy and train them sequentially.
        nodes = data.find_nodes(config.DATA_DIR)
        for node in nodes:
            train_node(node)
    else:
        # Resolve 'root' alias to the empty string to match folder parsing.
        node_path = "" if args.node == "root" else args.node
        train_node(node_path)