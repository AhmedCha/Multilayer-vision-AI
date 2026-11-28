# Trains the classification heads for individual nodes in the hierarchy.
# Uses a frozen VGG16 backbone and trains a linear layer with AMP for speed.
# Evaluates on the test split, producing a classification report and confusion matrix.

import argparse
import random
import signal
import sys
import time
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import config
from hiervision import data, model, events


def set_seed(seed: int) -> None:
    # Set seeds to ensure deterministic behavior across runs using the config seed.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    head: nn.Module, backbone: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, desc: str = "eval"
) -> tuple[float, list[int], list[int]]:
    # Run inference over a dataset to compute accuracy and collect predictions for evaluation.
    head.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    progress = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not config.SHOW_PROGRESS)

    # Disable gradient tracking since we are only evaluating.
    with torch.no_grad():
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            # Use Automatic Mixed Precision (AMP) on the entire sequence for maximum speed.
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                # The backbone is completely frozen and used only for feature extraction.
                features = backbone(images)
                outputs = head(features)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
            
            progress.set_postfix({"acc": f"{correct/total:.4f}"})

    acc = correct / total if total > 0 else 0.0
    return acc, y_true, y_pred


def save_confusion_matrix(node_path: str, y_true: list[int], y_pred: list[int], classes: list[str]) -> None:
    # Generate and save a confusion matrix plot to visualize misclassifications.
    out_dir = config.REPORTS_DIR / node_path
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    # Pass labels explicitly to prevent failure if a class is entirely missing from predictions.
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(len(classes))), display_labels=classes, xticks_rotation='vertical', ax=ax, cmap='Blues'
    )
    plt.tight_layout()

    out_file = out_dir / "confusion_matrix.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_file}")


def train_node(node_path: str, backbone: nn.Module) -> None:
    # Orchestrate the training of a single node's classification head.
    classes = data.list_classes(node_path)

    # Skip nodes with fewer than 2 classes since there's nothing to distinguish.
    if len(classes) < 2:
        reason = f"needs at least 2 classes, found {len(classes)}"
        tqdm.write(f"Skipping '{node_path}': {reason}.")
        events.emit("node_skipped", node=node_path, reason=reason)
        return

    tqdm.write(f"\n--- Training node: '{node_path}' ---")
    device = model.get_device()
    head = model.make_head(len(classes))

    # Initialize DataLoaders utilizing the hash-based deterministic split.
    train_loader = data.get_loader(node_path, "train")

    # Count train images per class using the dataset samples (which apply the exact same split rule).
    train_counts = [0] * len(classes)
    for _, cls_idx in train_loader.dataset.samples:
        train_counts[cls_idx] += 1

    deficient_classes = [cls for i, cls in enumerate(classes) if train_counts[i] < 10]
    if deficient_classes:
        tqdm.write(f"Classes with < 10 train images: {', '.join(deficient_classes)}")
        tqdm.write("add more images or delete the empty folder")
        return

    val_loader = data.get_loader(node_path, "val")
    test_loader = data.get_loader(node_path, "test")

    if len(val_loader.dataset) == 0:
        tqdm.write(f"Skipping '{node_path}': not enough images, add more, aim for 300 or more per class.")
        return

    # Check for empty classes in validation and test splits
    val_counts = [0] * len(classes)
    for _, cls_idx in val_loader.dataset.samples:
        val_counts[cls_idx] += 1

    test_counts = [0] * len(classes)
    for _, cls_idx in test_loader.dataset.samples:
        test_counts[cls_idx] += 1

    for i, cls in enumerate(classes):
        if val_counts[i] == 0:
            print(f"Warning: Class '{cls}' in '{node_path}' has 0 images in the validation split.")
        if test_counts[i] == 0:
            print(f"Warning: Class '{cls}' in '{node_path}' has 0 images in the test split.")

    batches = len(train_loader)
    events.emit("node_start", node=node_path, classes=classes, train_counts=train_counts,
                val_counts=val_counts, test_counts=test_counts, epochs=config.EPOCHS, batches=batches)

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
        samples_seen = 0
        
        start_time = time.time()
        last_emit_time = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} train", unit="batch", leave=False, disable=not config.SHOW_PROGRESS)

        for batch_idx, (images, labels) in enumerate(progress):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision (AMP) for faster training and lower VRAM usage on the RTX 3050.
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                # Extract features from the frozen VGG16 backbone without tracking gradients.
                with torch.no_grad():
                    features = backbone(images)
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
            samples_seen += images.size(0)
            avg_loss_so_far = total_loss / samples_seen
            progress.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

            # Emit live batch progress at most twice per second, and always on the last batch
            now = time.time()
            if now - last_emit_time >= 0.5 or batch_idx == batches - 1:
                events.emit("batch", node=node_path, epoch=epoch, batch=batch_idx, batches=batches, loss=avg_loss_so_far)
                last_emit_time = now

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate on the validation set to track the best model.
        val_acc, val_y_true, val_y_pred = evaluate(head, backbone, val_loader, device, desc=f"Epoch {epoch+1}/{config.EPOCHS} val")
        epoch_dur = int(time.time() - start_time)
        tqdm.write(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | {epoch_dur}s")

        # Keep track of the highest validation accuracy head.
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True
            # Copy the state dict to memory so we don't accidentally save an overfit later epoch.
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        val_cm = confusion_matrix(val_y_true, val_y_pred, labels=list(range(len(classes)))).tolist()
        events.emit("epoch_end", node=node_path, epoch=epoch, epochs=config.EPOCHS, 
                    train_loss=avg_train_loss, val_acc=val_acc, seconds=epoch_dur, 
                    best=is_best, val_confusion=val_cm)

    # Restore the best head for testing and saving.
    if best_state is not None:
        head.load_state_dict(best_state)

    model.save_head(node_path, head, classes)
    print(f"Saved best head for '{node_path}' with Val Acc: {best_val_acc:.4f}")

    # Prevent crash if the test split happens to be entirely empty.
    if len(test_loader.dataset) == 0:
        tqdm.write(f"Skipping evaluation for '{node_path}': Test split is empty.")
        events.emit("node_end", node=node_path, test_acc=0.0, confusion=[], report={})
        return

    # Final evaluation on the unseen test split.
    test_acc, y_true, y_pred = evaluate(head, backbone, test_loader, device, desc="test")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")

    # Pass labels explicitly so missing classes in the split don't break the metrics report.
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, zero_division=0)
    print(report)
    
    report_dict = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    events.emit("node_end", node=node_path, test_acc=test_acc, confusion=test_cm, report=report_dict)

    save_confusion_matrix(node_path, y_true, y_pred, classes)


def main() -> None:
    # Parses command line arguments to train either one specific node or all valid nodes in the hierarchy.
    parser = argparse.ArgumentParser(description="Train HierVision node classification heads.")
    group = parser.add_mutually_exclusive_group(required=True)
    # --node now accepts one or more nodes to process a custom sequence
    group.add_argument("--node", type=str, nargs="+", help="Path of the nodes to train (e.g. '', 'animal/mammal')")
    group.add_argument("--all", action="store_true", help="Train all nodes found in the dataset")
    group.add_argument("--check", action="store_true", help="Run data validation without training")
    parser.add_argument("--events", type=str, default=None, help="Path to JSONL file to stream live progress events")
    args = parser.parse_args()

    # Install the SIGTERM handler to gracefully stop and record run failure
    def handle_sigterm(signum, frame):
        events.emit("run_end", status="stopped", message="Run manually stopped via SIGTERM")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)

    if args.events:
        events.set_file(args.events)
        config.SHOW_PROGRESS = False

    if args.check:
        corruptions = data.check_data()
        if corruptions == 0:
            print("Data check passed. Hierarchy is valid.")
        return

    set_seed(config.SEED)

    try:
        if args.all:
            nodes = data.find_nodes(config.DATA_DIR)
        else:
            # Resolve 'root' alias to the empty string for all provided nodes
            nodes = ["" if n == "root" else n for n in args.node]

        events.emit("run_start", nodes=nodes)

        print("Loading backbone model...")
        backbone = model.load_backbone()

        if args.all:
            nodes_progress = tqdm(nodes, desc="Nodes", unit="node", disable=not config.SHOW_PROGRESS)
            for node in nodes_progress:
                nodes_progress.set_postfix({"node": "root" if node == "" else node})
                train_node(node, backbone)
        else:
            for node_path in nodes:
                train_node(node_path, backbone)

        events.emit("run_end", status="done", message="Training completed successfully.")
        
    except Exception as e:
        # Wrap everything so errors emit cleanly to the tracker before crashing
        err_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        events.emit("run_end", status="error", message=err_msg)
        raise

if __name__ == "__main__":
    main()