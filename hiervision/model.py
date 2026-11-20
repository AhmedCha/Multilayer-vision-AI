# Manages the VGG16 shared backbone and the individual linear classification heads for each node.
# It handles creating, loading, and saving these components to disk.
# Keeping the backbone frozen and separate allows hierarchical training without memory exhaustion.

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

import config

def get_device() -> torch.device:
    """Finds the best available hardware accelerator to run the model on."""
    # We prefer CUDA (NVIDIA) for speed, then MPS (Apple Silicon), then CPU as a fallback.
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_backbone() -> nn.Module:
    """Loads the VGG16 model, removes its final classification layer, and freezes its weights."""
    # We use ImageNet weights because they provide great general-purpose feature extraction.
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    # VGG16's classifier ends with a Linear(4096, 1000) layer at index 6.
    # We replace it with an Identity layer so it just outputs the 4096-dimensional feature vector.
    model.classifier[6] = nn.Identity()

    # We freeze all parameters in the backbone so they do not update during training.
    # WHY: This drastically reduces VRAM usage (allowing larger batches on our 8GB GPU),
    # speeds up training, and prevents our small node datasets from destroying the good ImageNet features.
    for param in model.parameters():
        param.requires_grad = False

    # We set the model to evaluation mode (turns off dropout) because it's only acting as a feature extractor.
    model.eval()

    # Move the model to the GPU (or chosen device) once, so it's ready for forward passes.
    model = model.to(get_device())
    return model

def make_head(n_classes: int) -> nn.Module:
    """Creates a new linear classification head for a specific node."""
    # The backbone outputs 4096 features, so we map from 4096 to the number of classes for this node.
    head = nn.Linear(4096, n_classes)
    return head.to(get_device())

def model_path(node_path: str) -> Path:
    """Determines the file path where a node's model is saved."""
    # We look up the base models directory from config.py.
    # If node_path is empty (""), this just resolves to MODELS_DIR / "model.pt".
    return config.MODELS_DIR / node_path / "model.pt"

def save_head(node_path: str, head: nn.Module, class_names: List[str]) -> None:
    """Saves the trained classification head and its class names to disk."""
    path = model_path(node_path)

    # We must ensure the folder exists before we try to write a file into it.
    path.parent.mkdir(parents=True, exist_ok=True)

    # We save both the weights and the list of class names so the model can map predictions to labels.
    checkpoint = {
        'state_dict': head.state_dict(),
        'class_names': class_names
    }
    torch.save(checkpoint, path)

def load_head(node_path: str) -> Optional[Tuple[nn.Module, List[str]]]:
    """Loads a classification head and its class names from disk, if they exist."""
    path = model_path(node_path)

    # If the file hasn't been created yet, this node hasn't been trained (or doesn't exist).
    if not path.exists():
        return None

    # Load the saved dictionary containing our weights and strings.
    # We use weights_only=False because we are loading string lists (class_names), not just tensors.
    checkpoint = torch.load(path, map_location=get_device(), weights_only=False)

    class_names = checkpoint['class_names']

    # We recreate the head structure with the correct output size, then load the weights into it.
    head = make_head(len(class_names))
    head.load_state_dict(checkpoint['state_dict'])

    # Put the head into eval mode so it's ready for inference.
    head.eval()

    return head, class_names
