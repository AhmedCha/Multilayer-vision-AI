# Executes the hierarchical inference pipeline using the cached backbone and heads.
# Walks down the taxonomy tree step by step, applying Grad-CAM explainability,
# and stops if confidence is too low or a leaf node is reached.

import functools
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

import config
import data
import explain
import model


@dataclass
class Step:
    node: str
    label: str
    confidence: float
    top3: List[Tuple[str, float]]
    heatmap: Union[Image.Image, None]


@dataclass
class Result:
    steps: List[Step]
    final_label: str
    status: str


@functools.lru_cache(maxsize=1)
def get_cached_backbone() -> torch.nn.Module:
    # Caches the shared, frozen VGG16 backbone so it is loaded into memory only once.
    return model.load_backbone()


@functools.lru_cache(maxsize=None)
def get_cached_head(node_path: str):
    # Caches node-specific heads to prevent repeated disk I/O during deeper traversals.
    return model.load_head(node_path)


def _run_step(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    class_names: List[str],
    img_tensor: torch.Tensor,
    pil_image: Image.Image,
    node_name: str,
    explain_flag: bool
) -> Step:
    # Forward pass to get predictions without storing unnecessary gradients.
    with torch.no_grad():
        features = backbone(img_tensor.unsqueeze(0))
        logits = head(features)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Extract the top 3 predictions for context, safely handling nodes with fewer than 3 classes.
    top_probs, top_indices = torch.topk(probs, min(3, len(class_names)))
    top3 = [(class_names[i], float(p)) for p, i in zip(top_probs, top_indices)]

    best_label, best_conf = top3[0]
    best_idx = top_indices[0].item()

    # Generate a Grad-CAM heatmap if explainability is requested.
    heatmap = None
    if explain_flag:
        # We must clone the tensor because Grad-CAM modifies the requires_grad attribute.
        heatmap = explain.grad_cam(backbone, head, img_tensor.clone(), pil_image, best_idx)

    return Step(node_name, best_label, best_conf, top3, heatmap)


def _build_result(steps: List[Step], threshold: float) -> Result:
    # Evaluates the final outcome to categorize the prediction status.
    last_step = steps[-1]

    if last_step.confidence < threshold:
        # The model was too uncertain to continue deeper into the hierarchy.
        status = "unknown" if len(steps) == 1 else "partial"
    else:
        # The model successfully reached a leaf class where no further subdivision exists.
        status = "full"

    return Result(steps, last_step.label, status)


def predict(
    pil_image: Image.Image,
    threshold: float = config.THRESHOLD,
    explain: bool = True
) -> Union[Result, str]:
    # Ensure the root model has been trained before attempting to start the prediction loop.
    if get_cached_head("") is None:
        return "Error: Root model 'models/model.pt' does not exist. Train the root node first."

    backbone = get_cached_backbone()
    device = model.get_device()
    img_tensor = data.make_transforms(train=False)(pil_image).to(device)

    steps = []
    current_node = ""

    # Continuously predict and descend the directory tree until confidence drops or a leaf is hit.
    while True:
        head_data = get_cached_head(current_node)
        if head_data is None:
            # We reached a leaf class since there is no trained head for this subfolder.
            break

        head, class_names = head_data
        step = _run_step(backbone, head, class_names, img_tensor, pil_image, current_node, explain)
        steps.append(step)

        if step.confidence < threshold:
            # Stop early because the model isn't confident enough to make a granular guess.
            break

        # Construct the path to the next node by appending the predicted label.
        current_node = step.label if current_node == "" else f"{current_node}/{step.label}"

    return _build_result(steps, threshold)