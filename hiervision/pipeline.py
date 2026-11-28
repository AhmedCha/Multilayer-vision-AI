# Executes the hierarchical inference pipeline using the cached backbone and heads.
# Walks down the taxonomy tree step by step, applying Grad-CAM explainability,
# and supports evaluating cropped multi-subject detections.

import functools
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

import config
from hiervision import data, explain, model, detect


@dataclass
class Step:
    node: str
    label: str
    confidence: float
    top3: List[Tuple[str, float]]
    all_probs: List[Tuple[str, float]]
    margin: float
    heatmap: Union[Image.Image, None]
    accepted: bool


@dataclass
class Result:
    steps: List[Step]
    final_label: str
    status: str


@dataclass
class Subject:
    box: Union[Tuple[int, int, int, int], None]
    result: Result


@functools.lru_cache(maxsize=1)
def get_cached_backbone() -> torch.nn.Module:
    # Caches the shared, frozen VGG16 backbone so it is loaded into memory only once.
    return model.load_backbone()


_head_cache = {}


def get_cached_head(node_path: str):
    # Uses a dict cache that ignores missing models so newly trained nodes can be found.
    if node_path not in _head_cache:
        head_data = model.load_head(node_path)
        if head_data is not None:
            _head_cache[node_path] = head_data
    return _head_cache.get(node_path)


def reload_models() -> None:
    # Clears the dictionary cache so new models can be loaded.
    _head_cache.clear()


def _run_step(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    class_names: List[str],
    img_tensor: torch.Tensor,
    features: torch.Tensor,
    pil_image: Image.Image,
    node_name: str,
    with_heatmaps: bool,
) -> Step:
    # Forward pass uses precomputed backbone features to save time.
    with torch.no_grad():
        logits = head(features)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Sort all probabilities from high to low so the UI can show full distributions and margins.
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    all_probs = [(class_names[i], float(p)) for p, i in zip(sorted_probs, sorted_indices)]
    top3 = all_probs[:3]

    # Compute the lead over the second choice (1.0 when only one class exists).
    margin = all_probs[0][1] - all_probs[1][1] if len(all_probs) >= 2 else 1.0

    best_label, best_conf = all_probs[0]
    best_idx = int(sorted_indices[0].item())

    # Generate a Grad-CAM heatmap if explainability is requested.
    heatmap = None
    if with_heatmaps:
        # Clone the tensor because Grad-CAM modifies the requires_grad attribute.
        heatmap = explain.grad_cam(backbone, head, img_tensor.clone(), pil_image, best_idx)

    # 'accepted' is evaluated and assigned in the main predict loop.
    return Step(node_name, best_label, best_conf, top3, all_probs, margin, heatmap, accepted=False)


def _build_result(steps: List[Step]) -> Result:
    # Evaluates the final outcome to categorize the prediction status.
    accepted_steps = [s for s in steps if s.accepted]

    if not accepted_steps:
        final_label = "unknown"
        status = "unknown"
    else:
        final_label = accepted_steps[-1].label
        # If the last step wasn't accepted, we stopped below root but didn't finish.
        status = "partial" if not steps[-1].accepted else "full"

    return Result(steps, final_label, status)


def predict(
    pil_image: Image.Image,
    threshold: float = config.THRESHOLD,
    with_heatmaps: bool = True,
) -> Result:
    # Ensure the root model has been trained before attempting to start the prediction loop.
    if get_cached_head("") is None:
        raise FileNotFoundError("Error: Root model 'models/model.pt' does not exist. Train the root node first.")

    backbone = get_cached_backbone()
    device = model.get_device()
    img_tensor = data.make_transforms(train=False)(pil_image).to(device)

    # Run the backbone ONCE per image and reuse the features.
    with torch.no_grad():
        features = backbone(img_tensor.unsqueeze(0))

    steps = []
    current_node = ""

    # Continuously predict and descend the directory tree until confidence drops or a leaf is hit.
    while True:
        head_data = get_cached_head(current_node)
        if head_data is None:
            # Reached a leaf class since there is no trained head for this subfolder.
            break

        head, class_names = head_data
        step = _run_step(backbone, head, class_names, img_tensor, features, pil_image, current_node, with_heatmaps)

        step.accepted = step.confidence >= threshold
        steps.append(step)

        if not step.accepted:
            # Stop early because the model isn't confident enough to make a granular guess.
            break

        # Construct the path to the next node by appending the predicted label.
        current_node = step.label if current_node == "" else f"{current_node}/{step.label}"

    return _build_result(steps)


def predict_subjects(
    pil_image: Image.Image,
    threshold: float = config.THRESHOLD,
    with_heatmaps: bool = True,
) -> List[Subject]:
    # Analyzes the whole image first, and if multiple subjects are found, evaluates them individually.
    whole_result = predict(pil_image, threshold, with_heatmaps)
    subjects = [Subject(box=None, result=whole_result)]

    if config.DETECT_SUBJECTS:
        boxes = detect.find_subjects(pil_image)
        # Only bother cropping if there are at least two distinct objects.
        if len(boxes) >= 2:
            for box in boxes:
                crop = pil_image.crop(box)
                crop_result = predict(crop, threshold, with_heatmaps)
                subjects.append(Subject(box=box, result=crop_result))

    return subjects