# Uses a pretrained object detector to find multiple subjects in an image.
# Ignores the detector's class labels and only returns padded bounding boxes.
# Provides a safe fallback so prediction never breaks if detection fails.

import functools
import torch
import torchvision
from PIL import Image

import config
from hiervision import model

@functools.lru_cache(maxsize=1)
def load_detector() -> torch.nn.Module:
    # Load a lightweight Faster R-CNN pretrained on COCO to propose generic subject boxes.
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        box_score_thresh=config.DETECTION_MIN_SCORE
    )
    detector.eval()
    detector = detector.to(model.get_device())
    return detector

def find_subjects(pil_image: Image.Image) -> list[tuple[int, int, int, int]]:
    # Extracts coordinate boxes for subjects large enough to be relevant.
    try:
        detector = load_detector()
        device = model.get_device()

        # Convert PIL to tensor (C, H, W) in [0, 1] range.
        img_tensor = torchvision.transforms.functional.to_tensor(pil_image).to(device)

        with torch.no_grad():
            predictions = detector([img_tensor])[0]

        boxes = predictions['boxes']
        scores = predictions['scores']

        img_width, img_height = pil_image.size
        img_area = img_width * img_height

        # Filter out tiny detections based on the configured area fraction.
        valid_boxes = []
        valid_scores = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area.item() / img_area >= config.MIN_SUBJECT_AREA:
                valid_boxes.append(boxes[i])
                valid_scores.append(scores[i])

        if not valid_boxes:
            return []

        boxes = torch.stack(valid_boxes)
        scores = torch.stack(valid_scores)

        # Remove duplicate boxes using Non-Maximum Suppression (IoU 0.7) across all objects.
        keep_idx = torchvision.ops.nms(boxes, scores, 0.7)
        boxes = boxes[keep_idx]

        # Keep only the highest scoring boxes up to the maximum limit.
        boxes = boxes[:config.MAX_SUBJECTS]

        final_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            # Pad the box by 10% on all sides so the classifier has some surrounding context.
            width, height = x2 - x1, y2 - y1
            pad_x, pad_y = width * 0.10, height * 0.10

            # Ensure padded coordinates don't exceed image boundaries.
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(img_width, int(x2 + pad_x))
            y2 = min(img_height, int(y2 + pad_y))

            final_boxes.append((x1, y1, x2, y2))

        return final_boxes

    except Exception as e:
        # Failsafe so broken detection never crashes the entire pipeline.
        print(f"Warning: Detection failed ({e})")
        return []