# Generates Grad-CAM explainability heatmaps for the hierarchical classifier.
# Wraps the backbone and head together to compute gradients for the predicted class,
# overlaying them on the original image to show what the model looked at.

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import config

def grad_cam(
    backbone: nn.Module,
    head: nn.Module,
    image_tensor: torch.Tensor,
    pil_image: Image.Image,
    class_index: int
) -> Image.Image:
    # We wrap the frozen feature extractor and the trained classification head into a single sequence.
    # This allows Grad-CAM to do a full forward and backward pass from pixels to logits.
    full_model = nn.Sequential(backbone, head)
    full_model.eval()

    # VGG16's last convolutional layer is located at index 28 in the features sequential block.
    # We target this layer because it contains the richest spatial features before pooling.
    target_layers = [backbone.features[29]]

    # WHY: Because the backbone is frozen to save memory, none of its weights require gradients.
    # The autograd engine will skip gradient tracking entirely unless we explicitly tell it
    # that our input image tensor requires gradients. This forces gradients to flow back to the target layer.
    image_tensor.requires_grad = True

    # Build the target lists and unsqueeze the input tensor for the expected (N, C, H, W) PyTorch format.
    targets = [ClassifierOutputTarget(class_index)]
    input_tensor = image_tensor.unsqueeze(0)

    # Use Grad-CAM within a context block to ensure internal forward/backward hooks are correctly released after execution.
    with GradCAM(model=full_model, target_layers=target_layers) as cam:
        # Compute the activation map. Grad-CAM returns a numpy array for the batch; we take the first item.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Recreate the exact center-crop eval transform applied to the raw image tensor
    # so the visualization overlay aligns perfectly with the model's exact view.
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE)
    ])
    processed_pil = display_transform(pil_image)

    # Convert the matching, cropped PIL image to a normalized float32 numpy array (0.0 to 1.0)
    rgb_img = np.array(processed_pil.convert("RGB"), dtype=np.float32) / 255.0

    # Overlay the heatmap on the identically-sized original crop. use_rgb=True ensures the output is RGB.
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert the resulting numpy array back into a PIL Image to return it.
    return Image.fromarray(visualization)