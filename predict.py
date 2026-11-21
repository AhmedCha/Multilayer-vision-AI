# Command-line interface for the hierarchical image classifier.
# Takes an image path, runs the prediction pipeline, prints the result path,
# and saves the Grad-CAM heatmaps to the reports directory.

import argparse
import sys
from pathlib import Path
from PIL import Image

import config
from hiervision import pipeline


def main() -> None:
    # Setup standard CLI arguments to allow quick terminal testing.
    parser = argparse.ArgumentParser(description="Predict image class hierarchically.")
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image {img_path} not found.")
        sys.exit(1)

    try:
        # Convert to RGB immediately to prevent issues with grayscale or RGBA inputs.
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    result = pipeline.predict(pil_img)

    # Handle the string error returned if the root model is missing.
    if isinstance(result, str):
        print(result)
        sys.exit(1)

    # Format the traversal path into a clean, readable string (e.g., animal 94% -> mammal 88%).
    path_parts = [f"{step.label} {int(step.confidence * 100)}%" for step in result.steps]
    path_str = " -> ".join(path_parts)

    print(f"Prediction: {path_str}")
    print(f"Status: {result.status}")

    # Ensure the reporting directory exists before attempting to save heatmaps.
    out_dir = config.REPORTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each generated Grad-CAM heatmap with a descriptive filename.
    for i, step in enumerate(result.steps):
        if step.heatmap:
            # Replace slashes with underscores in case the label contains nested formatting.
            safe_label = step.label.replace("/", "_")
            out_name = f"{img_path.stem}_step{i}_{safe_label}.png"
            out_path = out_dir / out_name

            step.heatmap.save(out_path)
            print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()