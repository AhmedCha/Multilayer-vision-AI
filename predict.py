# Command-line interface for the hierarchical image classifier.
# Takes an image path, runs the prediction pipeline (including multi-subject),
# prints the results, and saves the Grad-CAM heatmaps to the reports directory.

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageOps

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
        # Handle EXIF rotation and convert to RGB.
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    try:
        subjects = pipeline.predict_subjects(pil_img, with_heatmaps=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Retrieve and print the whole-image primary result first.
    whole_result = subjects[0].result
    accepted_steps = [step for step in whole_result.steps if step.accepted]
    path_parts = [f"{step.label} {int(step.confidence * 100)}%" for step in accepted_steps]
    path_str = " -> ".join(path_parts) if path_parts else "None"

    print("--- Whole Image ---")
    print(f"Prediction: {path_str}")
    if whole_result.steps and not whole_result.steps[-1].accepted:
        rejected = whole_result.steps[-1]
        print(f"Rejected guess: {rejected.label} {int(rejected.confidence * 100)}%")
    print(f"Status: {whole_result.status}")
    print()

    # Print a quick summary line per cropped subject.
    if len(subjects) > 1:
        print("--- Subjects ---")
        for i, subj in enumerate(subjects[1:], start=1):
            print(f"Subject {i} Box {subj.box}: {subj.result.final_label} ({subj.result.status})")
        print()

    # Ensure the reporting directory exists before attempting to save heatmaps.
    out_dir = config.REPORTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each generated Grad-CAM heatmap with a descriptive filename.
    for subj_idx, subj in enumerate(subjects):
        prefix = f"{img_path.stem}_whole" if subj_idx == 0 else f"{img_path.stem}_subj{subj_idx}"

        for i, step in enumerate(subj.result.steps):
            if step.heatmap:
                safe_label = step.label.replace("/", "_")
                out_name = f"{prefix}_step{i}_{safe_label}.png"
                out_path = out_dir / out_name

                step.heatmap.save(out_path)
                print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()