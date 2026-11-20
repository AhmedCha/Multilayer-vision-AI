# Hiervision

Hiervision is a small, easy-to-read hierarchical image classifier built for beginners. It uses a shared, frozen VGG16 backbone with custom linear heads trained locally for each node in your taxonomy tree.

## Taxonomy & Folder Rules

Hiervision treats your data folder structure directly as the taxonomy tree:
- **Root Node:** The `data/` directory acts as the root node (`""` or `"root"`).
- **Node vs. Leaf:**
  - A folder containing **subfolders** is a **Node** (it gets its own classifier head; its classes are its immediate subfolders).
  - A folder containing **images directly** is a **Leaf** class.
  - A folder never contains both images and subfolders.
- **Data Splitting:** Splits are computed deterministically via a hash of each image's relative path (80% train, 10% validation, 10% test). The same image always belongs to the same split across all nodes to prevent data leakage.

---

## Commands

### 1. Build Data Folders
Create your hierarchical folder structure under `data/`:
```bash
mkdir -p data/animal/mammal data/animal/reptile data/vehicle/car
# Place your training images inside the leaf folders (e.g., data/animal/mammal/dog/)

```

### 2. Train

Train the classifier heads across all taxonomy nodes using automatic mixed precision (AMP) for the RTX 3050:

```bash
python -m hiervision.train

```

### 3. Predict

Run inference on a single image to traverse the taxonomy tree and generate Grad-CAM explainability reports:

```bash
python -m hiervision.predict --image path/to/image.jpg

```

### 4. Run App

Launch the interactive Streamlit dashboard for browsing classifications and visual explanations:

```bash
streamlit run hiervision/app.py

```