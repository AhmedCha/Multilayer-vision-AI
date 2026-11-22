# Hiervision
A small, easy-to-read hierarchical image classifier for beginners.
An image is classified step by step: first the big category (animal, vehicle, building, other), then the sub category, and so on. Each step has its own small classifier on top of one shared, frozen VGG16 backbone. If the confidence at a step is below the threshold (default 0.80), the system stops and returns the last confident level, or "unknown" if it fails at the root. Every step comes with a Grad-CAM heatmap that shows where the model looked. If an image contains several subjects (for example a cat and a dog), each subject is also classified on its own.
**The folders are the taxonomy.** Add a folder of images, retrain that node, and the program picks it up with no code change.
## Quick start
```bash
# 1. Setup (once)
python3.14 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
# 2. Get data (Open Images, once). Needs Docker for MongoDB, see section 3
pip install -r requirements-tools.txt
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo mongo
python tools/download_openimages.py --only vehicle
# 3. Check, train, use
python train.py --check
python train.py --all
streamlit run app.py
```
---
## Requirements
- Python 3.14 (tested with 3.14.4)
- Linux or WSL2 (developed on WSL2). Windows and macOS should work for everything except the shell commands used for tips
- An NVIDIA GPU with a recent driver (developed on an RTX 3050 8GB). CPU works but is slow, especially with Grad-CAM and multi-subject detection
- Docker, only for the Open Images download step (FiftyOne needs a MongoDB server)
- About 1 GB of disk for the model weights that PyTorch downloads on first use (see section 4)
### Tested environment
These are the versions this README was verified against on 2026-11-22, read with `python tools/check_env.py`:
| Component | Version |
|---|---|
| Python | 3.14.4 |
| System | WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2) |
| GPU | NVIDIA GeForce RTX 3050, 8 GB (sm_86) |
| PyTorch | 2.15.0, built for CUDA 13.2 (`cu132`) |
| torchvision | 0.29.2 |
| cuDNN | 9.26.0 |
| numpy | 2.5.5 |
| Pillow | 13.0.0 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| tqdm | 4.70.1 |
| grad-cam | 1.5.7 |
| streamlit | 1.67.0 |
| fiftyone | 1.22.1 |
| Docker | 29.10.0 |
| MongoDB | 9.0.8 (the latest `mongo` image) |
The exact versions are pinned in `requirements.txt` (and `requirements-tools.txt` for FiftyOne), so a fresh install gets the same libraries.
---
## 1. Setup
```bash
git clone  hiervision
cd hiervision
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# PyTorch built for CUDA 13.2 (install this BEFORE requirements.txt)
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt      # pinned: grad-cam, matplotlib, numpy, Pillow, scikit-learn, streamlit, tqdm
```
About the PyTorch line:
- The last part of the URL (`cu132`) is the CUDA version PyTorch was built with. PyTorch ships its own CUDA runtime, so you do **not** need to install the CUDA toolkit yourself.
- That CUDA version must not be newer than what your NVIDIA driver supports (`nvidia-smi` shows it as "CUDA Version" at the top right). On an older driver, use the PyTorch index that matches it.
- If pip cannot find these exact versions on your index, remove the `==` pins from that line and let pip pick the newest available.
### Check your environment
```bash
python tools/check_env.py
```
It prints your Python, driver, PyTorch and library versions, runs a small GPU test, checks that the features the code relies on exist, shows the downloaded model weights and the Docker/MongoDB status, and compares this README with your machine (it prints `MISMATCH` if the PyTorch index or the Python version written here differs from yours). The GPU line should show your card, and `fp16 autocast test: ok`.
Always run commands **from the project root** with the virtual environment active.
To refresh the pinned requirements after upgrading something:
```bash
python tools/check_env.py --write       # rewrites requirements.txt and requirements-tools.txt, keeping .bak copies
```
---
## 2. Data folder rules
`data/` is the taxonomy. The code scans it to discover the tree. The current dataset looks like this:
```
data/
├── animal/
│   ├── mammal/   cat, dog, elephant, lion, other
│   └── bird/     duck, eagle, canary, other
├── vehicle/      car, truck, bus, train, bicycle, plane, motorcycle, other
├── building/     (images directly in this folder)
└── other/        (images directly in this folder)
```
- A folder with **subfolders** is a **node**. It gets its own classifier, and its classes are its subfolders. `data/` itself is the root node.
- A folder with **images** is a **leaf class**.
- A folder never contains both images and subfolders.
- A class may be called `train` (railway trains). Only a node that has two or more folders named `train`, `val`, `valid` or `test` side by side is rejected, because that means a dataset was downloaded already split. Merge those into one folder per class.
- Folders starting with `.` or `_` are ignored.
- Training images of a node are all images under each of its child folders.
- **Splits are automatic and never stored.** Each image goes to train (80%), validation (10%) or test (10%) based on a hash of its path, so the same image has the same split in every node. There are no `train/` or `test/` folders.
- **Do not rename or move images after you start training.** That changes the hash, and so the split.
- A class called `other` in each node holds things that look like the neighbours but are not in your list (for example tigers next to lions). It is optional, and you can name it anything.
---
## 3. Get training data (Open Images V7)
Images come from Open Images V7 through FiftyOne. Only the small **validation** and **test** splits are used, because loading the train split needs far too much RAM.
### 3.1 Install FiftyOne
```bash
source .venv/bin/activate
pip install -r requirements-tools.txt
```
### 3.2 Start MongoDB (needed by FiftyOne)
```bash
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo mongo
export FIFTYONE_DATABASE_URI="mongodb://localhost:27017"
```
`127.0.0.1:` keeps the database reachable only from your own machine. Without it, Docker publishes the port on all network interfaces and MongoDB has no password. The download script sets `FIFTYONE_DATABASE_URI` itself when it is missing, so the `export` is optional. FiftyOne 1.22.1 works with the latest `mongo` image (MongoDB 9.0.8 when this was tested).
```bash
docker ps                      # is it running?
docker logs fiftyone-mongo     # logs
docker stop fiftyone-mongo     # stop it when you are done
docker start fiftyone-mongo    # start it again later (do NOT run "docker run" twice)
docker update --restart unless-stopped fiftyone-mongo    # optional: start automatically after a reboot
```
Optional variants (use them instead of the first command, on a fresh container):
```bash
# Limit RAM use (recommended on a small machine)
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo --memory 2g mongo --wiredTigerCacheSizeGB 1
# Keep the database in a named volume so removing the container does not delete it
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo -v fiftyone-mongo-data:/data/db mongo
```
If a future FiftyOne or MongoDB release makes them disagree on the version, use an older image such as `mongo:6.0` instead of `mongo` (after `docker rm -f fiftyone-mongo`).
### 3.3 Choose what to download
Open `tools/download_openimages.py` and edit the `TARGETS` list. Each entry is:
```python
(destination folder under data/, Open Images class, quota, min box area)
# ("vehicle/car", "Car", 600, 0.25)
```
Several Open Images classes can feed the same folder (for example Horse, Sheep and Fox all feed `animal/mammal/other`). A destination must be a leaf folder, never a node: the script refuses to put images in a folder that has subfolders.
**Check what exists before downloading.** The counting tool reads only the small annotation files (about 90 MB, saved in `downloads/oi_annotations/`) and downloads no photos:
```bash
python tools/count_openimages.py                          # every class in TARGETS
python tools/count_openimages.py --classes Lion Canary    # only some classes
```
It prints, per class, how many validation and test images contain it and how many pass box-size filters of 25%, 15% and 10%. Keep each quota at or below the number available. Some classes are simply small in Open Images (lion, canary, elephant), and crow and pigeon do not exist there at all.
### 3.4 Download
```bash
python tools/download_openimages.py --list-classes | head   # valid class names
python tools/download_openimages.py --dry-run               # print what would be downloaded
python tools/download_openimages.py --only vehicle/car      # a small test first
python tools/download_openimages.py --only vehicle
python tools/download_openimages.py --only animal/mammal
python tools/download_openimages.py --only animal/bird
python tools/download_openimages.py --only building
python tools/download_openimages.py --only other
```
How it behaves:
- `--only PREFIX` processes only targets whose destination folder starts with that text.
- Classes are processed one at a time, to keep RAM low.
- FiftyOne stages everything in `downloads/open-images-v7/`. Only images where the object's box covers **at least 25% of the image** are copied into `data/`, renamed with the prefix `oi_`.
- A file that already exists under `data/` is skipped. The quota counts **new images per run**, so running the same target again adds more images, up to the quota again. Use `--only` to run just what you need.
- When Open Images has fewer matching images than the quota, the script prints `found X of Y quota, use another source`.
- Keep `downloads/` between runs so FiftyOne does not download its label files again. You can delete it once `data/` is final, since `data/` holds real copies.
If your machine struggles during a download, cap it so only the script can be killed:
```bash
systemd-run --user --scope -p MemoryMax=6G nice -n 19 ionice -c3   python tools/download_openimages.py --only vehicle/car
```
### 3.5 Clean and inspect the data
```bash
find data -type d -empty -delete        # remove folders that ended up with no images
python tools/find_duplicate.py          # byte-identical images (same photo in train and test inflates accuracy)
python tools/check_splits.py            # train/val/test counts per class
python train.py --check                 # structure check, corrupt images, class sizes
tree data -L 3                          # optional: look at the structure
```
`tools/check_splits.py` should show about 80% / 10% / 10% for each class, with normal variation on small classes. A class flagged "too few for val/test" has fewer than 30 images in the validation or test split, so its accuracy numbers will be unreliable.
**Look at the images yourself.** Open about 20 random files in each `other` folder and in the small classes (canary, elephant, lion) before training. Open Images boxes are not perfect, and one bad folder costs more than a small one.
### 3.6 Adding images from other sources
Datasets from elsewhere (Kaggle, your own photos, and so on) are added by copying them into the right leaf folder, for example `data/animal/bird/pigeon/`. Follow the folder rules in section 2: merge any `train/valid/test` subfolders into one folder per class, and give files unique names so nothing is overwritten. Then run `tools/find_duplicate.py` and `python train.py --check` again.
---
## 4. Train
Training uses a frozen VGG16 (so only a small head per node is trained), automatic mixed precision, class-weighted loss for unbalanced folders, and a progress bar per epoch. The first run downloads the VGG16 weights (528 MB) into `~/.cache/torch/hub/checkpoints`.
```bash
source .venv/bin/activate
python train.py --node root              # the root node (animal, vehicle, building, other)
python train.py --node vehicle
python train.py --node animal
python train.py --node animal/mammal
python train.py --node animal/bird
python train.py --all                    # or train every node found in data/
```
Node paths mirror your folders. The root node is `root`.
What happens for each node:
- Nodes with fewer than 2 classes are skipped. Nodes where any class has fewer than 10 training images are also skipped, with a message listing the class.
- After each epoch you see the training loss, validation accuracy and the epoch time. The head with the best validation accuracy is kept.
- At the end it evaluates on the test split and prints accuracy and a per-class report.
Outputs:
- `models/model.pt` (root), `models/animal/model.pt`, `models/animal/mammal/model.pt`, and so on. Each file holds the head weights and the class names (a few tens of KB).
- A confusion matrix per node in `reports//confusion_matrix.png` (for the root node, `reports/confusion_matrix.png`). Read it, together with the per-class scores printed at the end of training, to see which classes get confused.
If a split has fewer than about 30 images per class, the numbers mean little. They only prove that the pipeline runs.
To log training to a file, set `SHOW_PROGRESS = False` in `config.py` so the file does not fill with progress-bar updates.
---
## 5. Predict from the command line
```bash
python predict.py path/to/image.jpg
```
Example of the output format:
```
--- Whole Image ---
Prediction: animal 99% -> mammal 100%
Rejected guess: dog 59%
Status: partial
--- Subjects ---
Subject 1 Box (x1, y1, x2, y2):  ()
Subject 2 Box (x1, y1, x2, y2):  ()
```
- **Prediction** is the path of accepted steps, each with its confidence.
- **Rejected guess** is the last step that fell below the threshold. It is shown for information but not used.
- **Status**: `full` means a leaf class was reached, `partial` means it stopped below the root because confidence fell under the threshold, `unknown` means it was not confident even at the root.
- **Subjects** appears only when the detector found two or more subjects (see section 7).
Grad-CAM heatmaps for every step are saved in `reports/predictions/`, named like `photo_whole_step0_animal.png` and `photo_subj1_step1_cat.png`.
---
## 6. Run the app
```bash
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```
Open http://localhost:8501 (on WSL2, open it in your Windows browser). Without `--server.address 127.0.0.1`, Streamlit listens on every network interface, and the app has no login.
In the app:
- **Sidebar:** a **Reload models** button (press it after training new nodes), the **confidence threshold** slider, and a **Detect multiple subjects** checkbox.
- Upload an image (jpg, jpeg, png or webp) and press **Predict**. Phone photos are rotated correctly using their EXIF data.
- You get a **Whole image** tab with one column per step of the cascade: the label, the confidence, the Grad-CAM heatmap and the top 3 probabilities. A step below the threshold is marked "Below threshold, not used".
- If several subjects were found, the uploaded image is shown with numbered boxes and each subject gets its own tab. Subjects whose result is `unknown`, or whose final label is listed in `HIDE_LABELS`, are collected in an expander ("N detections not shown").
- The slider does not re-run the model. If you move it after predicting, the app tells you to press **Predict** again.
If the app runs on a remote server, use an SSH tunnel:
```bash
# on the server
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
# on your own computer
ssh -L 8501:localhost:8501 user@server
# then open http://localhost:8501
```
---
## 7. Several subjects in one image
Each node uses softmax, so it can only give one answer per image. With a cat and a dog in the same photo, the dog and cat scores compete (for example 60% and 39%), and neither passes the 80% threshold.
To handle this, a small pretrained object detector (Faster R-CNN with a MobileNetV3 backbone, trained on COCO, included in torchvision) proposes boxes:
- The detector is used **only to find boxes**. Its own class names are ignored. Your classifier decides what each crop is.
- The **whole image is always classified** first, as before.
- If the detector finds **two or more** subjects, each box (padded by 10%) is cropped and run through the same cascade, with its own heatmaps.
- The detector weights (74 MB) are downloaded automatically the first time it runs.
Settings in `config.py` control it (see the table below). Limits to know about:
- COCO has about 80 object types (people, vehicles, many animals). It will not find buildings, and it may miss or mislabel animals outside its list. The whole-image result covers those cases.
- It also finds people, chairs and other things. Those usually end up as `other` and are hidden by `HIDE_LABELS`.
- Each subject runs a full cascade plus Grad-CAM, so keep `MAX_SUBJECTS` small, especially on CPU.
- A wide or tall crop is resized and center-cropped to 224 x 224, so its edges can be cut off.
- Turn it off with the sidebar checkbox or by setting `DETECT_SUBJECTS = False`.
---
## 8. Reading the Grad-CAM heatmaps
The heatmap shows which regions pushed the model toward the predicted class. It uses VGG16's last convolutional block (the ReLU after `conv5_3`, `features[29]`) and is drawn on the 224 x 224 center crop that the model actually saw.
- **Colors are relative.** The strongest region in each image is red and the weakest blue, so red does not mean a high absolute score.
- **Maps are coarse.** The last conv layer is a 14 x 14 grid, so heatmaps are blobs, never outlines.
- **Leaf-level maps are the most informative** (dog, car, duck), because one object decides the answer. Broad nodes such as `root` and `animal` can be decided from fur, eyes and textures spread over the whole image, so their maps often look scattered. That is normal.
- Use heatmaps to catch shortcuts, for example a "lion" prediction lit up on the background instead of the animal.
---
## 9. Add a new class or branch (no code changes)
```bash
# New class under an existing node
mkdir data/animal/mammal/horse        # add images (300+ recommended)
python train.py --check
python train.py --node animal/mammal
# New branch under an existing node (e.g. data/animal/reptile/snake, .../lizard)
python train.py --check
python train.py --node animal/reptile     # the new node
python train.py --node animal             # its parent, which has a new class
# New top-level category
python train.py --node root
```
Prediction discovers new models automatically, because it looks for `models//model.pt` while descending. Press **Reload models** in the app after training.
---
## Settings (`config.py`)
| Setting | Default | Meaning |
|---|---|---|
| `IMG_SIZE` | 224 | Input size of the model |
| `BATCH_SIZE` | 32 | Lower it (for example 16) if the GPU runs out of memory |
| `EPOCHS` | 10 | Training epochs per node |
| `LR` | 1e-3 | Learning rate of the head |
| `THRESHOLD` | 0.80 | Minimum confidence to accept a step |
| `SEED` | 42 | Random seed |
| `NUM_WORKERS` | 4 | Data loading processes. Lower it if RAM runs out |
| `SHOW_PROGRESS` | True | Show progress bars while training |
| `DETECT_SUBJECTS` | True | Classify each detected subject as well as the whole image |
| `DETECTION_MIN_SCORE` | 0.5 | Minimum detector confidence for a box |
| `MIN_SUBJECT_AREA` | 0.05 | Ignore boxes smaller than this fraction of the image |
| `MAX_SUBJECTS` | 5 | Maximum number of boxes analyzed |
| `HIDE_LABELS` | `["other"]` | Final labels not shown as separate subject tabs |
---
## Project layout
```
hiervision/
├── README.md
├── PROJECT.md             # the project rules, attached to every LLM prompt
├── requirements.txt       # pinned runtime libraries (torch is installed separately)
├── requirements-tools.txt # pinned FiftyOne, only for the download tools
├── config.py              # all settings in one place
├── train.py               # training and data check command line
├── predict.py             # prediction command line
├── app.py                 # Streamlit app
├── hiervision/            # the code
│   ├── data.py            # scan folders, hash split, dataset, transforms, checks
│   ├── model.py           # VGG16 backbone, heads, save/load
│   ├── trainer.py         # train one node, report results
│   ├── explain.py         # Grad-CAM
│   ├── detect.py          # finds several subjects in one image
│   └── pipeline.py        # the cascade (predict, predict_subjects)
├── tools/
│   ├── check_env.py             # versions, GPU test, README check, pinned requirements
│   ├── download_openimages.py   # download and filter Open Images
│   ├── count_openimages.py      # how many images exist per class
│   ├── check_splits.py          # train/val/test counts per class
│   └── find_duplicate.py        # byte-identical images
├── data/                  # your images (not in git)
├── models/                # trained heads (not in git)
├── downloads/             # FiftyOne cache and annotation files (not in git)
└── reports/               # confusion matrices and heatmaps (not in git)
```
---
## Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` in `config.py` and close other GPU programs |
| Training is slow or RAM runs out | Lower `NUM_WORKERS` in `config.py` |
| `ModuleNotFoundError` | Activate `.venv` and run from the project root |
| pip cannot find `torch==2.15.0` | Your PyTorch index may differ from the one in this README. Remove the `==` pins from the torch line, or pick the index that matches your driver. `python tools/check_env.py` shows what is installed |
| `check_env.py` prints `MISMATCH` for the README | The PyTorch index or Python version written in this README differs from your machine. Edit those lines to match |
| An install fails on a very new Python | Create the venv with Python 3.12 and use the same commands |
| "Root model does not exist" | Train the root first: `python train.py --node root` |
| The app does not see a newly trained node | Press **Reload models** in the sidebar |
| "Folder mixes images and subfolders" | Move the loose images into a class folder, or remove the extra subfolders |
| "looks like a dataset that is already split" | A node has two or more folders named train, val, valid or test. Merge them into one folder per class |
| A node is skipped for "< 10 train images" | Add images to that class or delete its empty folder (`find data -type d -empty -delete`) |
| FiftyOne cannot connect to MongoDB | `docker ps`, then `docker start fiftyone-mongo` |
| `Crow` or `Pigeon` "is not a valid Open Images class" | They are not in Open Images. Add those images from another source (section 3.6) |
| `found X of Y quota, use another source` | Open Images has fewer matching images. Use another source or accept a smaller class |
| Download uses all RAM or disk | Use the capped command in 3.4, the `--memory 2g` MongoDB variant, and only the validation and test splits |
| "Detection failed" warning | Check your internet connection (the weights download once). Prediction still works on the whole image |
| Heatmaps look scattered | Normal for broad nodes such as `root` and `animal` (section 8) |
| Results look worse after renaming files | Renaming changes the hash split, so the test set changed. Avoid renaming after training starts |
---
## Known limitations
- **One label per image or crop.** The classifiers use softmax. Several subjects are handled by cropping them with the detector, not by multi-label heads.
- **The detector only knows COCO objects.** It will not box buildings and may miss some animals. The whole-image classification always runs as a fallback.
- **The backbone is frozen.** This is fast and light, and it works well for clearly different classes, but less well for similar ones. If a node scores poorly, add more and cleaner images first. Unfreezing the last convolutional block is a later upgrade.
- **Open Images validation and test only.** Some classes are small (lion, canary, elephant), and crow and pigeon are missing. Small classes get unreliable test numbers.
- **The `other` classes are limited.** Open Images has objects, not landscapes, so photos of scenery or random objects may be forced into a known class. Add more varied `other` images if that happens.
- **The download filter is 25% for every class.** The fourth value in each `TARGETS` entry is not used yet.