# Hiervision

A small, easy-to-read hierarchical image classifier for beginners.

An image is classified step by step: first the big category (for example animal, vehicle, building), then the sub category, and so on. Each step has its own small classifier on top of one shared, frozen VGG16 backbone. If the confidence at a step is below the threshold (default 0.80), the system stops and returns the last confident level, or "unknown" if it fails at the root. Every step comes with a Grad-CAM heatmap that shows where the model looked.

**The folders are the taxonomy.** Add a folder of images, retrain that node, and the program picks it up with no code change.

---

## Requirements

- Python 3.14
- An NVIDIA GPU with a recent driver (developed on an RTX 3050 8GB). CPU works but is slow
- Docker, only for the Open Images download step (FiftyOne needs a MongoDB server)
- A free Kaggle account, only if you use the Kaggle datasets

---

## 1. Setup

```bash
git clone <your-repo-url> hiervision
cd hiervision

python3.14 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip

# PyTorch with CUDA 12.8 wheels (install this BEFORE requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Check that everything works:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import pytorch_grad_cam, streamlit; print('imports ok')"
```

The first command should print `True` and your GPU name.

Always run commands **from the project root** with the virtual environment active.

---

## 2. Data folder rules

`data/` is the taxonomy. The code scans it to discover the tree.

```
data/
├── animal/
│   ├── mammal/
│   │   ├── cat/*.jpg
│   │   ├── dog/*.jpg
│   │   └── other/*.jpg
│   └── bird/
│       ├── duck/*.jpg
│       └── crow/*.jpg
├── vehicle/
│   ├── car/*.jpg
│   └── bus/*.jpg
├── building/*.jpg
└── other/*.jpg
```

- A folder with **subfolders** is a **node**. It gets its own classifier, and its classes are its subfolders. `data/` itself is the root node.
- A folder with **images** is a **leaf class**.
- A folder never contains both images and subfolders.
- Class folders must not be named `train`, `val`, `valid` or `test`. Merge datasets that come with those subfolders into one folder per class.
- Folders starting with `.` or `_` are ignored.
- Training images of a node are all images under each of its child folders.
- **Splits are automatic.** Each image goes to train (80%), validation (10%) or test (10%) based on a hash of its path, so the same image has the same split in every node.
- **Do not rename or move images after you start training.** That changes the hash, and so the split.

---

## 3. Get training data

### 3.1 Data environment

```bash
python3.12 -m venv .venv-data
source .venv-data/bin/activate
pip install fiftyone kaggle pillow
```

### 3.2 Start MongoDB (needed by FiftyOne)

```bash
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo mongo
export FIFTYONE_DATABASE_URI="mongodb://localhost:27017"
```

`127.0.0.1:` keeps the database reachable only from your own machine. Without it, Docker publishes the port on all network interfaces and MongoDB has no password.

Useful commands:

```bash
docker ps                      # is it running?
docker logs fiftyone-mongo     # logs
docker stop fiftyone-mongo     # stop it when you are done
docker start fiftyone-mongo    # start it again later (do NOT run "docker run" twice)
```

The `export` only lasts for the current terminal. To make it permanent:

```bash
echo 'export FIFTYONE_DATABASE_URI="mongodb://localhost:27017"' >> ~/.bashrc
```

Optional variants (use them instead of the first command, on a fresh container):

```bash
# Limit RAM use (recommended on a small machine)
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo --memory 2g mongo --wiredTigerCacheSizeGB 1

# Keep the database in a named volume so removing the container does not delete it
docker run -d -p 127.0.0.1:27017:27017 --name fiftyone-mongo -v fiftyone-mongo-data:/data/db mongo
```

If FiftyOne complains about the MongoDB version, replace `mongo` with `mongo:6.0` (after `docker rm -f fiftyone-mongo`).

### 3.3 Open Images (vehicles, mammals, buildings, and more)

Edit the `TARGETS` list at the top of `tools/download_openimages.py` to choose classes and quantities, then:

```bash
source .venv-data/bin/activate
export FIFTYONE_DATABASE_URI="mongodb://localhost:27017"

python tools/download_openimages.py --list-classes | head     # valid class names
python tools/download_openimages.py --dry-run                 # see what would be downloaded

# First real run: a small part, with a memory and I/O cap so the machine cannot freeze
systemd-run --user --scope -p MemoryMax=6G nice -n 19 ionice -c3 \
  python tools/download_openimages.py --only vehicle/car

# Then branch by branch
python tools/download_openimages.py --only vehicle
python tools/download_openimages.py --only animal
python tools/download_openimages.py --only building
python tools/download_openimages.py --only other
```

Images are staged in `downloads/` and only the filtered ones (object covers at least 25% of the image) are copied into `data/`. Keep `downloads/` between runs so FiftyOne does not download its label files again.

### 3.4 Kaggle datasets (birds, mammals, scenes)

Create an API token on your Kaggle account page and save it as `kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d gpiosenka/100-bird-species -p downloads/kaggle/birds --unzip
kaggle datasets download -d alessiocorrado99/animals10 -p downloads/kaggle/animals10 --unzip
kaggle datasets download -d puneet6060/intel-image-classification -p downloads/kaggle/intel --unzip

# Copy the right folders into data/ (rules are listed at the top of the script)
python tools/import_folders.py
```

Check each dataset slug on its Kaggle page if a download fails.

### 3.5 Classes you must collect yourself

Some classes are not covered well by these datasets (for example common pigeons and canaries). Put your own images directly in the leaf folder, for example `data/animal/bird/pigeon/`.

### 3.6 Check the data

```bash
deactivate
source .venv/bin/activate

python train.py --check
tree data -L 3                 # optional: look at the structure
```

This prints every node with its classes and image counts, and stops with a clear error if a folder mixes images and subfolders or if images are corrupt. Aim for **300 to 500+ images per leaf class**.

---

## 4. Train

Training uses automatic mixed precision (AMP) and a frozen VGG16, so only a small head per node is trained. Each node takes minutes on an 8 GB GPU.

```bash
source .venv/bin/activate

python train.py --node root              # root node first
python train.py --node vehicle
python train.py --node animal
python train.py --node animal/mammal
python train.py --node animal/bird

python train.py --all                    # or train every node found in data/
```

Node paths mirror your folders. The root node is `root`.

Outputs:

- `models/model.pt` (root), `models/animal/model.pt`, `models/animal/mammal/model.pt`, and so on. Each file holds the head weights and the class names.
- `reports/<node>/confusion_matrix.png`. Read it, and the per-class F1 scores printed at the end of training, to see which classes get confused.

---

## 5. Predict

```bash
python predict.py path/to/image.jpg
```

Prints the path the image took, for example `animal 94% -> mammal 88% -> dog 91%`, plus the status:

- `full`: reached a leaf class
- `partial`: stopped below the root because confidence fell under the threshold
- `unknown`: not confident even at the root

Grad-CAM heatmaps for each step are saved in `reports/predictions/`.

---

## 6. Run the app

```bash
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501). Upload an image, adjust the threshold, and press Predict. Each step shows its Grad-CAM heatmap and top 3 probabilities. After training new nodes, press **Reload models**.

If the app runs on a remote server, keep it private and use an SSH tunnel:

```bash
# on the server
streamlit run app.py --server.address 127.0.0.1 --server.port 8501

# on your own computer
ssh -L 8501:localhost:8501 user@server
# then open http://localhost:8501
```

---

## 7. Add a new class or branch (no code changes)

```bash
# New class under an existing node
mkdir data/animal/mammal/horse        # add images (300+)
python train.py --check
python train.py --node animal/mammal

# New branch under an existing node (e.g. data/animal/reptile/snake, .../lizard)
python train.py --check
python train.py --node animal/reptile     # the new node
python train.py --node animal             # its parent, which has a new class

# New top-level category
python train.py --node root
```

Prediction discovers new models automatically, because it looks for `models/<node>/model.pt` while descending.

---

## Project layout

```
hiervision/
├── README.md
├── requirements.txt
├── config.py              # all settings in one place
├── train.py               # training command line
├── predict.py             # prediction command line
├── app.py                 # Streamlit app
├── hiervision/            # the code
│   ├── data.py            # scan folders, split, dataset, transforms
│   ├── model.py           # VGG16 backbone, heads, save/load
│   ├── trainer.py         # train one node, report results
│   ├── explain.py         # Grad-CAM
│   └── pipeline.py        # the cascade (predict)
├── tools/                 # dataset helpers (run in .venv-data)
├── data/                  # your images (not in git)
├── models/                # trained heads (not in git)
├── downloads/             # staging area for downloads (not in git)
└── reports/               # confusion matrices and heatmaps (not in git)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` in `config.py` (for example 32 to 16) and close other GPU programs |
| Training is slow or RAM runs out | Lower `NUM_WORKERS` in `config.py` |
| `ModuleNotFoundError` | Activate the right environment (`.venv`) and run from the project root |
| torch will not install on Python 3.14 | Use Python 3.12: `python3.12 -m venv .venv`, then the same install commands |
| "Root model does not exist" | Train the root first: `python train.py --node root` |
| "Folder mixes images and subfolders" | Move the loose images into a class folder, or remove the extra subfolders |
| A class named `train` or `val` appears | A dataset came with its own split folders. Merge them into one folder per class |
| FiftyOne cannot connect to MongoDB | `docker ps`, then `docker start fiftyone-mongo`, and check that `FIFTYONE_DATABASE_URI` is set in this terminal |
| Download uses all RAM or disk | Use the capped command from 3.3, the `--memory 2g` MongoDB variant, and only the validation and test splits |
| Results look worse after renaming files | Renaming changes the hash split, so the test set changed. Avoid renaming after training starts |

---

## Notes on accuracy

The backbone is frozen, which keeps training fast and light. This works well for clearly different classes (animal vs vehicle, cat vs elephant) and less well for very similar ones (crow vs pigeon). If a node scores poorly, add more and cleaner images first. Unfreezing the last convolutional block is a later upgrade.