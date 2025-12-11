# Dataset Access

The dataset for this project is hosted on HuggingFace Hub due to its large size (~2GB).

## Access Dataset

**Dataset Repository:** [https://huggingface.co/datasets/Delcastillo8/machine_learning_project](https://huggingface.co/datasets/Delcastillo8/machine_learning_project)

**Interactive Visualization:** [https://lerobot-visualize-dataset.hf.space/Delcastillo8/machine_learning_project/episode_0](https://lerobot-visualize-dataset.hf.space/Delcastillo8/machine_learning_project/episode_0)

## Download Dataset

### Option 1: Using LeRobot CLI

```bash
# Download directly using LeRobot
lerobot download-dataset \
  --repo-id Delcastillo8/machine_learning_project \
  --local-dir ./data
```

### Option 2: Using HuggingFace Hub

```python
from huggingface_hub import snapshot_download

# Download entire dataset
snapshot_download(
    repo_id="Delcastillo8/machine_learning_project",
    repo_type="dataset",
    local_dir="./data"
)
```

### Option 3: Manual Download

1. Visit [https://huggingface.co/datasets/Delcastillo8/machine_learning_project](https://huggingface.co/datasets/Delcastillo8/machine_learning_project)
2. Click "Files and versions"
3. Download required files
4. Extract to `./data` folder

## Dataset Structure

```
data/
├── episode_000.hdf5
├── episode_001.hdf5
├── ...
├── episode_049.hdf5
├── meta/
│   ├── info.json
│   └── stats.json
└── videos/
    ├── episode_000_camera_overhead.mp4
    ├── episode_000_camera_laptop.mp4
    └── ...
```

## Dataset Details

- **Total Episodes:** 50
- **Total Frames:** ~35,000
- **Frame Rate:** 30 fps
- **Image Resolution:** 640x480 (both cameras)
- **Task:** Pick and place gray toy into red bucket
- **Cameras:** Overhead view + Laptop view
- **Observations:**
  - RGB images (2 cameras)
  - Joint positions (6 DOF)
  - Joint velocities (6 DOF)
  - Gripper state (open/closed)
- **Actions:**
  - Target joint positions (6 DOF)
  - Gripper command

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{delcastillo2025so101,
  author = {Del Castillo, Pol},
  title = {SO-101 Pick-and-Place Demonstrations for Imitation Learning},
  year = {2025},
  publisher = {HuggingFace Hub},
  url = {https://huggingface.co/datasets/Delcastillo8/machine_learning_project}
}
```
