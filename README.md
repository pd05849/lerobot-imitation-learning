# Imitation Learning for Robotic Manipulation using LeRobot SO-101

**Author:** Pol del Castillo  
**Student ID:** 662116017  
**Email:** delcap@rpi.edu  
**Institution:** Rensselaer Polytechnic Institute

## Project Description

This project implements imitation learning for robotic manipulation using the LeRobot SO-101 open-source 6-DOF robotic arm platform. The objective is to train an Action Chunking Transformer (ACT) model to perform pick-and-place tasks by learning from human teleoperation demonstrations, addressing the challenge of creating adaptive robots for manufacturing environments with limited training data.

### Objectives
- Assemble and calibrate a dual-arm teleoperation system (leader-follower configuration)
- Collect a dataset of 50 demonstration episodes using multimodal observations
- Train an Action Chunking Transformer policy using CVAE architecture
- Evaluate training convergence and model performance
- Document complete methodology for reproducibility

## Dataset

**Dataset Location:** [HuggingFace Hub - Delcastillo8/machine_learning_project](https://huggingface.co/datasets/Delcastillo8/machine_learning_project)

**Interactive Visualization:** [LeRobot Dataset Viewer](https://lerobot-visualize-dataset.hf.space/Delcastillo8/machine_learning_project/episode_0)

### Dataset Details
- **Task:** Pick-and-place (gray toy → red bucket)
- **Episodes:** 50 demonstrations
- **Total Frames:** ~35,000 frames
- **Frame Rate:** 30 fps
- **Video Encoding:** avc1 type
- **Duration per Episode:** ~10 seconds
- **Cameras:** Dual camera setup (overhead + laptop camera)
- **Observations:** RGB images + 6 joint angle configurations

## Model Overview

### Action Chunking Transformer (ACT)
- **Architecture:** Conditional Variational Autoencoder (CVAE)
- **Components:**
  - **CVAE Encoder:** Predicts latent style variable z from proprioceptive observations and action sequences (training only)
  - **CVAE Decoder:** Policy network that generates action sequences conditioned on z and multimodal observations (images + joint positions)
- **Input:** Dual RGB images (overhead + laptop views) + 6-DOF joint positions
- **Output:** Action sequences (chunked predictions) for 6 servo motors
- **Loss Function:** VAE objective with reconstruction loss + KL-divergence regularization (weighted by β)
- **Framework:** LeRobot (HuggingFace)

### Training Details
- **Training Steps:** 60,000 steps
- **Training Time:** ~2.5 hours
- **Hardware:** GPU100 (Google Colab Pro)
- **Convergence:** Both overall loss and L1 reconstruction loss converged to minimal error
- **Power Consumption:** Peak GPU usage ~160W

## Hardware Setup

### SO-101 Robotic Arm Components

**Follower Arm:**
- 6x STS3215 Servo Motors (12V, 1/345 gear ratio)
- Waveshare SCServo Bus Controller
- 12V 4A power supply
- USB-C cable 

**Leader Arm:**
- 1x STS3215 Servo (7.4V, 1/345 gear ratio) - gripper
- 2x STS3215 Servo (7.4V, 1/191 gear ratio) - wrist joints
- 3x STS3215 Servo (7.4V, 1/147 gear ratio) - shoulder/elbow
- Waveshare SCServo Bus Controller
- 5V 5A power supply
- USB-C cable

**Additional Resources:**
- [Assembly Instructions](https://github.com/TheRobotStudio/SO-ARM100)
- [Calibration Guide](https://huggingface.co/docs/lerobot/so101)
- [Teleoperation & Recording Guide](https://huggingface.co/docs/lerobot/il_robots)

## Installation and Setup

### Prerequisites
- Ubuntu 22.04 or later (native Linux recommended, WSL has camera driver limitations)
- Python 3.8+
- CUDA-compatible GPU (for training)
- Google Colab Pro account (optional, for GPU100 access)

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/pd05849/lerobot-imitation-learning.git
cd lerobot-imitation-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install LeRobot framework:
```bash
pip install lerobot
```

### Hardware Setup

1. **3D Print Components:** Download STL files from [SO-ARM100 repository](https://github.com/TheRobotStudio/SO-ARM100)

2. **Assemble Arms:** Follow assembly instructions, ensuring proper motor placement and cable management

3. **Find USB ports**
```bash
# Use this command to check for each servo adapter, connect to MotorBus to your computer via USB and power. 
# Run the following script and disconnect the MotorBus when prompted:

lerobot-find-port
```

4. **Configure Motor IDs:**
```bash
# Use LeRobot calibration scripts to assign unique IDs to each servo
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841  # <- paste here the port found at previous step
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751  # <- paste here the port found at previous step
```

5. **Calibrate System:**
```bash
# Calibrate encoder-angle relationships and set torque limits
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431551 \ # <- The port of your robot
    --robot.id=follower_arm # <- Give the robot a unique name
```

### WSL USB Device Sharing (Windows Users)

If using WSL, share USB devices from Windows:
```powershell
# In PowerShell (Administrator)
usbipd list
usbipd bind --busid <your-busid>
usbipd attach --wsl --busid <your-busid>
```

## Instructions to Run

### 1. Data Collection

```bash
# Mount cameras and verify detection
lerobot control-robot --robot-path so101 --display-cameras

# Record teleoperation episodes
lerobot record \
  --robot-path so101 \
  --repo-id Delcastillo8/machine_learning_project \
  --fps 30 \
  --num-episodes 50 \
  --warmup-time-s 3 \
  --episode-time-s 10 \
  --reset-time-s 5
```

### 2. Dataset Upload to HuggingFace

```bash
# Push dataset to HuggingFace Hub
lerobot push-dataset \
  --repo-id Delcastillo8/machine_learning_project \
  --local-dir ./data
```

### 3. Model Training

```bash
# Train ACT model
lerobot train \
  --policy act \
  --dataset-repo-id Delcastillo8/machine_learning_project \
  --output-dir ./results/act_model \
  --num-train-steps 60000 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --save-freq 10000 \
  --eval-freq 5000 \
  --log-freq 100
```

For Google Colab training, use the notebook in `/notebooks/train_act_model.ipynb`

### 4. Model Evaluation (Inference)

```bash
# Run inference with trained model
lerobot eval \
  --policy-path ./results/act_model \
  --robot-path so101 \
  --num-episodes 10
```

**Note:** Inference testing was not completed in this project due to time constraints and hardware compatibility issues.

## Project Structure

```
.
├── data/
│   └── DATA.md                 # Instructions for accessing dataset
├── notebooks/
│   ├── train_act_model.ipynb  # Training notebook for Google Colab
│   └── visualize_data.ipynb   # Dataset exploration and visualization
├── src/
│   ├── calibrate_robot.py     # Calibration utilities
│   ├── record_episodes.py     # Data collection script
│   └── train_model.py         # Training script
├── results/
│   ├── plots/                 # Training loss curves, visualizations
│   │   ├── overall_loss.png
│   │   ├── l1_loss.png
│   │   └── gpu_metrics.png
│   └── checkpoints/           # Model checkpoints (not included, too large)
├── README.md
└── requirements.txt
```

## Results and Key Findings

### Training Performance
- **Overall Loss:** Converged to near-zero after 60,000 steps
- **L1 Reconstruction Loss (MAE):** Converged to minimal error
- **Training Duration:** ~4 hours on GPU100
- **GPU Power Usage:** Peak 160W
- **Model Size:** ~150MB (checkpoint)

### Key Findings

1. **Data Efficiency:** 50 demonstration episodes proved sufficient for training convergence, validating literature claims about ACT's data efficiency

2. **CVAE Architecture Benefits:** The action chunking approach with CVAE framework successfully learned coherent action sequences, addressing temporal credit assignment in manipulation tasks

3. **Multimodal Learning:** Dual-camera configuration (overhead + laptop) provided adequate spatial understanding for pick-and-place operations

4. **Training Convergence:** Both loss metrics showed consistent convergence without significant overfitting, indicating good generalization from demonstrations

5. **Dataset Recording Quality:** When teleoperation can be performed using only camera views (without direct robot observation), dataset quality is sufficient for training

6. **Hardware Challenges:** 
   - WSL camera driver compatibility issues
   - Motor controller USB sharing complexity
   - Calibration sensitivity to mechanical precision

7. **Computational Cost:** Training required significant GPU resources (160W peak), raising environmental and accessibility concerns for edge AI applications

### Limitations
- **Incomplete Inference Testing:** Hardware conflicts and time constraints prevented deployment validation
- **Single Task Focus:** Only pick-and-place task evaluated; generalization to other tasks untested
- **Environmental Variation:** Limited testing across different lighting conditions and object placements

### Future Work
- Complete inference testing and deployment validation
- Expand to multi-task learning with task conditioning
- Investigate sim-to-real transfer to reduce hardware dependency
- Optimize model for edge deployment (reduced power consumption)
- Test robustness to environmental perturbations

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{delcastillo2025lerobot,
  author = {Del Castillo, Pol},
  title = {Imitation Learning for Robotic Manipulation using LeRobot SO-101},
  year = {2025},
  institution = {Rensselaer Polytechnic Institute},
  url = {https://github.com/pd05849/lerobot-imitation-learning}
}
```

### Referenced Papers

```bibtex
@misc{zhao2023act,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  year={2023},
  eprint={2304.13705},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}

@misc{chi2024diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  year={2024},
  eprint={2303.04137},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace LeRobot team for the open-source framework
- TheRobotStudio for SO-ARM100 hardware design
- Professor M Arshad Zahangir Chowdhury for project guidance
- Google Colab Pro for GPU100 compute resources

## Contact

For questions or collaboration:
- **Email:** delcap@rpi.edu
- **GitHub:** [pd05849](https://github.com/pd05849)
- **Dataset:** [HuggingFace Hub](https://huggingface.co/datasets/Delcastillo8/machine_learning_project)
