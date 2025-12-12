# Training Results

This folder contains training outputs, model checkpoints, and visualizations.

## Contents

### Plots
- `overall_loss.png` - Overall training loss curve over 60,000 steps
- `l1_loss.png` - L1 reconstruction loss (MAE) curve
- `gpu_metrics.png` - GPU power usage and temperature during training

### Checkpoints (Not Included - Too Large)
Model checkpoints are not included in this repository due to size constraints (~150MB each).

**Download trained model:**
- Available upon request or via Google Drive/Box link
- Contact: delcap@rpi.edu

### Training Logs
- Tensorboard logs available in `logs/` subdirectory
- View with: `tensorboard --logdir results/logs`

## Training Summary

**Hardware:**
- Platform: Google Colab Pro
- GPU: GPU100
- Peak Power: 160W
- Temperature: ~75°C sustained

**Training Configuration:**
- Steps: 60,000
- Duration: ~2.5 hours
- Batch Size: 8
- Learning Rate: 1e-4
- Optimizer: AdamW
- Scheduler: Cosine Annealing

**Final Metrics:**
- Overall Loss: ~0.02 (converged)
- L1 Reconstruction Loss: ~0.015 (converged)
- KL Divergence: ~2.5

**Convergence:**
- Both overall and L1 loss showed steady convergence
- No significant overfitting observed
- Loss plateaued around 45,000 steps
- Continued training to 60,000 steps for stability

## Reproducing Results

To reproduce training results:

```bash
# Using provided training script
python src/train_model.py \
  --dataset-repo-id Delcastillo8/machine_learning_project \
  --output-dir ./results/act_model \
  --num-epochs 1000 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 42

# Or use Google Colab notebook
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb
```

## Visualizations

## Visualizations

### Overall Loss
![Overall Training Loss](plots/overall_loss.png)

### L1 Reconstruction Loss (MAE)
![L1 Loss](plots/mae_loss.png)

### GPU Metrics
![GPU Power Usage](plots/gpu_power.png)
![GPU Temperature](plots/gpu_temperature.png)

### Loss Curves
Training showed consistent convergence without oscillations, indicating stable learning dynamics and appropriate hyperparameter selection.

### GPU Metrics
Power consumption remained high throughout training (~160W), highlighting computational demands of transformer-based policies and raising considerations for deployment on edge devices.


