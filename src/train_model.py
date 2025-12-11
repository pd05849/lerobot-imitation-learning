"""
Train Action Chunking Transformer (ACT) model
Implements CVAE-based policy for robotic manipulation
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from lerobot.models.act import ACTPolicy
from lerobot.datasets import LeRobotDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(config: dict):
    """
    Create ACT policy model
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        ACT policy model
    """
    model = ACTPolicy(
        input_shapes={
            "camera_overhead": (3, 480, 640),
            "camera_laptop": (3, 480, 640),
            "joint_positions": (6,)
        },
        output_shapes={
            "target_joint_positions": (6,),
            "gripper_command": (1,)
        },
        # CVAE parameters
        latent_dim=config.get("latent_dim", 32),
        hidden_dim=config.get("hidden_dim", 512),
        
        # Transformer parameters
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 4),
        
        # Action chunking
        chunk_size=config.get("chunk_size", 100),
        
        # VAE loss weight (beta)
        kl_weight=config.get("kl_weight", 10.0)
    )
    
    return model

def train_epoch(model, dataloader, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # Calculate losses
        loss = output["loss"]
        recon_loss = output["reconstruction_loss"]
        kl_loss = output["kl_loss"]
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/reconstruction_loss", recon_loss.item(), global_step)
        writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon, avg_kl

def main():
    parser = argparse.ArgumentParser(description="Train ACT policy")
    parser.add_argument("--dataset-repo-id", type=str, 
                      default="Delcastillo8/machine_learning_project",
                      help="HuggingFace dataset repository ID")
    parser.add_argument("--output-dir", type=str, default="./results/act_model",
                      help="Output directory for checkpoints and logs")
    parser.add_argument("--num-epochs", type=int, default=1000,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda/cpu)")
    parser.add_argument("--save-freq", type=int, default=100,
                      help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = vars(args)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("=== Training ACT Policy ===")
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        split="train"
    )
    print(f"Dataset size: {len(dataset)} frames")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )
    
    # Create model
    print("\nInitializing model...")
    model_config = {
        "latent_dim": 32,
        "hidden_dim": 512,
        "n_heads": 8,
        "n_layers": 4,
        "chunk_size": 100,
        "kl_weight": 10.0
    }
    model = create_model(model_config)
    
    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    # Tensorboard writer
    writer = SummaryWriter(output_dir / "logs")
    
    # Training loop
    print("\n=== Starting Training ===")
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        avg_loss, avg_recon, avg_kl = train_epoch(
            model, dataloader, optimizer, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        writer.add_scalar("epoch/reconstruction_loss", avg_recon, epoch)
        writer.add_scalar("epoch/kl_loss", avg_kl, epoch)
        writer.add_scalar("epoch/learning_rate", 
                         optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch + 1} complete - "
              f"Avg Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or avg_loss < best_loss:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model_config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'config': model_config
                }, best_path)
                print(f"New best model saved (loss: {best_loss:.4f})")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config
    }, final_path)
    print(f"\n=== Training Complete ===")
    print(f"Final model saved to {final_path}")
    print(f"Best loss: {best_loss:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()
