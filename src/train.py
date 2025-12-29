import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import RetinaDataset
from src.model import MultiHeadRetinaModel, CombinedLoss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    vessel_loss_sum = 0
    centerline_loss_sum = 0
    av_loss_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)
        vessel_masks = batch['vessel_mask'].to(device)
        centerline_masks = batch['centerline_mask'].to(device)
        av_masks = batch['av_mask'].to(device)
        has_av_mask = batch['has_av_mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        losses = criterion(
            outputs,
            vessel_masks,
            centerline_masks,
            av_masks,
            has_av_mask
        )
        
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        vessel_loss_sum += losses['vessel'].item()
        centerline_loss_sum += losses['centerline'].item()
        av_loss_sum += losses['av'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'vessel': f"{losses['vessel'].item():.4f}",
            'centerline': f"{losses['centerline'].item():.4f}",
            'av': f"{losses['av'].item():.4f}"
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'vessel_loss': vessel_loss_sum / num_batches,
        'centerline_loss': centerline_loss_sum / num_batches,
        'av_loss': av_loss_sum / num_batches
    }


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    img_size: int = 512,
    lr: float = 1e-3,
    device: str = None
):
    """Main training function."""
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    print(f"Loading dataset from {data_dir}...")
    dataset = RetinaDataset(
        data_dir=data_dir,
        img_size=img_size,
        train=True
    )
    
    print(f"Found {len(dataset)} training samples")
    
    # Count samples with A/V masks
    num_av_samples = sum(1 for s in dataset.samples if s['av_mask'] is not None)
    print(f"Samples with A/V masks: {num_av_samples}/{len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    print("Building model...")
    model = MultiHeadRetinaModel(
        encoder_name="resnet34",
        encoder_weights="imagenet"
    )
    model = model.to(device)
    
    # Create loss and optimizer
    criterion = CombinedLoss(av_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Vessel Loss: {metrics['vessel_loss']:.4f}")
        print(f"  Centerline Loss: {metrics['centerline_loss']:.4f}")
        print(f"  A/V Loss: {metrics['av_loss']:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / "mvp_model.pt"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'img_size': img_size,
    }, checkpoint_path)
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train retinal vessel segmentation model")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
