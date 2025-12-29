import os
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from src.model import MultiHeadRetinaModel
from src.dataset import load_image, RetinaDataset
from src.metrics import (
    compute_skeleton_metrics,
    extract_centerline_from_vessel,
    postprocess_vessel_mask
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_inference_transform(img_size: int) -> A.Compose:
    """Get inference transform."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiHeadRetinaModel(
        encoder_name="resnet34",
        encoder_weights=None  # Load from checkpoint
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    img_size = checkpoint.get('img_size', 512)
    
    return model, img_size


def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: A.Compose,
    device: str,
    original_size: tuple
) -> dict:
    """Run inference on a single image."""
    
    # Transform image
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Post-process predictions
    vessel_pred = torch.sigmoid(outputs['vessel']).cpu().numpy()[0, 0]
    centerline_pred = torch.sigmoid(outputs['centerline']).cpu().numpy()[0, 0]
    av_pred = torch.softmax(outputs['av'], dim=1).cpu().numpy()[0]
    av_pred = np.argmax(av_pred, axis=0)
    
    # Resize to original size
    vessel_pred = resize_prediction(vessel_pred, original_size)
    centerline_pred = resize_prediction(centerline_pred, original_size)
    av_pred = resize_prediction(av_pred, original_size, interpolation='nearest')
    
    # Threshold vessel and centerline
    vessel_binary = (vessel_pred > 0.5).astype(np.uint8)
    centerline_binary = (centerline_pred > 0.5).astype(np.uint8)
    
    # Postprocess vessel mask (morphological closing)
    vessel_binary = postprocess_vessel_mask(vessel_binary, kernel_size=3) // 255
    
    # Re-extract centerline from postprocessed vessel
    centerline_binary = extract_centerline_from_vessel(vessel_binary)
    
    return {
        'vessel': vessel_binary,
        'centerline': centerline_binary,
        'av': av_pred.astype(np.uint8)
    }


def resize_prediction(pred: np.ndarray, target_size: tuple, interpolation: str = 'linear') -> np.ndarray:
    """Resize prediction to target size."""
    from scipy.ndimage import zoom
    
    h, w = target_size
    zoom_factors = (h / pred.shape[0], w / pred.shape[1])
    
    if interpolation == 'nearest':
        order = 0
    else:
        order = 1
    
    resized = zoom(pred, zoom_factors, order=order)
    return resized


def save_predictions(
    predictions: dict,
    output_dir: Path,
    filename: str
):
    """Save predictions to disk."""
    
    # Save vessel mask (0 or 255)
    vessel_img = (predictions['vessel'] * 255).astype(np.uint8)
    vessel_path = output_dir / f"{filename}_vessel.png"
    Image.fromarray(vessel_img).save(vessel_path)
    
    # Save centerline mask (0 or 255)
    centerline_img = (predictions['centerline'] * 255).astype(np.uint8)
    centerline_path = output_dir / f"{filename}_centerline.png"
    Image.fromarray(centerline_img).save(centerline_path)
    
    # Save A/V mask (0/1/2)
    av_img = predictions['av'].astype(np.uint8)
    av_path = output_dir / f"{filename}_av.png"
    Image.fromarray(av_img).save(av_path)


def save_metrics(
    metrics: dict,
    output_dir: Path,
    filename: str
):
    """Save metrics to JSON."""
    metrics_path = output_dir / f"{filename}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def infer(
    checkpoint: str,
    data_dir: str,
    output_dir: str,
    max_images: int = None,
    device: str = None
):
    """Run inference on images."""
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {checkpoint}...")
    model, img_size = load_model(checkpoint, device)
    print(f"Model loaded with img_size={img_size}")
    
    # Get transform
    transform = get_inference_transform(img_size)
    
    # Find images
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm']
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    image_files = sorted(image_files)
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    for img_path in tqdm(image_files, desc="Processing images"):
        filename = img_path.stem
        
        # Load image
        image = load_image(img_path)
        original_size = image.shape[:2]
        
        # Predict
        predictions = predict_image(
            model=model,
            image=image,
            transform=transform,
            device=device,
            original_size=original_size
        )
        
        # Compute metrics from centerline
        metrics = compute_skeleton_metrics(predictions['centerline'])
        
        # Save outputs
        save_predictions(predictions, output_dir, filename)
        save_metrics(metrics, output_dir, filename)
    
    print(f"\nInference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on retinal images")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    infer(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        device=args.device
    )


if __name__ == '__main__':
    main()
