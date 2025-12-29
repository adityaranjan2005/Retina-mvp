# Retinal Vessel Analysis Pipeline - MVP v1

A PyTorch-based MVP pipeline for retinal vessel segmentation and analysis with multi-head architecture supporting vessel segmentation, centerline extraction, and artery/vein classification.

## Features

- **Multi-head segmentation model** with three outputs:
  - Vessel segmentation (binary)
  - Centerline extraction (skeleton)
  - Artery/Vein classification (3-class: background/artery/vein)
- **Robust data handling**: Supports mixed image formats (.png, .jpg, .tif, .ppm)
- **Automatic A/V mask handling**: Gracefully handles missing A/V masks
- **Comprehensive metrics**: Computes centerline length, branch points, endpoints, and tortuosity
- **Morphological post-processing**: Closes gaps in vessel predictions

## Project Structure

```
Retina-mvp/
├── data/
│   ├── images/          # Fundus RGB images (.png/.jpg/.tif/.ppm)
│   ├── vessel_masks/    # Binary vessel masks (0=bg, 255=vessel or 0/1)
│   └── av_masks/        # Optional A/V masks (0=bg, 1=artery, 2=vein)
├── src/
│   ├── dataset.py       # Dataset loader and transforms
│   ├── model.py         # Multi-head segmentation model
│   ├── train.py         # Training script
│   ├── infer.py         # Inference script
│   ├── metrics.py       # Centerline metrics computation
│   └── sanity_check.py  # Data integrity checker
├── outputs/             # Generated during training/inference
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. **Clone or navigate to the repository**:
```bash
cd Retina-mvp
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Requirements include:
- PyTorch >= 2.0.0
- segmentation-models-pytorch
- albumentations
- scikit-image
- opencv-python
- numpy, Pillow, tqdm

## Usage

### Step 1: Check Data Integrity

Run the sanity check to verify your dataset:

```bash
python -m src.sanity_check
```

This will display:
- Total number of images
- Number of matched vessel masks
- Number of matched A/V masks
- List of any missing masks

### Step 2: Train the Model

Train the multi-head segmentation model:

```bash
python -m src.train --epochs 3 --img_size 512 --batch_size 4
```

**Training options**:
- `--data_dir`: Path to data directory (default: `data`)
- `--output_dir`: Path to save checkpoints (default: `outputs`)
- `--epochs`: Number of training epochs (default: `3`)
- `--batch_size`: Batch size (default: `4`)
- `--img_size`: Input image size (default: `512`)
- `--lr`: Learning rate (default: `1e-3`)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)

The model will be saved to `outputs/mvp_model.pt`.

**Training details**:
- Uses ResNet34 encoder with ImageNet pre-training
- Combined loss: BCEWithLogits + Dice for vessel/centerline, CrossEntropy for A/V
- Adam optimizer with learning rate 1e-3
- Augmentations: flips, rotations, brightness/contrast, Gaussian noise

### Step 3: Run Inference

Process images and generate predictions:

```bash
python -m src.infer --checkpoint outputs/mvp_model.pt --max_images 10
```

**Inference options**:
- `--checkpoint`: Path to model checkpoint (required)
- `--data_dir`: Path to data directory (default: `data`)
- `--output_dir`: Path to save outputs (default: `outputs`)
- `--max_images`: Maximum number of images to process (default: all)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)

**Output files** (per image):
- `<name>_vessel.png`: Binary vessel segmentation (0 or 255)
- `<name>_centerline.png`: Binary centerline skeleton (0 or 255)
- `<name>_av.png`: A/V classification mask (0/1/2)
- `<name>_metrics.json`: Computed metrics

**Metrics** (in JSON):
```json
{
  "centerline_length_px": 12345.0,
  "branch_points": 42,
  "endpoints": 18,
  "tortuosity_proxy_mean": 1.23,
  "tortuosity_proxy_max": 1.85
}
```

## Model Architecture

- **Encoder**: ResNet34 with ImageNet pre-training
- **Decoder**: U-Net style decoder
- **Heads**: Three separate heads for:
  1. Vessel segmentation (1 channel, sigmoid activation)
  2. Centerline segmentation (1 channel, sigmoid activation)
  3. A/V classification (3 channels, softmax activation)

## Loss Functions

- **Vessel head**: BCE + Dice loss
- **Centerline head**: BCE + Dice loss
- **A/V head**: CrossEntropy (weight: 0.5)
- Samples without A/V masks automatically skip A/V loss

## Post-processing

1. **Vessel prediction**: Morphological closing (3x3 kernel) to bridge small gaps
2. **Centerline extraction**: Skeletonization using scikit-image
3. **Metrics computation**: From centerline skeleton

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (CPU supported)
- At least 8GB RAM

## Troubleshooting

**Missing masks**: The pipeline automatically handles missing A/V masks. Vessel masks are required for all images.

**Out of memory**: Reduce `--batch_size` or `--img_size`

**Slow training**: Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

**Mixed extensions**: The pipeline automatically detects .png, .jpg, .tif, .ppm files using filename stem matching

## Example Workflow

```bash
# 1. Check dataset
python -m src.sanity_check

# 2. Train model (quick MVP run)
python -m src.train --epochs 3 --img_size 512 --batch_size 4

# 3. Run inference on first 10 images
python -m src.infer --checkpoint outputs/mvp_model.pt --max_images 10

# 4. Check outputs
ls outputs/
```

## Notes

- This is an MVP (Minimum Viable Product) - designed to work quickly with reasonable results
- For production use, consider:
  - More training epochs (50-100)
  - Data splitting (train/val/test)
  - Model validation and early stopping
  - Ensemble methods
  - Advanced post-processing

## License

This project is for research and educational purposes.
