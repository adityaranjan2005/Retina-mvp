import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize


def find_matching_file(image_path: Path, mask_dir: Path, extensions: List[str]) -> Optional[Path]:
    """Find a mask file matching the image stem with any of the given extensions."""
    stem = image_path.stem
    
    # Try exact match first
    for ext in extensions:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    
    # Try common vessel mask naming patterns
    patterns = [
        f"{stem}-vessels4",  # im0001-vessels4.ppm
        f"{stem}.ah",         # im0001.ah.ppm
        f"{stem}.vk",         # im0001.vk.ppm
        f"{stem}_manual1",    # 21_manual1.png
    ]
    
    for pattern in patterns:
        for ext in extensions:
            candidate = mask_dir / f"{pattern}{ext}"
            if candidate.exists():
                return candidate
    
    return None


def load_image(path: Path) -> np.ndarray:
    """Load image and convert to RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    """Load mask and convert to numpy array."""
    mask = Image.open(path)
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask_arr = np.array(mask)
    return mask_arr


def normalize_vessel_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize vessel mask to 0/1 values."""
    # Handle both 0/255 and 0/1 formats
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    return mask


def normalize_av_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize A/V mask to 0/1/2 values."""
    # Ensure values are in 0/1/2 range
    mask = np.clip(mask, 0, 2).astype(np.uint8)
    return mask


def generate_centerline(vessel_mask: np.ndarray) -> np.ndarray:
    """Generate centerline from vessel mask using skeletonization."""
    binary_vessel = (vessel_mask > 0).astype(bool)
    skeleton = skeletonize(binary_vessel)
    return skeleton.astype(np.uint8)


class RetinaDataset(Dataset):
    """Dataset for retinal vessel segmentation with multi-head outputs."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        img_size: int = 512,
        train: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_size = img_size
        self.train = train
        
        # Directories
        self.images_dir = self.data_dir / "images"
        self.vessel_masks_dir = self.data_dir / "vessel_masks"
        self.av_masks_dir = self.data_dir / "av_masks"
        
        # Extensions to check
        self.extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm']
        
        # Find all images
        self.image_files = []
        for ext in self.extensions:
            self.image_files.extend(list(self.images_dir.glob(f"*{ext}")))
        
        self.image_files = sorted(self.image_files)
        
        # Match masks
        self.samples = []
        for img_path in self.image_files:
            vessel_mask_path = find_matching_file(img_path, self.vessel_masks_dir, self.extensions)
            if vessel_mask_path is None:
                continue  # Skip if no vessel mask
            
            av_mask_path = find_matching_file(img_path, self.av_masks_dir, self.extensions)
            
            self.samples.append({
                'image': img_path,
                'vessel_mask': vessel_mask_path,
                'av_mask': av_mask_path  # Can be None
            })
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transform(train)
    
    def get_default_transform(self, train: bool) -> A.Compose:
        """Get default augmentation pipeline."""
        if train:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = load_image(sample['image'])
        
        # Load vessel mask
        vessel_mask = load_mask(sample['vessel_mask'])
        vessel_mask = normalize_vessel_mask(vessel_mask)
        
        # Generate centerline from vessel mask
        centerline_mask = generate_centerline(vessel_mask)
        
        # Load A/V mask if available
        has_av_mask = sample['av_mask'] is not None
        if has_av_mask:
            av_mask = load_mask(sample['av_mask'])
            av_mask = normalize_av_mask(av_mask)
        else:
            # Create dummy A/V mask (all background)
            av_mask = np.zeros_like(vessel_mask, dtype=np.uint8)
        
        # Apply augmentations
        if self.transform:
            # Albumentations expects additional_targets for multiple masks
            transformed = self.transform(
                image=image,
                masks=[vessel_mask, centerline_mask, av_mask]
            )
            image = transformed['image']
            masks = transformed['masks']
            
            # Masks are now tensors from ToTensorV2
            # Check if they're tensors or numpy arrays
            if isinstance(masks[0], torch.Tensor):
                vessel_mask = masks[0].float().unsqueeze(0)
                centerline_mask = masks[1].float().unsqueeze(0)
                av_mask = masks[2].long()
            else:
                vessel_mask = torch.from_numpy(masks[0]).float().unsqueeze(0)
                centerline_mask = torch.from_numpy(masks[1]).float().unsqueeze(0)
                av_mask = torch.from_numpy(masks[2]).long()
        
        return {
            'image': image,
            'vessel_mask': vessel_mask,
            'centerline_mask': centerline_mask,
            'av_mask': av_mask,
            'has_av_mask': torch.tensor(has_av_mask, dtype=torch.bool),
            'filename': sample['image'].stem
        }
