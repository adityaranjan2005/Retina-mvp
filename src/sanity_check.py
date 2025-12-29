import os
import argparse
from pathlib import Path
from typing import List


def find_matching_file(image_path: Path, mask_dir: Path, extensions: List[str]) -> bool:
    """Check if a matching mask file exists."""
    stem = image_path.stem
    
    # Try exact match first
    for ext in extensions:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return True
    
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
                return True
    
    return False


def sanity_check(data_dir: str = "data"):
    """Perform sanity check on the dataset."""
    
    data_dir = Path(data_dir)
    
    print("=" * 60)
    print("RETINAL VESSEL DATASET SANITY CHECK")
    print("=" * 60)
    
    # Directories
    images_dir = data_dir / "images"
    vessel_masks_dir = data_dir / "vessel_masks"
    av_masks_dir = data_dir / "av_masks"
    
    # Check if directories exist
    print("\n1. Directory Structure:")
    print(f"   Data directory: {data_dir.absolute()}")
    print(f"   Images directory exists: {images_dir.exists()}")
    print(f"   Vessel masks directory exists: {vessel_masks_dir.exists()}")
    print(f"   A/V masks directory exists: {av_masks_dir.exists()}")
    
    if not images_dir.exists():
        print("\n❌ ERROR: Images directory not found!")
        return
    
    if not vessel_masks_dir.exists():
        print("\n❌ ERROR: Vessel masks directory not found!")
        return
    
    # Extensions to check
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm']
    
    # Find all images
    print("\n2. Image Files:")
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    image_files = sorted(image_files)
    print(f"   Total images found: {len(image_files)}")
    
    if len(image_files) == 0:
        print("\n❌ ERROR: No images found!")
        return
    
    # Show image extensions breakdown
    ext_count = {}
    for img in image_files:
        ext = img.suffix.lower()
        ext_count[ext] = ext_count.get(ext, 0) + 1
    
    print(f"   Image extensions:")
    for ext, count in sorted(ext_count.items()):
        print(f"      {ext}: {count} images")
    
    # Match vessel masks
    print("\n3. Vessel Masks:")
    vessel_matches = []
    vessel_missing = []
    
    for img_path in image_files:
        if find_matching_file(img_path, vessel_masks_dir, extensions):
            vessel_matches.append(img_path)
        else:
            vessel_missing.append(img_path)
    
    print(f"   Vessel masks matched: {len(vessel_matches)}/{len(image_files)}")
    
    if len(vessel_missing) > 0:
        print(f"   ⚠️  Missing vessel masks for {len(vessel_missing)} images:")
        for img_path in vessel_missing[:10]:  # Show first 10
            print(f"      - {img_path.name}")
        if len(vessel_missing) > 10:
            print(f"      ... and {len(vessel_missing) - 10} more")
    else:
        print(f"   ✓ All images have vessel masks!")
    
    # Match A/V masks
    print("\n4. A/V Masks:")
    if not av_masks_dir.exists():
        print(f"   A/V masks directory does not exist - A/V training will be disabled")
    else:
        av_matches = []
        av_missing = []
        
        for img_path in image_files:
            if find_matching_file(img_path, av_masks_dir, extensions):
                av_matches.append(img_path)
            else:
                av_missing.append(img_path)
        
        print(f"   A/V masks matched: {len(av_matches)}/{len(image_files)}")
        
        if len(av_matches) == 0:
            print(f"   ⚠️  No A/V masks found - A/V training will be disabled")
        elif len(av_missing) > 0:
            print(f"   ⚠️  Missing A/V masks for {len(av_missing)} images:")
            for img_path in av_missing[:10]:  # Show first 10
                print(f"      - {img_path.name}")
            if len(av_missing) > 10:
                print(f"      ... and {len(av_missing) - 10} more")
            print(f"   Note: Images without A/V masks will skip A/V loss during training")
        else:
            print(f"   ✓ All images have A/V masks!")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Images with vessel masks: {len(vessel_matches)}")
    
    if av_masks_dir.exists():
        av_matches_count = len([img for img in image_files 
                                if find_matching_file(img, av_masks_dir, extensions)])
        print(f"Images with A/V masks: {av_matches_count}")
    else:
        print(f"Images with A/V masks: 0 (directory not found)")
    
    print("\n✓ Dataset sanity check complete!")
    
    if len(vessel_matches) > 0:
        print(f"\n✓ Ready for training with {len(vessel_matches)} samples")
    else:
        print(f"\n❌ Cannot train - no matched image/vessel mask pairs!")


def main():
    parser = argparse.ArgumentParser(description="Sanity check for retinal vessel dataset")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    
    args = parser.parse_args()
    sanity_check(args.data_dir)


if __name__ == '__main__':
    main()
