import os
import random
import shutil
from pathlib import Path

# Configuration - UPDATE THESE PATHS
SOURCE_DIR = Path("140k/real_vs_fake/real_vs_fake")  
DEST_DIR = Path("gan_dataset")  

def move_with_rename(src, dst_dir):
    """Move file with automatic renaming if duplicate exists"""
    filename = src.name
    name, ext = os.path.splitext(filename)
    counter = 1
    dst_path = dst_dir / filename
    while dst_path.exists():
        dst_path = dst_dir / f"{name}_{counter}{ext}"
        counter += 1
    shutil.move(str(src), str(dst_path))
    return dst_path

def collect_images(category):
    """Collect all images of a category from nested folders"""
    images = []
    for split in ["train", "test", "valid"]:
        split_dir = SOURCE_DIR / split / category
        if split_dir.exists():
            images.extend(split_dir.glob("*.[jpJP][pnPN]*[gG]"))  # Match jpg/jpeg/png
    return list(set(images))  # Remove duplicates

def create_gan_dataset():
    # Collect all available images
    all_real = collect_images("real")
    all_fake = collect_images("fake")
    
    # Validate sufficient images
    if len(all_real) < 600:
        raise ValueError(f"Not enough real images: found {len(all_real)}, need 600")
    if len(all_fake) < 600:
        raise ValueError(f"Not enough fake images: found {len(all_fake)}, need 600")
    
    # Random selection
    random.seed(42)  # For reproducibility
    selected_real = random.sample(all_real, 600)
    selected_fake = random.sample(all_fake, 600)
    
    # Split into 70-15-15
    splits = {
        "train": (0, 420),
        "test": (420, 510),
        "valid": (510, 600)
    }
    
    # Move images
    for split_name, (start, end) in splits.items():
        # Move real images
        real_dest = DEST_DIR / split_name / "real"
        for img in selected_real[start:end]:
            move_with_rename(img, real_dest)
        
        # Move fake images
        fake_dest = DEST_DIR / split_name / "fake"
        for img in selected_fake[start:end]:
            move_with_rename(img, fake_dest)

if __name__ == "__main__":
    try:
        create_gan_dataset()
        print("✅ Successfully created dataset with:")
        print(f"- 840 train images (420 real + 420 fake)")
        print(f"- 180 test images (90 real + 90 fake)")
        print(f"- 180 valid images (90 real + 90 fake)")
    except Exception as e:
        print(f"❌ Error: {e}")