import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("140k/real_vs_fake/real_vs_fake") 
DEST_DIR = Path("gan_dataset")  

def move_with_rename(src, dst_dir):

    """Move file with automatic renaming if duplicate exists"""

    dst_dir.mkdir(parents=True, exist_ok=True)

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

    """Collect all .jpg images from category folders"""

    images = []

    for split_dir in ["train", "test", "valid"]:
        source_path = SOURCE_DIR / split_dir / category
        if source_path.exists():
            images.extend(source_path.glob("*.jpg")) 

    return list(set(images))  

def create_gan_dataset():
    all_real = collect_images("real")
    all_fake = collect_images("fake")
    
    print(f"Found {len(all_real)} real images and {len(all_fake)} fake images")
    if len(all_real) < 600 or len(all_fake) < 600:
        raise ValueError("Need at least 600 real and 600 fake images")
    
    # Random selection
    random.seed(42)
    selected_real = random.sample(all_real, 600)
    selected_fake = random.sample(all_fake, 600)
    
    # Split indices (70-15-15)
    splits = {
        "train": (0, 420),       # 70% of 600
        "test": (420, 510),     # 15%
        "valid": (510, 600)     # 15%
    }
    
    # Move images to destination
    for split_name, (start, end) in splits.items():
        # real images
        real_dest = DEST_DIR / split_name / "real"
        for img in selected_real[start:end]:
            move_with_rename(img, real_dest)
        
        # fake images
        fake_dest = DEST_DIR / split_name / "fake"
        for img in selected_fake[start:end]:
            move_with_rename(img, fake_dest)

if __name__ == "__main__":
    try:
        create_gan_dataset()
        print("Dataset created successfully!")
        print("Final distribution:")
        print(f"- Train: 420 real + 420 fake")
        print(f"- Test: 90 real + 90 fake")
        print(f"- Valid: 90 real + 90 fake")
    except Exception as e:
        print(f"❌ Error: {e}")