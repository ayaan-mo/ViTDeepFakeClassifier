import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = Path("140k")  # Update this path
DEST_DIR = Path("gan_dataset")  # Update this path
RANDOM_SEED = 42

def collect_images():
    pass

def create_splits():
    pass

def move_images(images, split_name, category):
    """Move images to destination directory"""
    dest_dir = DEST_DIR / split_name / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for img in images:
        try:
            shutil.move(str(img), str(dest_dir / img.name))
        except shutil.Error:
            
            base, ext = os.path.splitext(img.name)
            counter = 1
            new_name = f"{base}_{counter}{ext}"
            while (dest_dir / new_name).exists():
                counter += 1
                new_name = f"{base}_{counter}{ext}"
            shutil.move(str(img), str(dest_dir / new_name))

def main():
    for split in ['train', 'test', 'valid']:
        for category in ['real', 'fake']:
            (DEST_DIR / split / category).mkdir(parents=True, exist_ok=True)
    
    # Collect and split images
    real_images, fake_images = collect_images()
    
    real_images = real_images[:600]
    fake_images = fake_images[:600]
    
    real_train, real_test, real_valid = create_splits(real_images)
    fake_train, fake_test, fake_valid = create_splits(fake_images)
    
    move_images(real_train, 'train', 'real')
    move_images(real_test, 'test', 'real')
    move_images(real_valid, 'valid', 'real')
    
    move_images(fake_train, 'train', 'fake')
    move_images(fake_test, 'test', 'fake')
    move_images(fake_valid, 'valid', 'fake')

if __name__ == "__main__":
    main()