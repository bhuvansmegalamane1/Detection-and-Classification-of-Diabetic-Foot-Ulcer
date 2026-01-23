#!/usr/bin/env python3
"""
Cleanup script for the Diabetic Foot Ulcer Detection project
Removes unnecessary files and organizes the project structure
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary files and organize the project"""
    
    project_root = Path.cwd()
    
    # Files to remove
    files_to_remove = [
        'TestWithImage.py',
        'train_high_accuracy.py',
        'launch_app.py',
        'run_app.py',
        'bg.png',
        'water-drops-on-leaf-uhd-4k-wallpaper.jpg',
        '3.jpg',
        'foot (14).jpg',
        'foot (15).jpg',
        'foot (16).jpg',
        'foot (54).jpg',
        'images (2).jfif'
    ]
    
    # Directories to remove
    dirs_to_remove = [
        'waste'
    ]
    
    # Create test_images directory for sample images
    test_images_dir = project_root / 'test_images'
    test_images_dir.mkdir(exist_ok=True)
    
    print("🧹 Cleaning up project...")
    
    # Remove unnecessary files
    for file_path in files_to_remove:
        file_full_path = project_root / file_path
        if file_full_path.exists():
            try:
                file_full_path.unlink()
                print(f"✅ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
    
    # Move test images to test_images directory
    test_image_patterns = ['foot*.jpg', '*.jfif']
    for pattern in test_image_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.name not in ['streamlit_app.py']:
                try:
                    destination = test_images_dir / file_path.name
                    shutil.move(str(file_path), str(destination))
                    print(f"📁 Moved to test_images: {file_path.name}")
                except Exception as e:
                    print(f"❌ Error moving {file_path.name}: {e}")
    
    # Remove empty directories
    for dir_path in dirs_to_remove:
        dir_full_path = project_root / dir_path
        if dir_full_path.exists() and not any(dir_full_path.iterdir()):
            try:
                dir_full_path.rmdir()
                print(f"✅ Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"❌ Error removing directory {dir_path}: {e}")
    
    print("\n✨ Cleanup completed!")
    print(f"📁 Test images moved to: {test_images_dir}")
    print("\n📁 Recommended project structure:")
    print("Diabetic-Foot-Ulcer-Detection/")
    print("├── streamlit_app.py          # Main web application")
    print("├── GUI.py                   # Desktop application")
    print("├── requirements.txt         # Dependencies")
    print("├── README.md               # Documentation")
    print("├── test_images/            # Sample test images")
    print("├── configs/                # Configuration files")
    print("└── models/                 # Trained models")

if __name__ == "__main__":
    cleanup_project()