#!/usr/bin/env python
"""
Dataset preprocessing and download script.
Downloads sample BDD100K or COCO dataset, creates splits, applies low-vis simulation.
"""

import os
import yaml
import requests
import zipfile
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utils.augment import add_fog, LowVisibilityAugmentation
import shutil

def load_config(config_path: str = 'config.yaml'):
    """Load config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_sample_dataset(config):
    """Download sample hazard dataset or BDD100K subset."""
    data_dir = Path(config['paths']['data_root_dir'] or './data')
    data_dir.mkdir(exist_ok=True)
    
    # Create sample dataset for demo (real BDD100K needs API key)
    print("Creating sample road hazard dataset...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = data_dir / split / 'images'
        label_dir = data_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 100 sample images per split with synthetic annotations
        for i in range(100):
            # Load sample image or create synthetic
            img_path = img_dir / f'{split}_{i:03d}.jpg'
            label_path = label_dir / f'{split}_{i:03d}.txt'
            
            # Create synthetic road image (for demo)
            img = create_sample_road_image()
            
            # Add synthetic hazards
            img, labels = add_synthetic_hazards(img)
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                for cls_id, x, y, w, h in labels:
                    f.write(f'{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    
    # Create dataset.yaml for YOLO
    dataset_yaml = data_dir / 'dataset.yaml'
    with open(dataset_yaml, 'w') as f:
        f.write('''path: ./data
train: train/images
val: val/images
test: test/images

nc: 8
names: ['pothole', 'obstacle', 'pedestrian', 'vehicle', 'debris', 'person', 'car', 'truck']
''')
    
    print(f"✅ Sample dataset created at {data_dir}")
    print(f"Dataset YAML: {dataset_yaml}")

def create_sample_road_image(size=(640, 640)):
    """Create synthetic road image."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 100  # Dark road
    
    # Road lines
    cv2.line(img, (100, 200), (540, 200), (255, 255, 0), 20)
    cv2.line(img, (100, 400), (540, 400), (255, 255, 0), 20)
    
    # Road edges
    cv2.rectangle(img, (0, 0), (size[0], 50), (50, 50, 50), -1)
    cv2.rectangle(img, (0, size[1]-50), (size[0], size[1]), (50, 50, 50), -1)
    
    return img

def add_synthetic_hazards(img, prob=0.7):
    """Add synthetic hazards to image."""
    h, w = img.shape[:2]
    labels = []
    hazards = []
    
    if np.random.random() < prob:
        # Pothole
        x, y = np.random.randint(50, w-50, 2)
        size = np.random.randint(30, 60)
        cv2.circle(img, (x, y), size//2, (50, 50, 50), -1)
        norm_x = (x / w)
        norm_y = (y / h)
        norm_w = (size / w)
        norm_h = (size / h)
        labels.append((0, norm_x, norm_y, norm_w, norm_h))
    
    if np.random.random() < prob:
        # Pedestrian
        x, y = np.random.randint(150, w-150, 2)
        cv2.rectangle(img, (x-20, y-50), (x+20, y+20), (0, 255, 255), -1)
        norm_x = (x / w)
        norm_y = (y / h)
        norm_w = 40 / w
        norm_h = 70 / h
        labels.append((2, norm_x, norm_y, norm_w, norm_h))
    
    # Apply low visibility
    img = add_fog(img, fog_coef=0.3)
    
    return img, labels

def main():
    config = load_config()
    download_sample_dataset(config)
    print("\nDataset ready! Run: python train.py")

if __name__ == "__main__":
    main()

