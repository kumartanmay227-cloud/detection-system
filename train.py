#!/usr/bin/env python
"""
Training script for Road Hazard Detection using YOLOv8.
Fine-tunes pre-trained model on low-visibility road hazards.
"""

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import wandb
from utils.visualize import Visualizer
from utils.data import create_dataloaders
import argparse
import os
from datetime import datetime

def load_config(config_path: str = 'config.yaml'):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config):
    """Train YOLOv8 model."""
    model_name = config['model']['name']
    epochs = config['model']['epochs']
    img_size = config['data']['img_size']
    device = config['model']['device']
    project_dir = Path(config['paths']['models_dir'] or './models')
    project_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = YOLO(f'{model_name}.pt')  # Load pretrained
    
    # Dataset path
    data_yaml = Path(config['data']['root_dir']) / 'dataset.yaml'
    
    print(f"🚀 Starting training on {device}")
    print(f"Dataset: {data_yaml}")
    print(f"Model: {model_name}, Epochs: {epochs}")
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        device=device,
        batch=config['data']['batch_size'],
        patience=config['model']['patience'],
        lr0=config['model']['lr'],
        project=str(project_dir),
        name='road_hazard_v1',
        save=True,
        plots=True,
        val=True
    )
    
    # Plot metrics
    viz = Visualizer()
    # Note: results.history available post-train
    # viz.plot_metrics(results.results_dict)
    
    print("✅ Training complete!")
    print(f"Best model: {project_dir / 'road_hazard_v1' / 'weights' / 'best.pt'}")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(config['project']['seed'])
    
    # Train
    model = train_model(config)
    
    # Test on val set
    print("\n🔍 Running validation...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    main()

