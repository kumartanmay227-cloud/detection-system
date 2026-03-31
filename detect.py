#!/usr/bin/env python
"""
Real-time road hazard detection from webcam/video.
Includes low visibility enhancement and FPS display.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils.augment import LowVisibilityAugmentation
from utils.visualize import Visualizer
from utils.data import denormalize_bboxes
import yaml
import time
import torch

def load_config(config_path: str = 'config.yaml'):
    """Load config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=0, help='webcam(0), video path, or image')
    parser.add_argument('--model', default='models/road_hazard_v1/weights/best.pt', help='Model path')
    parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    conf_threshold = args.conf or config['inference']['conf_threshold']
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Run: python preprocess.py && python train.py")
        return
    
    model = YOLO(str(model_path))
    model.to('cpu')  # Force CPU
    
    # Visualizer
    viz = Visualizer()
    
    # Source setup
    source = args.source
    if source == '0' or source == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    print(f"🔍 Detection started. Press 'q' to quit.")
    print(f"Model: {model_path}")
    print(f"Conf: {conf_threshold}")
    
    fps_start = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Enhance low visibility
        enhanced = LowVisibilityAugmentation.enhance_low_vis(frame)
        
        # Inference
        results = model(enhanced, conf=conf_threshold, verbose=False)
        
        # Visualize
        annotated = viz.draw_detections(enhanced, results, conf_threshold)
        
        # FPS
        fps = viz.calculate_fps(fps_start)
        fps_start = time.time()
        annotated = viz.draw_fps(annotated, fps)
        
        # Display hazard alert
        hazard_count = sum(len(r.boxes) for r in results if len(r.boxes) > 0)
        if hazard_count > 0:
            cv2.putText(annotated, f"HAZARDS: {hazard_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Road Hazard Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

