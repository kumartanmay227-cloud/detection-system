#!/usr/bin/env python
"""
Real-time road hazard detection for dashcam - NO optional deps.
Pretrained YOLOv8n + enhancement + FPS + alerts.
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils.visualize import Visualizer

# Load pretrained model
print('Loading YOLOv8n...')
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
viz = Visualizer()

print('Real-time dashcam detection ON. Press Q to quit.')
print('Detects people/cars as hazards...')

fps_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Basic low-vis enhancement (no albumentations)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Inference
    results = model(enhanced, conf=0.4, verbose=False)
    
    # Visualize
    annotated = viz.draw_detections(enhanced, results, 0.4)
    
    # FPS
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Hazard count (person=0, car=2)
    hazards = 0
    for r in results:
        if r.boxes is not None:
            cls_ids = r.boxes.cls.cpu().numpy()
            hazards += sum(1 for cls in cls_ids if int(cls) in [0, 2])
    
    if hazards > 0:
        cv2.putText(annotated, f'HAZARD: {hazards}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
    
    cv2.imshow('Dashcam Hazard Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Ready for production real-time use!')

