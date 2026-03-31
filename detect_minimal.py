#!/usr/bin/env python
"""
Minimal YOLOv8 road hazard detection demo without optional deps.
"""
import cv2
import time
from ultralytics import YOLO
from utils.visualize import Visualizer

# Load model (auto-downloads yolov8n.pt)
print('Loading YOLOv8n...')
model = YOLO('yolov8n.pt')
model.to('cpu')

# Webcam
cap = cv2.VideoCapture(0)
viz = Visualizer()

print('Detection ready. Press Q to quit.')

fps_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    results = model(frame, conf=0.5, verbose=False)
    
    # Visualize
    annotated = viz.draw_detections(frame, results)
    
    # FPS
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Road Hazard Detection Demo', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Demo complete!')

