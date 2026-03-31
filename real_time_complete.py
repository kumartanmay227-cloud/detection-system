#!/usr/bin/env python3
# Complete real-time road hazard detection - self-contained
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Colors for boxes
COLORS = [(0,255,0), (0,0,255), (255,0,0), (255,255,0)]

print('=== ROAD HAZARD DETECTION ===')
print('Loading YOLOv8n (pretrained)...')
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
print('Detection started. Press Q to quit.')

fps_timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print('No video feed')
        break
    
    # Low visibility enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # YOLO detection
    results = model(enhanced, conf=0.4, verbose=False)
    
    annotated = enhanced.copy()
    
    # Draw detections
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                color = COLORS[int(cls_id) % len(COLORS)]
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f'{int(cls_id)}:{conf:.2f}'
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # FPS
    fps = 1.0 / (time.time() - fps_timer)
    fps_timer = time.time()
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Real-Time Road Hazard Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Detection complete. Use for dashcam/real-time monitoring!')

