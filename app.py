#!/usr/bin/env python
"""
Streamlit demo app for Road Hazard Detection.
Upload image/video or use webcam for real-time detection.
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from utils.visualize import Visualizer
from utils.augment import LowVisibilityAugmentation
import time
import tempfile
import os

@st.cache_resource
def load_model(model_path: str):
    """Load cached model."""
    model = YOLO(model_path)
    model.to('cpu')
    return model

def main():
    st.set_page_config(page_title="Road Hazard Detection", layout="wide")
    st.title("🚗 Deep Learning Road Hazard Detection")
    st.markdown("Detect potholes, pedestrians, vehicles & obstacles in low visibility conditions")
    
    # Sidebar
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Path", value="models/road_hazard_v1/weights/best.pt")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                model = load_model(model_path)
                st.sidebar.success("Model loaded!")
                st.session_state.model = model
            except:
                st.error("Model not found. Run: python train.py")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📹 Webcam"])
    
    with tab1:
        st.header("Image Upload & Detection")
        uploaded_file = st.file_uploader("Upload road image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file is not None and 'model' in st.session_state:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhance
            enhanced = LowVisibilityAugmentation.enhance_low_vis(image)
            
            # Detect
            results = st.session_state.model(enhanced, conf=conf_threshold, verbose=False)
            
            # Visualize
            viz = Visualizer()
            annotated = viz.draw_detections(enhanced, results, conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.image([image_rgb, annotated_rgb], caption=['Original', 'Detection'], width=600)
            
            # Stats
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        detections.append((conf, cls))
            
            if detections:
                st.metric("Hazards Detected", len(detections))
                st.dataframe(pd.DataFrame(detections, columns=['Confidence', 'Class ID']))
    
    with tab2:
        st.header("Video Detection")
        uploaded_video = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None and 'model' in st.session_state:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for demo
                if frame_count % 10 == 0:
                    enhanced = LowVisibilityAugmentation.enhance_low_vis(frame)
                    results = st.session_state.model(enhanced, conf=conf_threshold, verbose=False)
                    annotated = Visualizer().draw_detections(enhanced, results, conf_threshold)
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption='Processing...')
                
                frame_count += 1
            
            cap.release()
            os.unlink(tfile.name)
    
    with tab3:
        st.header("Live Webcam Detection")
        picture = st.camera_input("Take a picture")
        
        if picture and 'model' in st.session_state:
            image = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            enhanced = LowVisibilityAugmentation.enhance_low_vis(image)
            results = st.session_state.model(enhanced, conf=conf_threshold, verbose=False)
            annotated = Visualizer().draw_detections(enhanced, results, conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.image([image_rgb, annotated_rgb], caption=['Live', 'Detection'])

if __name__ == "__main__":
    import pandas as pd
    main()

