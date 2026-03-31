import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")
st.title("🚗 Road Hazard Detection Demo")
st.markdown("**Upload image/video or use webcam**")

# Load model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.header("📁 Upload Image")
    uploaded = st.file_uploader("Choose image", type=['jpg','png','jpeg'])
    
    if uploaded:
        image = Image.open(uploaded)
        image_np = np.array(image)
        
        # Enhance
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(3.0)
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Detect
        results = model(enhanced, conf=0.4)
        annotated = results[0].plot()
        
        st.image([image_np, annotated], caption=['Original', 'Hazards Detected'])

with col2:
    st.header("📹 Live Webcam")
    picture = st.camera_input("Take photo")
    
    if picture:
        img = Image.open(picture)
        img_np = np.array(img)
        
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(3.0)
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        results = model(enhanced, conf=0.4)
        annotated = results[0].plot()
        
        st.image([img_np, annotated], caption=['Live', 'Detection'])

st.balloons()
st.success("Production-ready hazard detection!")

