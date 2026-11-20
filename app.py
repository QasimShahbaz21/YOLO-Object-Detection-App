import os
import tempfile

# Ensure Ultralytics settings use temp directory to prevent any permission issues
os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import glob
from ultralytics import YOLO
import io

# Page Configuration
st.set_page_config(
    page_title="Traffic Signal & Sign AI Detector",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Model (cached so it only loads once)
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the application directory!")
        st.stop()
    return YOLO(model_path)

model = load_yolo_model("best.pt")

# Title and Description
st.markdown('<div class="main-header">🚦 Traffic Signal & Sign AI Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect traffic lights (Red, Green, Yellow) and road signs in real-time with fine-tuned YOLOv8.</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Reverse.svg", width=180)
st.sidebar.header("⚙️ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.35,
    step=0.05,
    help="Filter out detections with a confidence score below this threshold."
)

with st.sidebar.expander("Advanced Settings"):
    iou_threshold = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression."
    )

st.sidebar.divider()
st.sidebar.header("📁 Input Source")
input_option = st.sidebar.radio("Choose Input:", ["Upload an Image", "Use a Sample Image"])

input_image = None
image_name = "image"

if input_option == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
else:
    sample_files = glob.glob("sample_images/*.jpg") + glob.glob("sample_images/*.png")
    if sample_files:
        sample_names = [os.path.basename(f) for f in sample_files]
        selected_sample = st.sidebar.selectbox("Select Sample Image:", sample_names)
        sample_path = os.path.join("sample_images", selected_sample)
        input_image = Image.open(sample_path).convert("RGB")
        image_name = selected_sample
    else:
        st.sidebar.info("No sample images found in `sample_images/` folder.")

# Sidebar Model Info
st.sidebar.divider()
st.sidebar.markdown(f"""
**Model Details:**
- **Architecture:** YOLOv8 Nano (`yolov8n.pt`)
- **Total Classes:** {len(model.names)}
- **Trained Epochs:** 20
""")

# Main Content Area
if input_image is not None:
    # Run YOLO Prediction
    with st.spinner("Running YOLOv8 inference..."):
        results = model.predict(
            source=input_image,
            conf=conf_threshold,
            iou=iou_threshold
        )
        res = results[0]

    # Render Side-by-Side Images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Result")
        # Plot annotated image (returns BGR, convert to RGB)
        res_bgr = res.plot()
        res_rgb = res_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(res_rgb)
        st.image(annotated_pil, use_container_width=True)

    # Detections Extraction
    boxes = res.boxes
    num_detections = len(boxes)

    st.divider()

    # Metrics Summary Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Objects Detected", num_detections)
    with m_col2:
        unique_classes = len(set([int(b.cls[0].item()) for b in boxes])) if num_detections > 0 else 0
        st.metric("Unique Classes", unique_classes)
    with m_col3:
        max_conf = max([float(b.conf[0].item()) for b in boxes]) if num_detections > 0 else 0.0
        st.metric("Highest Confidence", f"{max_conf * 100:.1f}%" if num_detections > 0 else "N/A")

    # Table of Detected Objects
    st.subheader("📋 Detected Objects & Scores")

    if num_detections == 0:
        st.warning(f"No objects detected at confidence threshold >= {conf_threshold:.2f}. Try lowering the slider in the sidebar!")
    else:
        table_data = []
        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [round(val, 1) for val in box.xyxy[0].tolist()]

            table_data.append({
                "ID": idx,
                "Class": class_name,
                "Confidence": f"{confidence * 100:.2f}%",
                "Bounding Box [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]"
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download Annotated Image
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=byte_im,
            file_name=f"detected_{image_name}",
            mime="image/jpeg"
        )
else:
    # Placeholder state when no image is loaded
    st.info("👈 Please upload an image or select a sample image from the sidebar to begin object detection!")
    st.markdown("""
    ### About this Project:
    - **Classes:** Detects 39 classes including traffic signals (Red, Green, Yellow) and highway/city road signs.
    - **Model:** Fine-tuned `yolov8n.pt` for 20 epochs with mAP50 of **85.4%**.
    - **Interface:** Allows interactive confidence adjustment and instant bounding box visualization.
    """)
