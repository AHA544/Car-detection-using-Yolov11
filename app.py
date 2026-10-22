import io
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
TRAINED_WEIGHTS = BASE / "models" / "best.pt"   # produced by train.py
CLASSES_FILE = BASE / "classes.txt"

st.set_page_config(page_title="AI Car Detection", page_icon="🎯", layout="wide")


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def load_class_names():
    """Optional readable names, one per line, in class-id order."""
    if CLASSES_FILE.exists():
        lines = CLASSES_FILE.read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    return None


# ---------- choose the model ----------
if TRAINED_WEIGHTS.exists():
    weights = str(TRAINED_WEIGHTS)
    model_label = "Your trained model (models/best.pt)"
else:
    weights = "yolo11n.pt"
    model_label = "Default YOLO11n (not trained on your data)"

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.selectbox("Inference image size", [512, 640, 800, 1024], index=1,
                         help="Use the same size you trained with, or larger for tiny objects.")
    st.caption(f"Model: {model_label}")
    if not TRAINED_WEIGHTS.exists():
        st.warning("models/best.pt not found. Run train.py first.")

st.title("AI Car Detection System")
st.write("Upload an image to detect cars using your trained YOLO model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, width="stretch")

    if st.button("Detect Objects", type="primary"):
        with st.spinner("Detecting objects..."):
            model = load_model(weights)
            result = model.predict(source=image, conf=confidence, imgsz=imgsz, verbose=False)[0]

            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            st.subheader("Detection Result")
            st.image(annotated, width="stretch")

            custom_names = load_class_names()

            def name_of(cid: int) -> str:
                if custom_names and cid < len(custom_names):
                    return custom_names[cid]
                return result.names[cid]

            detections = [
                {
                    "Object": name_of(int(b.cls[0])),
                    "Confidence (%)": round(float(b.conf[0]) * 100, 2),
                }
                for b in result.boxes
            ]

            st.subheader("Detection Summary")
            if detections:
                df = pd.DataFrame(detections)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Objects detected", len(df))
                    st.write("Count per class")
                    st.dataframe(df["Object"].value_counts().rename("Count"), width="stretch")
                with col2:
                    st.dataframe(df, width="stretch", hide_index=True)

                buffer = io.BytesIO()
                Image.fromarray(annotated).save(buffer, format="PNG")
                st.download_button("Download detected image", data=buffer.getvalue(),
                                   file_name="detected_objects.png", mime="image/png")
            else:
                st.warning("No objects detected. Try another image or lower the confidence threshold.")
