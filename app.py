import streamlit as st
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label
from scipy.ndimage import median_filter

# --- Load class colors from CSV ---
@st.cache_data
def load_class_colors(csv_path='class_dict.csv'):
    df = pd.read_csv(csv_path)
    return list(zip(df['r'], df['g'], df['b']))

color_map = load_class_colors()

# --- Load model ---
@st.cache_resource
def load_unet_model(model_path='deepglobe_unet_jay.keras'):
    model = load_model(model_path, compile=False)
    return model

model = load_unet_model()

# --- Convert class mask to RGB ---
def class_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        rgb[mask == i] = color
    return rgb

# --- Morphological postprocessing ---
def postprocess_mask(mask_class, min_size=64, hole_area=64):
    processed = np.zeros_like(mask_class)
    for cls in np.unique(mask_class):
        if cls == 0:
            continue
        binary = (mask_class == cls)
        binary = remove_small_objects(label(binary), min_size=min_size)
        binary = remove_small_holes(binary, area_threshold=hole_area)
        processed[binary] = cls
    return processed

# --- Additional simple smoothing postprocessing ---
def median_smoothing(mask_class, size=3):
    # Apply median filter class-wise to smooth boundaries
    smoothed = np.zeros_like(mask_class)
    for cls in np.unique(mask_class):
        binary = (mask_class == cls).astype(np.uint8)
        filtered = median_filter(binary, size=size)
        smoothed[filtered > 0] = cls
    return smoothed

# --- Streamlit UI ---
st.title("DeepGlobe Land Cover Segmentation with Postprocessing")

uploaded_file = st.file_uploader("Upload an RGB satellite image (recommended size: 224x224)", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize and normalize for model input
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_norm = img_resized / 255.0

    st.image(img_rgb, caption="Original Image", use_column_width=True)

    if st.button("Segment and Postprocess"):
        # Predict mask probabilities
        softmax_pred = model.predict(np.expand_dims(img_norm, 0))[0]  # shape (224, 224, num_classes)
        raw_mask = np.argmax(softmax_pred, axis=-1)

        # Morphological postprocessing
        morph_mask = postprocess_mask(raw_mask)

        # Optional median smoothing
        use_smoothing = st.checkbox("Apply Median Smoothing Postprocessing", value=False)
        if use_smoothing:
            smooth_mask = median_smoothing(morph_mask)
        else:
            smooth_mask = None

        # Show results side-by-side
        st.subheader("Segmentation Results")

        # Decide columns dynamically
        n_cols = 3 if smooth_mask is not None else 2
        cols = st.columns(n_cols)

        with cols[0]:
            st.text("Raw Prediction")
            st.image(class_to_rgb(raw_mask), use_column_width=True)

        with cols[1]:
            st.text("Morphological Postprocessing")
            st.image(class_to_rgb(morph_mask), use_column_width=True)

        if smooth_mask is not None:
            with cols[2]:
                st.text("Median Smoothing Postprocessing")
                st.image(class_to_rgb(smooth_mask), use_column_width=True)
