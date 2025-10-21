# ==========================================================
# 🐟 Multiclass Fish Classifier Streamlit App (Final Version)
# ==========================================================

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import json
import os

# ----------------------------------------------------------
# 1️⃣ Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="🐠 Fish Classifier", layout="centered", page_icon="🐟")

st.title("🐠 Fish Image Classification App")
st.markdown("""
Upload an image of a **fish or seafood 🐟🦐**  
and the trained deep learning model will predict its species with confidence visualization 📊
""")

# ----------------------------------------------------------
# 2️⃣ Load Model
# ----------------------------------------------------------
MODEL_PATH = r"D:\DATA SCIENCE\CODE\git\project_5\best_fish_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ----------------------------------------------------------
# 3️⃣ Load Class Labels
# ----------------------------------------------------------
LABEL_PATH = r"D:\DATA SCIENCE\CODE\git\project_5\class_indices.json"

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH) as f:
        class_indices = json.load(f)
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    st.sidebar.success("✅ Class labels loaded!")
else:
    st.sidebar.error("⚠️ class_indices.json not found.")
    class_names = []

# ----------------------------------------------------------
# 4️⃣ Prediction Function
# ----------------------------------------------------------
def predict(image: Image.Image):
    # Convert to grayscale (1 channel)
    image = image.convert("L")  # ✅ "L" = grayscale
    img = image.resize((224, 224))  # ✅ match model input size
    img_array = np.array(img).reshape(224, 224, 1) / 255.0  # ✅ reshape to (224,224,1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return predicted_class, confidence, scores


# ----------------------------------------------------------
# 5️⃣ File Upload
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.markdown("### 🧠 Classifying... Please wait")

    predicted_class, confidence, scores = predict(image)

    # ✅ Display Prediction
    st.success(f"**Predicted Category:** {predicted_class}")
    st.info(f"**Model Confidence:** {confidence:.2f}%")

    # ----------------------------------------------------------
    # 6️⃣ Confidence Visualization
    # ----------------------------------------------------------
    df = pd.DataFrame({
        "Fish Category": class_names,
        "Confidence (%)": [float(s) * 100 for s in scores]
    }).sort_values("Confidence (%)", ascending=True)

    fig = px.bar(
        df,
        x="Confidence (%)",
        y="Fish Category",
        orientation="h",
        color="Confidence (%)",
        color_continuous_scale="teal",
        text_auto=".2f",
        title="Model Confidence per Class"
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Tip: Longer bars = higher model confidence.")
else:
    st.info("👆 Upload a fish image (JPG/PNG) to start classification.")

# ----------------------------------------------------------
# 7️⃣ Footer
# ----------------------------------------------------------
st.markdown("""
---
👩‍💻 **Developed by:** Priyaa  
🧠 *Deep Learning Project — Multiclass Fish & Seafood Classification*  
🚀 *Powered by TensorFlow, Keras, and Streamlit*
""")
