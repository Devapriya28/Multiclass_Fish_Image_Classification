# ==========================================================
# 🐟 Final Streamlit Fish Classifier 
# ==========================================================
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import json
import os

# ---------------- Page Setup ----------------
st.set_page_config(page_title="🐠 Fish Classifier", layout="centered", page_icon="🐟")

# ---------------- Custom CSS (trendy colourful background + glass cards) ----------------
st.markdown(
    """
    <style>
    :root{
      --primary-gradient-start: #00c6ff;  /* change these two for different gradient */
      --primary-gradient-end: #0072ff;
      --accent: #ff6b6b;
      --card-bg: rgba(255,255,255,0.72);
      --glass-border: rgba(255,255,255,0.35);
      --muted: rgba(0,0,0,0.55);
    }

    /* whole page gradient */
    [data-testid="stAppViewContainer"]{
      background: linear-gradient(135deg, var(--primary-gradient-start) 0%, var(--primary-gradient-end) 50%, #8e44ad 100%);
      min-height: 100vh;
      background-attachment: fixed;
      padding: 2rem;
    }

    /* center main content card */
    [data-testid="stMain"] > div.block-container{
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.40);
      backdrop-filter: blur(6px) saturate(120%);
      border: 1px solid rgba(255,255,255,0.06);
    }

    /* sidebar glass */
    [data-testid="stSidebar"]{
      background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 20px rgba(2,6,23,0.35);
      border: 1px solid var(--glass-border);
    }

    /* headings */
    h1, h2, h3 {
      font-family: 'Segoe UI', Roboto, "Helvetica Neue", Arial, sans-serif;
      color: white;
      text-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }

    /* card style for uploaded image & results */
    .result-card{
      background: var(--card-bg);
      padding: 14px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.15);
      box-shadow: 0 6px 18px rgba(0,0,0,0.18);
      color: #042a2b;
    }

    /* buttons */
    .stButton>button {
      background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.14);
      padding: 8px 14px;
      border-radius: 10px;
      color: white;
      box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }

    /* smaller text / footer */
    .footer {
      color: rgba(255,255,255,0.9);
      text-align: center;
      padding-top: 10px;
      font-size: 14px;
      opacity: 0.95;
    }

    /* plotly background transparent */
    .js-plotly-plot .plotly, .js-plotly-plot svg {
      background: transparent !important;
    }

    /* responsive tweaks */
    @media (max-width: 600px){
      [data-testid="stMain"] > div.block-container{ padding: 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Title & Intro ----------------
st.title("🐠 Fish Image Classification App")
st.markdown(
    """
    <div style="margin-top:-10px;">
    <p style="color:rgba(255,255,255,0.95); font-size:16px;">
    Upload an image of a <strong>fish or seafood 🦐🐟</strong>. This app predicts the category and shows a stylish confidence chart.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Load Model ----------------
MODEL_PATH = r"D:\DATA SCIENCE\CODE\git\project_5\best_fish_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ---------------- Load Class Labels ----------------
LABEL_PATH = r"D:\DATA SCIENCE\CODE\git\project_5\class_indices.json"

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH) as f:
        class_indices = json.load(f)
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    st.sidebar.success("✅ Class labels loaded successfully!")
else:
    st.sidebar.error("⚠️ class_indices.json not found.")
    class_names = []

# ---------------- Confidence Calibration ----------------
def calibrate_predictions(predictions, temperature=0.7):
    logits = tf.math.log(predictions + 1e-9)
    scaled_logits = logits / temperature
    return tf.nn.softmax(scaled_logits).numpy()

# ---------------- Prediction Function ----------------
def predict(image: Image.Image):
    image = image.convert("L")
    img = image.resize((224, 224))
    img_array = np.array(img).reshape(224, 224, 1) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    scores = calibrate_predictions(predictions)[0]
    predicted_class = class_names[np.argmax(scores)] if class_names else "Unknown"
    confidence = 100 * np.max(scores)
    return predicted_class, confidence, scores

# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader("📤 Upload a fish image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # display inside a styled div
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.markdown("### 🧠 Classifying... Please wait", unsafe_allow_html=True)

    predicted_class, confidence, scores = predict(image)

    # Results
    st.success(f"**Predicted Category:** {predicted_class}")
    st.info(f"**Model Confidence:** {confidence:.2f}%")

    # Visualization dataframe and plotly bar
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
        color_continuous_scale=px.colors.sequential.Teal,
        text_auto=".2f",
        title="Model Confidence per Class",
        template="plotly_white"
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color="#042a2b"),
        height=560,
        margin=dict(l=120, r=30, t=70, b=30)
    )
    fig.update_traces(marker_line_color='rgba(255,255,255,0.08)', marker_line_width=1)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if confidence < 40:
        st.warning("⚠️ Model is not very confident — image may be unclear or similar to another species.")
else:
    st.info("👆 Upload a fish image to start classification.")

# ---------------- Footer ----------------
st.markdown(
    """
    <div class="footer">
      --- <br>
      👩‍💻 <strong>Developed by:</strong> Devaa  &nbsp; • &nbsp; 🧠 Deep Learning Project — Multiclass Fish Classification
    </div>
    """,
    unsafe_allow_html=True,
)
