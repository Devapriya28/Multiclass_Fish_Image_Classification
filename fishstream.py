import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# Load your trained model
model = tf.keras.models.load_model(r"D:\DATA SCIENCE\CODE\git\project_5\best_fish_model.keras")

# Define class names (replace with your actual class names)
class_names = [
    'animal_fish', 'animal_fish_bass', 'fish_sea_food_black_sea_sprat',
    'fish_sea_food_gilt_head_bream', 'fish_sea_food_hourse_mackerel',
    'fish_sea_food_red_mullet', 'fish_sea_food_red_sea_bream',
    'fish_sea_food_sea_bass', 'fish_sea_food_shrimp',
    'fish_sea_food_striped_red_mullet', 'fish_sea_food_trout'
]

# Page configuration
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered", page_icon="🐠")

# Header
st.title("🐠 Fish Classification App")
st.markdown("Upload an image of a fish, and the model will predict its category with **confidence visualization** 📊")

# Function to predict the fish category
def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return predicted_class, confidence, scores

# Upload section
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', use_container_width=True)
    st.markdown("### 🧠 Classifying...")
    
    predicted_class, confidence, scores = predict(image)

    # Display main prediction
    st.success(f"✅ **Predicted Category:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    # Prepare data for visualization
    df = pd.DataFrame({
        'Fish Category': class_names,
        'Confidence (%)': [float(s) * 100 for s in scores]
    }).sort_values('Confidence (%)', ascending=True)

    # Trendy visualization using Plotly
    fig = px.bar(
        df,
        x='Confidence (%)',
        y='Fish Category',
        orientation='h',
        color='Confidence (%)',
        color_continuous_scale='teal',
        text_auto='.2f',
        title='Model Confidence per Class'
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 Tip: The longer the bar, the more confident the model is in that class.")
else:
    st.info("👆 Upload a fish image (JPG/PNG) to begin classification.")
