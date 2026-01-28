import streamlit as st
import numpy as np
from PIL import Image

from utils.fairface_predict import predict_race
from utils.mapping import map_nationality
from utils.emotion_predict import predict_emotion
from utils.age_predict import predict_age
from utils.dress_color import get_dress_color

st.set_page_config(page_title="Nationality Detection", layout="wide")

st.title("Nationality Detection & Emotion Prediction System")
st.caption(
    "⚠️ Note: Nationality prediction is inferred from facial features using FairFace race classification. "
    "Results may vary due to lighting, pose, and dataset bias."
)
uploaded = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    # Predictions
    race, conf = predict_race(img_np)
    nationality = map_nationality(race, conf)

    st.write(f"Race Predicted: {race}")
    st.write(f"Confidence: {conf*100:.2f}%")
    emotion = predict_emotion(img_np)

    with col2:
        st.subheader("Prediction Output")
        st.success(f"Nationality: {nationality}")
        st.info(f"Emotion: {emotion}")

        if nationality == "Indian":
            age = predict_age(img_np)
            dress = get_dress_color(img_np)
            st.warning(f"Age: {age}")
            st.warning(f"Dress Color: {dress}")

        elif nationality == "United States":
            age = predict_age(img_np)
            st.warning(f"Age: {age}")

        elif nationality == "African":
            dress = get_dress_color(img_np)
            st.warning(f"Dress Color: {dress}")

        else:
            st.write("No additional attributes required.")