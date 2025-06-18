import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("location.keras")  # loads the highest-accuracy version

# Preprocess function
def preprocess_image(image):
    image = image.resize((350, 350))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🎬 Movie Genre Predictor")
uploaded_file = st.file_uploader("Upload a poster image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Poster', use_container_width=True)

    st.write("Making prediction...")
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]

    genres = [
        "Action", "Adventure", "Animation", "Biography", "Comedy",
        "Crime", "Documentary", "Drama", "Family", "Fantasy",
        "History", "Horror", "Music", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Sport", "Thriller", "War",
        "Western", "Short", "Film-Noir", "Reality-TV", "Talk-Show"
    ]

    st.subheader("Top 2 Predicted Genres:")

    top_k = 2
    top_idxs = np.argsort(prediction)[::-1][:top_k]

    #st.write("Prediction shape:", prediction.shape)
    #st.write("Genres list length:", len(genres))

    for idx in top_idxs:
        if idx < len(genres):
            st.write(f"{genres[idx]}: {round(float(prediction[idx]), 2)}")
        else:
            st.write(f"⚠️ Predicted index {idx} is out of range for genres list.")
