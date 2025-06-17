import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_model.keras")  # loads the highest-accuracy version


# Preprocess function
def preprocess_image(image):
    # Resize to match the model
    image = image.resize((350, 350))
    # Convert to array and normalize
    image = np.array(image) / 255.0
    # Add batch dimension
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

    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
    st.subheader("Top 2 Predicted Genres:")
    top_idxs = np.argsort(prediction)[::-1][:2]
    for idx in top_idxs:
        st.write(f"{genres[idx]}: {round(float(prediction[idx]), 2)}")