
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Animal Detection AI",
    page_icon="🐾"
)

CLASS_NAMES = ["cats", "dogs", "elephamt", "lions", "tiger"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_detection_model.keras")

st.title("🐾 Animal Detection AI")
st.write("Upload an image to identify the animal.")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model file missing or could not be loaded.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an animal picture",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    if st.button("Detect Animal"):
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        index = int(np.argmax(predictions))

        st.subheader("Predicted Animal: " + CLASS_NAMES[index].title())
        st.write(f"Confidence: {predictions[index] * 100:.2f}%")
        st.warning(
            "This model was trained on only 10 images, "
            "so predictions may be inaccurate."
        )
