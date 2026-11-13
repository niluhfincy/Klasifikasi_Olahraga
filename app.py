
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# KONFIGURASI APLIKASI
st.set_page_config(
    page_title="Sports Classification",
    page_icon="🏅",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "keras" / "best_custom_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

IMG_HEIGHT = 224
IMG_WIDTH = 224
TOP_K = 5


# LOAD MODEL DAN NAMA KELAS
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# PREPROCESSING GAMBAR
def preprocess_image(image):
    image = image.convert("RGB")

    image_array = np.array(image)

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = tf.image.resize_with_pad(
        image_tensor,
        IMG_HEIGHT,
        IMG_WIDTH
    )

    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    return image_tensor


# PREDIKSI GAMBAR
def predict_image(model, image, class_names):
    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        results.append({
            "class_name": class_names[index],
            "confidence": float(predictions[index]) * 100
        })

    return results


# TAMPILAN APLIKASI
st.title("🏅 100 Sports Classification")

st.write(
    "Aplikasi klasifikasi gambar menggunakan "
    "Custom Convolutional Neural Network (CNN)."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload gambar olahraga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("Klasifikasikan Gambar", type="primary"):

        try:
            with st.spinner("Model sedang menganalisis gambar..."):

                model = load_model()
                class_names = load_class_names()

                if len(class_names) != model.output_shape[-1]:
                    st.error(
                        "Jumlah kelas tidak sesuai dengan output model."
                    )
                    st.stop()

                results = predict_image(
                    model,
                    image,
                    class_names
                )

            st.success("Klasifikasi berhasil!")

            st.subheader("Hasil Prediksi")

            first_result = results[0]

            st.metric(
                label="Prediksi Utama",
                value=first_result["class_name"].title(),
                delta=f"{first_result['confidence']:.2f}%"
            )

            st.subheader("Top 5 Prediksi")

            for i, result in enumerate(results, start=1):

                st.write(
                    f"**{i}. {result['class_name'].title()}**"
                )

                st.progress(
                    min(result["confidence"] / 100, 1.0)
                )

                st.caption(
                    f"Confidence: {result['confidence']:.2f}%"
                )

        except Exception as error:
            st.error(f"Terjadi kesalahan: {error}")
