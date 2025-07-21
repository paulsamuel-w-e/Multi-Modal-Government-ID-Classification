import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="DocClassifier", layout="centered")
st.title("Document Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

model_choice = st.selectbox("Choose a model to run", ["OCR", "LayoutLMv3", "Fusion"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Calling API and running model..."):
            files = {"image": uploaded_file.getvalue()}

            if model_choice == "OCR":
                response = requests.post(f"{API_BASE}/ocr", files={"image": (uploaded_file.name, uploaded_file, "image/jpeg")})
            elif model_choice == "LayoutLMv3":
                response = requests.post(f"{API_BASE}/layoutlm")
            elif model_choice == "Fusion":
                response = requests.post(f"{API_BASE}/fusion")

            if response.status_code == 200:
                st.success("Prediction Complete ✅")
                st.json(response.json())
            else:
                st.error(f"Error {response.status_code}")
                st.json(response.json())