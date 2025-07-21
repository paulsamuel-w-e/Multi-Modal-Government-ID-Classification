#streamlit_ui/app.py

import streamlit as st
import requests
import sys

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="DocClassifier", layout="centered")
st.title("Document Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

model_choice = st.selectbox("Choose a model to run", ["OCR", "LayoutLMv3", "Fusion"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Calling API and running model..."):
            # Prepare the correct files payload
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            if model_choice == "OCR":
                response = requests.post(f"{API_BASE}/ocr", files=files)

            elif model_choice == "LayoutLMv3":
                # First call OCR (required preprocessing)
                requests.post(f"{API_BASE}/ocr", files=files)
                sys.stdout.flush()
                # Then call LayoutLMv3 with the same file
                response = requests.post(f"{API_BASE}/layoutlm", files=files)

            elif model_choice == "Fusion":
                requests.post(f"{API_BASE}/ocr", files=files)
                sys.stdout.flush()
                response = requests.post(f"{API_BASE}/fusion", files=files)

            if response.status_code == 200:
                st.success("Prediction Complete ✅")
                st.json(response.json())
            else:
                st.error(f"Error {response.status_code}")
                st.json(response.json())