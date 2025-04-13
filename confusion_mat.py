import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("Confusion Matrix Checker")
st.image("com.jpg", caption="Reference Confusion Matrix")
st.write("Upload your confusion matrix image:")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is not None:
    uploaded_pil = Image.open(file).convert('RGB')
    uploaded_cv = cv2.cvtColor(np.array(uploaded_pil), cv2.COLOR_RGB2GRAY)

    ref_img = cv2.imread("com.jpg", cv2.IMREAD_GRAYSCALE)

    if ref_img.shape != uploaded_cv.shape:
        uploaded_cv = cv2.resize(uploaded_cv, (ref_img.shape[1], ref_img.shape[0]))

    comparison = ref_img == uploaded_cv
    num_equal_pixels = np.sum(comparison)
    total_pixels = ref_img.size

    sim = (num_equal_pixels / total_pixels) * 100

    st.write(f"Similarity: {sim:.2f}%")

    if sim >= 50:
        st.success("Your answer is correct!")
    else:
        st.error("Your answer is wrong.")