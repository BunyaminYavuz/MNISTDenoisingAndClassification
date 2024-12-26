import streamlit as st
import numpy as np
import tensorflow as tf
from utils.preprocess import add_gaussian_noise, gaussian_blur
from utils.visualize import visualize_images
from models.train_model import train_and_evaluate
import os

# Load the data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Streamlit UI
st.title("MNIST Image Enhancement and Accuracy Evaluation")

# Hyperparameter selection (epochs)
epochs = st.slider('Select Epochs', min_value=1, max_value=20, value=7)

# Rakamları seçme
selected_digits = st.multiselect("Select Digits to Visualize", options=list(range(10)), default=list(range(10)))

# Eğitim butonu
train_button = st.button("Train Model")

# Modelin eğitimini başlatma ve görsel sonuçları gösterme
if train_button:
    # Add noise
    X_train_noisy = np.array([add_gaussian_noise(img) for img in X_train.reshape(-1, 28, 28)])
    X_test_noisy = np.array([add_gaussian_noise(img) for img in X_test.reshape(-1, 28, 28)])

    # Apply Gaussian blur
    X_train_cleaned = np.array([gaussian_blur(img) for img in X_train_noisy])
    X_test_cleaned = np.array([gaussian_blur(img) for img in X_test_noisy])

    # Visualize selected digits
    visualize_images(X_train, X_train_noisy, X_train_cleaned, y_train, selected_digits)

    # Train models with the selected epochs
    st.write("Training with noisy data...")
    acc_noisy = train_and_evaluate(X_train_noisy, y_train, X_test_noisy, y_test, 'models/noisy_model.h5', epochs=epochs)
    st.write(f"Accuracy with noisy data: {acc_noisy:.2f}")

    st.write("Training with cleaned data...")
    acc_cleaned = train_and_evaluate(X_train_cleaned, y_train, X_test_cleaned, y_test, 'models/cleaned_model.h5', epochs=epochs)
    st.write(f"Accuracy with cleaned data: {acc_cleaned:.2f}")
    
    # Show results in the sidebar
    st.sidebar.header("Model Results")
    st.sidebar.write(f"Accuracy with noisy data: {acc_noisy:.2f}")
    st.sidebar.write(f"Accuracy with cleaned data: {acc_cleaned:.2f}")

    # Show the comparison image
    st.image('results/comparison.png', caption="Original vs Noisy vs Cleaned Images")

# ...existing code...

# Footer section
st.markdown("---")
st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 10px;
        background-color: #f1f1f1;
        border-top: 1px solid #eaeaea;
        margin-top: 20px;
    }
    .footer a {
        color: #0366d6;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
<div class="footer">
    <h4>Developed by <a href="https://www.linkedin.com/in/bunyaminyavuz/" target="_blank">Bunyamin Yavuz</a></h4>
    <p>GitHub Repository: <a href="https://github.com/BunyaminYavuz/MNISTimageEnhancementAndAccuracyEvaluation" target="_blank">MNIST Noise Reduction Project</a></p>
    <p>© 2025 Bunyamin Yavuz. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)