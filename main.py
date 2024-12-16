import numpy as np
from utils.preprocess import add_gaussian_noise, gaussian_blur
from utils.visualize import visualize_images
from models.train_model import train_and_evaluate

# Load the data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Add noise
X_train_noisy = np.array([add_gaussian_noise(img) for img in X_train.reshape(-1, 28, 28)])
X_test_noisy = np.array([add_gaussian_noise(img) for img in X_test.reshape(-1, 28, 28)])

# Apply Gaussian blur
X_train_cleaned = np.array([gaussian_blur(img) for img in X_train_noisy])
X_test_cleaned = np.array([gaussian_blur(img) for img in X_test_noisy])

# Visualize
visualize_images(X_train, X_train_noisy, X_train_cleaned, y_train)

# Train and evaluate
acc_noisy = train_and_evaluate(X_train_noisy, y_train, X_test_noisy, y_test, 'models/noisy_model.h5')
acc_cleaned = train_and_evaluate(X_train_cleaned, y_train, X_test_cleaned, y_test, 'models/cleaned_model.h5')

print(f"Accuracy with noisy data: {acc_noisy:.2f}")
print(f"Accuracy with cleaned data: {acc_cleaned:.2f}")
