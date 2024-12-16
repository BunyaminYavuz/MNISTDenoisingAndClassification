import numpy as np
import cv2

# Add Gaussian noise to an image
def add_gaussian_noise(image, noise_factor=0.5):
    noisy_image = image + noise_factor * np.random.randn(*image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image

# Apply Gaussian blur to denoise
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
