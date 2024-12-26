import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_images(original, noisy, cleaned, labels, selected_digits):
    """
    Displays the original, noisy, and cleaned versions of the images for selected digits.
    """
    # Visualize each selected digit
    for digit_index in selected_digits:
        # Get the indices of the relevant digit
        digit_indices = np.where(labels == digit_index)[0]
        
        if len(digit_indices) > 0:
            # 1st image (original)
            plt.subplot(3, len(selected_digits), selected_digits.index(digit_index) + 1)
            plt.imshow(original[digit_indices[0]].reshape(28, 28), cmap='gray')
            plt.title(f'Original {digit_index}')
            plt.axis('off')
            
            # 2nd image (noisy)
            plt.subplot(3, len(selected_digits), selected_digits.index(digit_index) + len(selected_digits) + 1)
            plt.imshow(noisy[digit_indices[0]].reshape(28, 28), cmap='gray')
            plt.title(f'Noisy {digit_index}')
            plt.axis('off')
            
            # 3rd image (cleaned)
            plt.subplot(3, len(selected_digits), selected_digits.index(digit_index) + 2 * len(selected_digits) + 1)
            plt.imshow(cleaned[digit_indices[0]].reshape(28, 28), cmap='gray')
            plt.title(f'Cleaned {digit_index}')
            plt.axis('off')
        else:
            print(f"No images found for digit {digit_index}")
    
    # Save the comparison visualization
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison.png'))
    plt.show()
