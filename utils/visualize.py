import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_images(original, noisy, cleaned, labels):
    """
    Displays the original, noisy, and cleaned versions of the images.
    Finally, saves the comparison visualization.
    """
    # Visualize each digit between 0-9
    for digit_index in range(10):
        # Get the indices of the relevant digit
        digit_indices = np.where(labels == digit_index)[0]
        
        # 1st image (original)
        plt.subplot(3, 10, digit_index + 1)
        plt.imshow(original[digit_indices[0]].reshape(28, 28), cmap='gray')
        plt.title(f'Original {digit_index}')
        plt.axis('off')
        
        # 2nd image (noisy)
        plt.subplot(3, 10, digit_index + 11)
        plt.imshow(noisy[digit_indices[0]].reshape(28, 28), cmap='gray')
        plt.title(f'Noisy {digit_index}')
        plt.axis('off')
        
        # 3rd image (cleaned)
        plt.subplot(3, 10, digit_index + 21)
        plt.imshow(cleaned[digit_indices[0]].reshape(28, 28), cmap='gray')
        plt.title(f'Cleaned {digit_index}')
        plt.axis('off')

    # Save the comparison visualization
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison.png'))
    plt.show()
