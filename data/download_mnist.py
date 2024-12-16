import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Load MNIST data
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist['data'].values
    target = mnist['target'].astype(int).values
    return data, target

# 2. Save the data
def save_data(X_train, y_train, X_test, y_test):
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

if __name__ == "__main__":
    # Load and split the data
    data, target = load_mnist_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Save data to files
    save_data(X_train, y_train, X_test, y_test)
    print("Datasets saved successfully!")
