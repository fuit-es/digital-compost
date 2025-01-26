import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os

# Function to load images from a local directory and convert them to numpy arrays
def load_images_with_labels(directory):
    images = []
    labels = []  # List to store the filenames (labels)
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.webp'):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize((100, 100))  # Resize for consistency
            img = img.convert('RGB')
            images.append(np.array(img))
            labels.append(filename)  # Store the filename
    return images, labels

# Replace 'your_image_directory' with the path to your image directory
image_directory = 'img'
images, labels = load_images_with_labels(image_directory)

# Flatten images to create a dataset suitable for PCA
flattened_images = np.array([image.flatten() for image in images])

# Standardize the dataset
scaler = StandardScaler()
flattened_images_standardized = scaler.fit_transform(flattened_images)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(flattened_images)

# Plot the PCA results in 2D with labels
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(principal_components):
    plt.scatter(x, y)
    plt.text(x + 0.1, y + 0.1, labels[i], fontsize=9)  # Offset the text a bit for clarity

plt.title('PCA of Images with Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
